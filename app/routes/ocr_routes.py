"""OCR 与 PS 检测相关路由"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from app.database import get_all_ps_results, get_ps_result, is_db_available
from app.fraud.ocr_service import ocr_service
from app.fraud.ps_service import ps_detection_service
from app.schemas import OcrListItem, OcrListResponse
from app.utils import get_id_to_filename, get_image_path_by_id, scan_upload_ids

logger = logging.getLogger(__name__)
router = APIRouter()

_STATIC_DIR = Path(__file__).parent.parent.parent / "static"


@router.get("/ocr", response_class=FileResponse)
async def ocr_page():
    """OCR 清单页面"""
    path = _STATIC_DIR / "ocr.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ocr.html not found")
    return FileResponse(str(path))


@router.get("/ocr_list", response_model=OcrListResponse)
async def ocr_list():
    """
    返回所有已上传证件及其 OCR 识别状态/结果。

    - 已运行 OCR 的证件返回结构化字段与原始文本
    - 未运行的返回 ocr_done=false
    - 按 entity_id 升序排列
    """
    if not is_db_available():
        raise HTTPException(status_code=503, detail="数据库不可用，请检查 MySQL 连接")

    id_to_name = get_id_to_filename()
    for k, v in scan_upload_ids().items():
        if k not in id_to_name:
            id_to_name[k] = v

    ocr_cache = ocr_service.get_all_cached()
    ps_cache = get_all_ps_results()

    items = []
    for id_str, filename in sorted(id_to_name.items(), key=lambda x: int(x[0])):
        cached = ocr_cache.get(id_str)
        ocr_done = bool(cached) and "_ocr_error" not in cached
        fields = {k: v for k, v in (cached or {}).items() if not k.startswith("_")} if ocr_done else None
        raw_text = cached.get("_raw_text") if cached else None
        items.append(OcrListItem(
            id=id_str,
            filename=filename,
            ocr_done=ocr_done,
            fields=fields,
            raw_text=raw_text,
            ps_data=ps_cache.get(id_str),
        ))

    return OcrListResponse(
        items=items,
        total=len(items),
        ocr_done_count=sum(1 for i in items if i.ocr_done),
    )


@router.get("/ocr/{entity_id}")
async def run_ocr_single(entity_id: int):
    """
    对单张证件执行 OCR（已有缓存则直接返回，不重复计算）。

    返回结构化字段与原始识别文本。
    """
    img_path = get_image_path_by_id(entity_id)
    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image not found for id {entity_id}")
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

    fields = ocr_service.extract_and_cache(entity_id, image)
    return {
        "id": str(entity_id),
        "ocr_done": bool(fields) and "_ocr_error" not in fields,
        "fields": {k: v for k, v in fields.items() if not k.startswith("_")},
        "raw_text": fields.get("_raw_text"),
    }


@router.post("/ocr/{entity_id}/rerun")
async def rerun_ocr_single(entity_id: int):
    """强制重新运行指定证件的 OCR（忽略已有缓存）。"""
    img_path = get_image_path_by_id(entity_id)
    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image not found for id {entity_id}")
    ocr_service.clear_cache(entity_id)

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

    fields = ocr_service.extract_and_cache(entity_id, image)
    return {
        "id": str(entity_id),
        "ocr_done": bool(fields) and "_ocr_error" not in fields,
        "fields": {k: v for k, v in fields.items() if not k.startswith("_")},
        "raw_text": fields.get("_raw_text"),
    }


@router.get("/ps_detect/{entity_id}")
async def ps_detect_single(entity_id: int):
    """
    对指定证件运行 PS/篡改鉴别（已有缓存则直接返回）。
    鉴别通过多模态 API 完成，与 OCR 同步提取。
    """
    if not is_db_available():
        raise HTTPException(status_code=503, detail="数据库不可用，请检查 MySQL 连接")

    cached = get_ps_result(entity_id)
    if cached is not None:
        return {"id": str(entity_id), **cached}

    img_path = get_image_path_by_id(entity_id)
    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image not found for id {entity_id}")
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

    result = ps_detection_service.detect_and_save(entity_id, image)
    return {"id": str(entity_id), **result}


@router.post("/ps_detect/{entity_id}/rerun")
async def ps_redetect_single(entity_id: int):
    """强制重新运行 PS 鉴别（忽略缓存）"""
    img_path = get_image_path_by_id(entity_id)
    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image not found for id {entity_id}")
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

    result = ps_detection_service.redetect_and_save(entity_id, image)
    return {"id": str(entity_id), **result}
