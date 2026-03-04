"""欺诈检测相关路由：单张分析、批量分析、交叉核验"""
import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from app.database import (
    is_db_available,
    save_cross_validate_run,
    save_forensic_result,
    save_ocr_result,
    save_single_analyze_issues,
)
from app.fraud.forensic_service import forensic_service
from app.fraud.ocr_service import ocr_service
from app.fraud.rule_engine import rule_engine
from app.schemas import (
    AnalyzeResponse,
    CrossValidateResponse,
    FaceInfo,
    FraudIssue,
    OcrFields,
)
from app.utils import get_id_to_filename, get_image_path_by_id, scan_upload_ids
from app.vector_service import vector_service

logger = logging.getLogger(__name__)
router = APIRouter()

_STATIC_DIR = Path(__file__).parent.parent.parent / "static"


@router.get("/analyze/{entity_id}", response_model=AnalyzeResponse)
async def analyze_certificate(entity_id: int):
    """
    单张证件全面分析

    对指定证件执行：
    - OCR文字提取 + 字段解析（缓存，重复调用秒返回）
    - 人脸属性检测（年龄、性别）
    - 图像取证（ELA、照片边缘、印章真实性）
    - 单张规则校验（学制、年份逻辑、年龄性别）
    """
    if not is_db_available():
        raise HTTPException(status_code=503, detail="数据库不可用，请检查 MySQL 连接")

    img_path = get_image_path_by_id(entity_id)
    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image not found for id {entity_id}")

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

    id_to_name = get_id_to_filename() or scan_upload_ids()
    filename = id_to_name.get(str(entity_id))

    # 1. 人脸属性
    face_info_raw = vector_service.get_face_attributes(image)
    face_info = None
    face_bbox = None
    if face_info_raw:
        face_info = FaceInfo(
            age=face_info_raw.get("age"),
            gender=face_info_raw.get("gender"),
            det_score=face_info_raw.get("det_score"),
            bbox=face_info_raw.get("bbox"),
        )
        bbox_raw = face_info_raw.get("bbox")
        if bbox_raw and len(bbox_raw) == 4:
            face_bbox = tuple(bbox_raw)

    # 2. OCR（带缓存）
    try:
        raw_fields = ocr_service.extract_and_cache(entity_id, image)
        ocr_fields = OcrFields(
            school=raw_fields.get("school"),
            principal=raw_fields.get("principal"),
            grad_year=raw_fields.get("grad_year"),
            enrollment_year=raw_fields.get("enrollment_year"),
            issue_year=raw_fields.get("issue_year"),
            cert_no=raw_fields.get("cert_no"),
            gender=raw_fields.get("gender"),
            birth_year=raw_fields.get("birth_year"),
        )
    except Exception as e:
        logger.warning("OCR failed for id=%s: %s", entity_id, e)
        raw_fields = {}
        ocr_fields = None

    # 3. 图像取证
    try:
        forensic_result = forensic_service.full_analysis(image, face_bbox)
    except Exception as e:
        logger.warning("Forensic analysis failed for id=%s: %s", entity_id, e)
        forensic_result = {"error": str(e)}

    # 4. 单张规则校验
    rule_issues_raw = rule_engine.check_single(raw_fields, face_info_raw)
    rule_issues = [FraudIssue(**i) for i in rule_issues_raw]

    # 5. 汇总风险
    forensic_flags = forensic_result.get("risk_flags", [])
    all_flags = forensic_flags + [i.rule for i in rule_issues]
    risk_level = (
        "高" if any(i.severity == "高" for i in rule_issues) or len(forensic_flags) >= 2
        else ("中" if rule_issues or forensic_flags else "低")
    )

    # save_ocr_result 不再重复调用：extract_and_cache() 内部已写入完整字段
    # 再调用一次 save_ocr_result 会用字段不完整的旧版本覆盖完整数据
    save_forensic_result(entity_id, forensic_result)
    save_single_analyze_issues(entity_id, [i.model_dump() for i in rule_issues])

    return AnalyzeResponse(
        entity_id=str(entity_id),  # 转字符串，防止 JS 大整数精度丢失
        filename=filename,
        ocr_fields=ocr_fields,
        face_info=face_info,
        forensic=forensic_result,
        rule_issues=rule_issues,
        risk_level=risk_level,
        risk_flags=all_flags,
    )


@router.get("/cross_validate", response_model=CrossValidateResponse)
async def cross_validate(
    run_phash: bool = Query(default=False, description="是否同时运行感知哈希区域比对（较慢）"),
):
    """
    批量交叉核验

    对所有已上传证件：
    1. 运行OCR提取结构化字段（未缓存的会实时处理）
    2. 执行跨证件规则校验
    3. 可选：感知哈希比对（照片/印章/底部区域雷同检测）
    """
    if not is_db_available():
        raise HTTPException(status_code=503, detail="数据库不可用，请检查 MySQL 连接")

    id_to_name = get_id_to_filename()
    if not id_to_name:
        id_to_name = scan_upload_ids()
    if not id_to_name:
        return CrossValidateResponse(
            total_certs=0,
            ocr_success=0,
            issues=[],
            issue_count=0,
            high_severity_count=0,
            message="尚未上传任何证件",
        )

    # 按需加载图片（避免预加载所有图片导致 OOM）
    id_path_map: dict = {}
    for id_str in id_to_name:
        try:
            entity_id = int(id_str)
            img_path = get_image_path_by_id(entity_id)
            if img_path:
                id_path_map[entity_id] = img_path
        except Exception as e:
            logger.warning("Failed to locate image id=%s: %s", id_str, e)

    def _load_images(id_path: dict) -> dict:
        """按需打开图片，跳过无法读取的"""
        result = {}
        for eid, path in id_path.items():
            try:
                result[eid] = Image.open(path).convert("RGB")
            except Exception as e:
                logger.warning("Failed to open image id=%s: %s", eid, e)
        return result

    try:
        # run_batch 只对无缓存的 ID 调用 OCR API，先传路径映射转图片
        id_image_map_for_ocr = _load_images(id_path_map)
        all_fields = ocr_service.run_batch(id_image_map_for_ocr)
        ocr_success = sum(1 for f in all_fields.values() if not f.get("_ocr_error"))
    except RuntimeError as e:
        logger.warning("OCR unavailable: %s", e)
        all_fields = ocr_service.get_all_cached()
        ocr_success = len(all_fields)
        id_image_map_for_ocr = {}

    issues_raw = rule_engine.check_batch(all_fields)

    if run_phash and id_path_map:
        try:
            # phash 批量比对时才按需加载全量图片
            id_image_map_phash = _load_images(id_path_map)
            phash_issues = forensic_service.compare_regions_batch(id_image_map_phash)
            issues_raw.extend(phash_issues)
        except Exception as e:
            logger.warning("Phash compare failed: %s", e)

    issues = [FraudIssue(**i) for i in issues_raw]
    high_count = sum(1 for i in issues if i.severity == "高")
    msg = (
        f"发现 {len(issues)} 个问题（高风险 {high_count} 个）"
        if issues else "未发现跨证件逻辑矛盾"
    )

    save_cross_validate_run(
        total_certs=len(id_to_name),
        ocr_success=ocr_success,
        issues=issues_raw,
        message=msg,
    )

    return CrossValidateResponse(
        total_certs=len(id_to_name),
        ocr_success=ocr_success,
        issues=issues,
        issue_count=len(issues),
        high_severity_count=high_count,
        message=msg,
    )


@router.get("/admin", response_class=FileResponse)
async def admin_page():
    """运营管理操作面板"""
    path = _STATIC_DIR / "admin.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="admin.html not found")
    return FileResponse(str(path))


@router.post("/batch_analyze")
async def batch_analyze():
    """
    批量OCR + 图像取证 + 规则校验（SSE流式推送进度）

    对所有已上传证件逐条执行分析，实时推送进度事件。
    响应格式：text/event-stream
    每条事件 JSON 字段：done, total, id, filename, status, risk_level, error
    """
    if not is_db_available():
        raise HTTPException(status_code=503, detail="数据库不可用，请检查 MySQL 连接")

    id_to_name = get_id_to_filename()
    if not id_to_name:
        id_to_name = scan_upload_ids()

    ids = sorted(int(k) for k in id_to_name.keys())
    total = len(ids)

    async def event_stream():
        yield f"data: {json.dumps({'done': 0, 'total': total, 'status': 'start'}, ensure_ascii=False)}\n\n"

        for idx, entity_id in enumerate(ids, start=1):
            filename = id_to_name.get(str(entity_id))
            try:
                img_path = get_image_path_by_id(entity_id)
                if not img_path:
                    yield f"data: {json.dumps({'done': idx, 'total': total, 'id': entity_id, 'filename': filename, 'status': 'skip', 'error': 'image not found'}, ensure_ascii=False)}\n\n"
                    continue

                image = Image.open(img_path).convert("RGB")

                face_info_raw = vector_service.get_face_attributes(image)
                face_bbox = None
                if face_info_raw:
                    bbox_raw = face_info_raw.get("bbox")
                    if bbox_raw and len(bbox_raw) == 4:
                        face_bbox = tuple(bbox_raw)

                try:
                    raw_fields = ocr_service.extract_and_cache(entity_id, image)
                except Exception as e:
                    logger.warning("OCR failed for id=%s: %s", entity_id, e)
                    raw_fields = {}

                try:
                    forensic_result = forensic_service.full_analysis(image, face_bbox)
                except Exception as e:
                    logger.warning("Forensic failed for id=%s: %s", entity_id, e)
                    forensic_result = {}

                rule_issues_raw = rule_engine.check_single(raw_fields, face_info_raw)

                save_ocr_result(entity_id, raw_fields)
                save_forensic_result(entity_id, forensic_result)
                save_single_analyze_issues(entity_id, rule_issues_raw)

                forensic_flags = forensic_result.get("risk_flags", [])
                risk_level = (
                    "高" if any(i.get("severity") == "高" for i in rule_issues_raw) or len(forensic_flags) >= 2
                    else ("中" if rule_issues_raw or forensic_flags else "低")
                )

                yield f"data: {json.dumps({'done': idx, 'total': total, 'id': entity_id, 'filename': filename, 'status': 'ok', 'risk_level': risk_level}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error("batch_analyze error for id=%s: %s", entity_id, e)
                yield f"data: {json.dumps({'done': idx, 'total': total, 'id': entity_id, 'filename': filename, 'status': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

            await asyncio.sleep(0)

        yield f"data: {json.dumps({'done': total, 'total': total, 'status': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
