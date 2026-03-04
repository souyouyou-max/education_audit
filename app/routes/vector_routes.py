"""向量相关路由：上传、搜索、聚类"""
import logging
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.cluster_service import cluster_service
from app.config import settings
from app.database import is_db_available, save_certificate
from app.milvus_client import milvus_client
from app.schemas import (
    BatchUploadItem,
    BatchUploadResponse,
    ClusterResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.utils import (
    get_image_path_by_id,
    load_image_from_bytes,
    save_image_by_id,
    scan_upload_ids,
    validate_image_file,
)
from app.vector_service import vector_service

logger = logging.getLogger(__name__)
router = APIRouter()

_STATIC_DIR = Path(__file__).parent.parent.parent / "static"


@router.get("/")
async def root():
    """根路径"""
    return {
        "message": "学历证件相似度检测系统",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - 批量上传学历图片（支持多文件）",
            "search": "POST /search - 相似度搜索",
            "cluster": "GET /cluster - 批量聚类分析",
            "view": "GET /view - 分组图片查看（Web 页面，支持左右滑动）",
        },
    }


@router.get("/view", response_class=FileResponse)
async def view_groups_page():
    """分组图片查看 Web 页面（左右滑动查看各分组）"""
    path = _STATIC_DIR / "groups.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="groups.html not found")
    return FileResponse(str(path))


@router.get("/search", response_class=FileResponse)
async def search_page():
    """图片相似搜索 Web 页面"""
    path = _STATIC_DIR / "search.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="search.html not found")
    return FileResponse(str(path))


@router.get("/api/image/{entity_id}", response_class=FileResponse)
async def get_image_by_id(entity_id: int):
    """根据实体 id 返回已保存的图片。用于分组查看页展示。"""
    path = get_image_path_by_id(entity_id)
    if not path:
        logger.warning(
            "Image not found for id=%s (looked in %s)",
            entity_id,
            os.path.abspath(settings.UPLOAD_DIR),
        )
        raise HTTPException(status_code=404, detail=f"Image not found for id {entity_id}")
    return FileResponse(path)


@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        stats = milvus_client.get_collection_stats()
        mysql_status = "connected" if is_db_available() else "unavailable"
        return {
            "status": "healthy",
            "milvus": "connected",
            "mysql": mysql_status,
            "image_count": len(scan_upload_ids()),
            **stats,
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_images(files: List[UploadFile] = File(..., description="一个或多个学历图片")):
    """
    批量上传学历图片

    - 支持一次上传多张图片（multipart/form-data，多文件字段名均为 files）
    - 每张图提取图像向量、人脸向量、模板向量并存入 Milvus
    - 返回每张图的上传结果（id、是否检测到人脸、成功/失败及错误信息）
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")
    items: List[BatchUploadItem] = []
    success_count = 0
    fail_count = 0
    for file in files:
        filename = file.filename or "unknown"
        try:
            file_content = await file.read()
            if not validate_image_file(file_content, filename):
                items.append(BatchUploadItem(
                    filename=filename,
                    success=False,
                    error="Invalid image file. Supported formats: JPG, JPEG, PNG, BMP. Max size: 10MB",
                ))
                fail_count += 1
                continue
            image = load_image_from_bytes(file_content)
            logger.info("Extracting vectors from image: %s", filename)
            image_vector, face_vector, template_vector = vector_service.extract_all_vectors(image)
            has_face = face_vector is not None and not all(v == 0.0 for v in face_vector)
            inserted_id = milvus_client.insert(
                image_vector=image_vector,
                face_vector=face_vector,
                template_vector=template_vector,
            )
            logger.info("Image uploaded successfully: %s -> id=%s", filename, inserted_id)
            save_image_by_id(file_content, inserted_id, filename)
            save_certificate(inserted_id, filename, has_face)
            items.append(BatchUploadItem(
                filename=filename,
                id=inserted_id,
                has_face=has_face,
                success=True,
            ))
            success_count += 1
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error uploading image %s: %s", filename, e)
            items.append(BatchUploadItem(filename=filename, success=False, error=str(e)))
            fail_count += 1
    message = f"成功 {success_count} 张，失败 {fail_count} 张" if (success_count + fail_count) > 1 else (
        "上传成功" if success_count == 1 else "上传失败"
    )
    return BatchUploadResponse(
        items=items,
        success_count=success_count,
        fail_count=fail_count,
        message=message,
    )


@router.post("/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """
    相似度搜索

    支持通过 ID 或向量进行搜索
    """
    query_vector = None
    query_id = None

    if request.id is not None:
        query_id = request.id
        data = milvus_client.get_by_id(query_id)
        if not data:
            raise HTTPException(status_code=404, detail=f"Record with ID {request.id} not found")
        _valid_fields = {"image_vector", "face_vector", "template_vector"}
        if request.vector_field not in _valid_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vector_field: {request.vector_field}. Must be one of: {', '.join(sorted(_valid_fields))}",
            )
        query_vector = data[request.vector_field]
    elif request.image_vector:
        query_vector = request.image_vector
        request.vector_field = "image_vector"
    elif request.face_vector:
        query_vector = request.face_vector
        request.vector_field = "face_vector"
    elif request.template_vector:
        query_vector = request.template_vector
        request.vector_field = "template_vector"
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'id' or one of 'image_vector', 'face_vector', 'template_vector'",
        )

    expected_dims = {
        "image_vector": settings.IMAGE_VECTOR_DIM,
        "face_vector": settings.FACE_VECTOR_DIM,
        "template_vector": settings.TEMPLATE_VECTOR_DIM,
    }
    if len(query_vector) != expected_dims[request.vector_field]:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension mismatch. Expected {expected_dims[request.vector_field]}, got {len(query_vector)}",
        )

    logger.info("Searching with vector_field: %s, top_k: %s", request.vector_field, request.top_k)
    results = milvus_client.search(
        vector_field=request.vector_field,
        query_vector=query_vector,
        top_k=request.top_k,
    )

    formatted_results = [
        SearchResultItem(
            id=r["id"],
            score=r["score"],
            distance=r["distance"],
            abnormal=r["score"] > settings.SIMILARITY_THRESHOLD,
        )
        for r in results
    ]
    return SearchResponse(query_id=query_id, results=formatted_results, total=len(formatted_results))


@router.get("/cluster", response_model=ClusterResponse)
async def cluster_analysis(vector_field: str = "image_vector"):
    """
    批量聚类分析

    - 读取全部向量
    - 使用 DBSCAN 聚类
    - 返回分组结果和异常组
    """
    if vector_field not in ["image_vector", "face_vector", "template_vector"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vector_field: {vector_field}. Must be one of: image_vector, face_vector, template_vector",
        )

    logger.info("Starting clustering with vector_field: %s", vector_field)

    if vector_field == "image_vector":
        result = cluster_service.cluster_by_image_vector()
    elif vector_field == "face_vector":
        result = cluster_service.cluster_by_face_vector()
    else:
        result = cluster_service.cluster_by_template_vector()

    return ClusterResponse(**result)


@router.post("/search_by_image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    vector_field: str = Form(default="image_vector"),
    top_k: int = Form(default=5),
):
    """通过上传图片进行相似度搜索"""
    file_content = await file.read()
    if not validate_image_file(file_content, file.filename):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image = load_image_from_bytes(file_content)
    image_vector, face_vector, template_vector = vector_service.extract_all_vectors(image)

    _vec_map = {"image_vector": image_vector, "face_vector": face_vector, "template_vector": template_vector}
    if vector_field not in _vec_map:
        raise HTTPException(status_code=400, detail=f"Invalid vector_field: {vector_field}. Must be one of: image_vector, face_vector, template_vector")
    query_vector = _vec_map[vector_field]

    results = milvus_client.search(vector_field=vector_field, query_vector=query_vector, top_k=top_k)

    formatted_results = [
        SearchResultItem(
            id=r["id"],
            score=r["score"],
            distance=r["distance"],
            abnormal=r["score"] > settings.SIMILARITY_THRESHOLD,
        )
        for r in results
    ]
    return SearchResponse(query_id=None, results=formatted_results, total=len(formatted_results))


@router.post("/upload_dir", response_model=BatchUploadResponse)
async def upload_from_directory(
    directory: str = Query(..., description="服务器上的图片目录绝对路径"),
):
    """
    从服务器本地目录批量导入图片

    - 扫描指定目录下所有支持格式的图片（jpg/jpeg/png/bmp）
    - 不递归子目录
    - 逐张提取向量并存入 Milvus
    """
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"目录不存在: {directory}")

    image_files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in settings.ALLOWED_EXTENSIONS
    )
    if not image_files:
        raise HTTPException(
            status_code=400,
            detail=f"目录下没有支持的图片文件（{', '.join(settings.ALLOWED_EXTENSIONS)}）",
        )

    items: List[BatchUploadItem] = []
    success_count = 0
    fail_count = 0

    for filename in image_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "rb") as f:
                file_content = f.read()

            if not validate_image_file(file_content, filename):
                items.append(BatchUploadItem(
                    filename=filename, success=False,
                    error="Invalid image file or exceeds size limit",
                ))
                fail_count += 1
                continue

            image = load_image_from_bytes(file_content)
            logger.info("Extracting vectors from image: %s", filename)
            image_vector, face_vector, template_vector = vector_service.extract_all_vectors(image)
            has_face = face_vector is not None and not all(v == 0.0 for v in face_vector)

            inserted_id = milvus_client.insert(
                image_vector=image_vector,
                face_vector=face_vector,
                template_vector=template_vector,
            )
            logger.info("Image uploaded successfully: %s -> id=%s", filename, inserted_id)
            save_image_by_id(file_content, inserted_id, filename)
            save_certificate(inserted_id, filename, has_face)
            items.append(BatchUploadItem(
                filename=filename, id=inserted_id, has_face=has_face, success=True,
            ))
            success_count += 1

        except Exception as e:
            logger.error("Error processing %s: %s", filename, e)
            items.append(BatchUploadItem(filename=filename, success=False, error=str(e)))
            fail_count += 1

    message = f"成功 {success_count} 张，失败 {fail_count} 张"
    return BatchUploadResponse(
        items=items,
        success_count=success_count,
        fail_count=fail_count,
        message=message,
    )
