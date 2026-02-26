"""
FastAPI 主应用
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.config import settings
from app.milvus_client import milvus_client
from app.vector_service import vector_service
from app.cluster_service import cluster_service
from app.schemas import (
    UploadResponse,
    BatchUploadResponse,
    BatchUploadItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ClusterResponse,
    ErrorResponse
)
from app.utils import validate_image_file, load_image_from_bytes, save_image_by_id, get_image_path_by_id
from app.id_filename_store import set_id_filename

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理"""
    logger.info("Starting up application...")
    logger.info("UPLOAD_DIR (image storage for /view): %s", os.path.abspath(settings.UPLOAD_DIR))
    try:
        milvus_client.connect()
        milvus_client.create_collection_if_not_exists()
        logger.info("Milvus initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {e}")
        raise
    yield
    logger.info("Shutting down application...")
    try:
        milvus_client.disconnect()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# 创建 FastAPI 应用
app = FastAPI(
    title="学历证件相似度检测系统",
    description="基于 Milvus 和向量相似度的学历证件智能稽核系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "学历证件相似度检测系统",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - 批量上传学历图片（支持多文件）",
            "search": "POST /search - 相似度搜索",
            "cluster": "GET /cluster - 批量聚类分析",
            "view": "GET /view - 分组图片查看（Web 页面，支持左右滑动）"
        }
    }


def _groups_html_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "groups.html")


@app.get("/view", response_class=FileResponse)
async def view_groups_page():
    """分组图片查看 Web 页面（左右滑动查看各分组）"""
    path = _groups_html_path()
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="groups.html not found")
    return FileResponse(path)


@app.get("/api/image/{entity_id}", response_class=FileResponse)
async def get_image_by_id(entity_id: int):
    """根据实体 id 返回已保存的图片（上传时按 id 保存的）。用于分组查看页展示。"""
    path = get_image_path_by_id(entity_id)
    if not path:
        logger.warning("Image not found for id=%s (looked in %s)", entity_id, os.path.abspath(settings.UPLOAD_DIR))
        raise HTTPException(status_code=404, detail=f"Image not found for id {entity_id}")
    return FileResponse(path)


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        stats = milvus_client.get_collection_stats()
        return {
            "status": "healthy",
            "milvus": "connected",
            **stats
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/upload", response_model=BatchUploadResponse)
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
                    error="Invalid image file. Supported formats: JPG, JPEG, PNG, BMP. Max size: 10MB"
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
                template_vector=template_vector
            )
            logger.info("Image uploaded successfully: %s -> id=%s", filename, inserted_id)
            set_id_filename(inserted_id, filename)
            save_image_by_id(file_content, inserted_id, filename)
            items.append(BatchUploadItem(
                filename=filename,
                id=inserted_id,
                has_face=has_face,
                success=True
            ))
            success_count += 1
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error uploading image %s: %s", filename, e)
            items.append(BatchUploadItem(
                filename=filename,
                success=False,
                error=str(e)
            ))
            fail_count += 1
    message = f"成功 {success_count} 张，失败 {fail_count} 张" if (success_count + fail_count) > 1 else (
        "上传成功" if success_count == 1 else "上传失败"
    )
    return BatchUploadResponse(
        items=items,
        success_count=success_count,
        fail_count=fail_count,
        message=message
    )


@app.post("/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """
    相似度搜索
    
    支持通过 ID 或向量进行搜索
    - 如果提供 id，从 Milvus 获取对应的向量进行搜索
    - 如果提供向量，直接使用该向量搜索
    - 返回 Top5 相似证件及相似度
    """
    try:
        query_vector = None
        query_id = None
        
        # 确定查询向量
        if request.id is not None:
            # 通过 ID 查询
            query_id = request.id
            data = milvus_client.get_by_id(query_id)
            if not data:
                raise HTTPException(status_code=404, detail=f"Record with ID {request.id} not found")
            
            # 根据 vector_field 选择对应的向量
            if request.vector_field == "image_vector":
                query_vector = data["image_vector"]
            elif request.vector_field == "face_vector":
                query_vector = data["face_vector"]
            elif request.vector_field == "template_vector":
                query_vector = data["template_vector"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid vector_field: {request.vector_field}. Must be one of: image_vector, face_vector, template_vector"
                )
        
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
                detail="Must provide either 'id' or one of 'image_vector', 'face_vector', 'template_vector'"
            )
        
        # 验证向量维度
        expected_dims = {
            "image_vector": settings.IMAGE_VECTOR_DIM,
            "face_vector": settings.FACE_VECTOR_DIM,
            "template_vector": settings.TEMPLATE_VECTOR_DIM
        }
        
        if len(query_vector) != expected_dims[request.vector_field]:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension mismatch. Expected {expected_dims[request.vector_field]}, got {len(query_vector)}"
            )
        
        # 执行搜索
        logger.info(f"Searching with vector_field: {request.vector_field}, top_k: {request.top_k}")
        results = milvus_client.search(
            vector_field=request.vector_field,
            query_vector=query_vector,
            top_k=request.top_k
        )
        
        # 标记异常结果（相似度 > 0.92）
        formatted_results = []
        for result in results:
            abnormal = result["score"] > settings.SIMILARITY_THRESHOLD
            formatted_results.append(
                SearchResultItem(
                    id=result["id"],
                    score=result["score"],
                    distance=result["distance"],
                    abnormal=abnormal
                )
            )
        
        return SearchResponse(
            query_id=query_id,
            results=formatted_results,
            total=len(formatted_results)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.get("/cluster", response_model=ClusterResponse)
async def cluster_analysis(vector_field: str = "image_vector"):
    """
    批量聚类分析
    
    - 读取全部向量
    - 使用 DBSCAN 聚类
    - 返回分组结果和异常组
    """
    try:
        if vector_field not in ["image_vector", "face_vector", "template_vector"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vector_field: {vector_field}. Must be one of: image_vector, face_vector, template_vector"
            )
        
        logger.info(f"Starting clustering with vector_field: {vector_field}")
        
        if vector_field == "image_vector":
            result = cluster_service.cluster_by_image_vector()
        elif vector_field == "face_vector":
            result = cluster_service.cluster_by_face_vector()
        elif vector_field == "template_vector":
            result = cluster_service.cluster_by_template_vector()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported vector_field: {vector_field}"
            )
        
        return ClusterResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clustering: {str(e)}")


@app.post("/search_by_image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    vector_field: str = Form(default="image_vector"),
    top_k: int = Form(default=5)
):
    """
    通过上传图片进行相似度搜索
    
    便捷接口：上传图片后直接搜索相似证件
    """
    try:
        # 读取并验证文件
        file_content = await file.read()
        if not validate_image_file(file_content, file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        # 加载图片
        image = load_image_from_bytes(file_content)
        
        # 提取向量
        image_vector, face_vector, template_vector = vector_service.extract_all_vectors(image)
        
        # 根据 vector_field 选择向量
        if vector_field == "image_vector":
            query_vector = image_vector
        elif vector_field == "face_vector":
            query_vector = face_vector
        elif vector_field == "template_vector":
            query_vector = template_vector
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vector_field: {vector_field}"
            )
        
        # 执行搜索
        results = milvus_client.search(
            vector_field=vector_field,
            query_vector=query_vector,
            top_k=top_k
        )
        
        # 格式化结果
        formatted_results = []
        for result in results:
            abnormal = result["score"] > settings.SIMILARITY_THRESHOLD
            formatted_results.append(
                SearchResultItem(
                    id=result["id"],
                    score=result["score"],
                    distance=result["distance"],
                    abnormal=abnormal
                )
            )
        
        return SearchResponse(
            query_id=None,
            results=formatted_results,
            total=len(formatted_results)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_by_image: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upload_dir", response_model=BatchUploadResponse)
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

    # 收集目录下所有图片文件
    image_files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in settings.ALLOWED_EXTENSIONS
    )

    if not image_files:
        raise HTTPException(
            status_code=400,
            detail=f"目录下没有支持的图片文件（{', '.join(settings.ALLOWED_EXTENSIONS)}）"
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
                    error="Invalid image file or exceeds size limit"
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
                template_vector=template_vector
            )
            logger.info("Image uploaded successfully: %s -> id=%s", filename, inserted_id)
            set_id_filename(inserted_id, filename)
            save_image_by_id(file_content, inserted_id, filename)
            items.append(BatchUploadItem(
                filename=filename, id=inserted_id,
                has_face=has_face, success=True
            ))
            success_count += 1

        except Exception as e:
            logger.error("Error processing %s: %s", filename, e)
            items.append(BatchUploadItem(
                filename=filename, success=False, error=str(e)
            ))
            fail_count += 1

    message = f"成功 {success_count} 张，失败 {fail_count} 张"
    return BatchUploadResponse(
        items=items,
        success_count=success_count,
        fail_count=fail_count,
        message=message
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

