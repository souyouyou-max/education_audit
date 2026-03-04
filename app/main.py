"""
FastAPI 主应用 - 初始化与路由注册
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.milvus_client import milvus_client
from app.routes.fraud_routes import router as fraud_router
from app.routes.ocr_routes import router as ocr_router
from app.routes.vector_routes import router as vector_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理"""
    logger.info("Starting up application...")
    logger.info("UPLOAD_DIR (image storage for /view): %s", os.path.abspath(settings.UPLOAD_DIR))

    # 初始化 MySQL（失败不阻断启动，降级为无DB模式）
    try:
        from app.database import init_db
        init_db()
        logger.info("MySQL initialized successfully")
    except Exception as e:
        logger.warning("MySQL init failed (running without DB): %s", e)

    try:
        milvus_client.connect()
        milvus_client.create_collection_if_not_exists()
        logger.info("Milvus initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize Milvus: %s", e)
        raise
    yield
    logger.info("Shutting down application...")
    try:
        milvus_client.disconnect()
    except Exception as e:
        logger.error("Error during shutdown: %s", e)


app = FastAPI(
    title="学历证件相似度检测系统",
    description="基于 Milvus 和向量相似度的学历证件智能稽核系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    logger.error(
        "Unhandled exception %s %s: %s", request.method, request.url, exc, exc_info=True
    )
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(vector_router)
app.include_router(fraud_router)
app.include_router(ocr_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
