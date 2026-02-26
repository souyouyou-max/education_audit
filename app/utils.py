"""
工具函数
"""
import os
import logging
from typing import Optional
from PIL import Image
import io
from app.config import settings

logger = logging.getLogger(__name__)


def ensure_upload_dir():
    """确保上传目录存在"""
    if not os.path.exists(settings.UPLOAD_DIR):
        os.makedirs(settings.UPLOAD_DIR)
        logger.info(f"Created upload directory: {settings.UPLOAD_DIR}")


def validate_image_file(file_content: bytes, filename: str) -> bool:
    """验证图片文件"""
    # 检查文件大小
    if len(file_content) > settings.MAX_UPLOAD_SIZE:
        return False
    
    # 检查文件扩展名
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False
    
    # 尝试打开图片
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file: {e}")
        return False


def load_image_from_bytes(file_content: bytes) -> Image.Image:
    """从字节流加载图片"""
    try:
        image = Image.open(io.BytesIO(file_content))
        # 转换为 RGB（如果是 RGBA 或其他格式）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """保存上传的文件"""
    ensure_upload_dir()
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(filepath, 'wb') as f:
        f.write(file_content)
    
    return filepath


def save_image_by_id(file_content: bytes, entity_id: int, original_filename: str) -> str:
    """按实体 id 保存图片，便于分组查看页按 id 拉取图片。保存为 uploads/{id}.{ext}"""
    ensure_upload_dir()
    ext = os.path.splitext(original_filename)[1].lower() or ".png"
    if ext not in settings.ALLOWED_EXTENSIONS:
        ext = ".png"
    filepath = os.path.join(settings.UPLOAD_DIR, f"{entity_id}{ext}")
    try:
        with open(filepath, "wb") as f:
            f.write(file_content)
        logger.info("Saved image by id: %s -> %s", entity_id, filepath)
        return filepath
    except Exception as e:
        logger.error("Failed to save image by id %s: %s", entity_id, e)
        raise


def get_image_path_by_id(entity_id: int) -> Optional[str]:
    """根据实体 id 查找已保存的图片路径，不存在返回 None"""
    ensure_upload_dir()
    base = os.path.join(settings.UPLOAD_DIR, str(entity_id))
    for ext in settings.ALLOWED_EXTENSIONS:
        path = base + ext
        if os.path.isfile(path):
            return path
    return None

