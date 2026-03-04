"""
工具函数（ID→文件名映射从 MySQL 读取）
"""
import os
import logging
from typing import Dict, List, Optional
from PIL import Image
import io
from app.config import settings

logger = logging.getLogger(__name__)


def ensure_upload_dir():
    """确保上传目录存在"""
    if not os.path.exists(settings.UPLOAD_DIR):
        os.makedirs(settings.UPLOAD_DIR)
        logger.info("Created upload directory: %s", settings.UPLOAD_DIR)


def validate_image_file(file_content: bytes, filename: str) -> bool:
    """验证图片文件"""
    if len(file_content) > settings.MAX_UPLOAD_SIZE:
        return False
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()
        return True
    except Exception as e:
        logger.error("Invalid image file: %s", e)
        return False


def load_image_from_bytes(file_content: bytes) -> Image.Image:
    """从字节流加载图片"""
    try:
        image = Image.open(io.BytesIO(file_content))
        image.load()  # 立即读取所有像素数据，防止 fp 被 JPEG 插件置 None 后延迟加载失败
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.error("Error loading image: %s", e)
        raise


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """保存上传的文件"""
    ensure_upload_dir()
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(file_content)
    return filepath


def save_image_by_id(file_content: bytes, entity_id: int, original_filename: str) -> str:
    """按实体 id 保存图片。保存为 uploads/{id}.{ext}"""
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


def scan_upload_ids() -> Dict[str, str]:
    """
    扫描 uploads/ 目录，从文件名推断 entity_id。
    文件名格式为 {entity_id}.{ext}（纯数字），返回 {str(entity_id): filename}。
    用于 MySQL 不可用时的兜底。
    """
    ensure_upload_dir()
    result = {}
    for fname in os.listdir(settings.UPLOAD_DIR):
        name, ext = os.path.splitext(fname)
        if ext.lower() in settings.ALLOWED_EXTENSIONS and name.isdigit():
            result[name] = fname
    return result


# ── ID → 文件名映射（从 MySQL certificates 表读取） ─────────────────

def get_id_to_filename() -> Dict[str, str]:
    """返回完整 id -> filename 映射，从 MySQL 读取，不可用时扫描 uploads/ 兜底"""
    try:
        from app.database import is_db_available, db_session
        from app.models import Certificate
        if is_db_available():
            with db_session() as db:
                rows = db.query(Certificate.id, Certificate.filename).all()
                if rows:
                    return {str(r.id): (r.filename or "") for r in rows}
    except Exception as e:
        logger.warning("MySQL get_id_to_filename failed: %s", e)
    return scan_upload_ids()


def get_filenames_for_ids(ids: List) -> Dict[str, str]:
    """返回给定 id 列表对应的 id -> filename 子集，从 MySQL 读取"""
    try:
        from app.database import is_db_available, db_session
        from app.models import Certificate
        if is_db_available():
            id_ints = [int(i) for i in ids]
            with db_session() as db:
                rows = db.query(Certificate.id, Certificate.filename).filter(
                    Certificate.id.in_(id_ints)
                ).all()
                return {str(r.id): (r.filename or "") for r in rows}
    except Exception as e:
        logger.warning("MySQL get_filenames_for_ids failed: %s", e)
    # 兜底：从 uploads/ 目录扫描
    all_ids = scan_upload_ids()
    return {str(i): all_ids[str(i)] for i in ids if str(i) in all_ids}
