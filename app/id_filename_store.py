"""
ID 与上传文件名的映射存储（用于聚类等接口返回源文件名）
"""
import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "id_to_filename.json",
)


def _ensure_dir():
    d = os.path.dirname(_STORE_PATH)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _load() -> Dict[str, str]:
    if not os.path.isfile(_STORE_PATH):
        return {}
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load id_to_filename store: %s", e)
        return {}


def _save(data: Dict[str, str]) -> None:
    _ensure_dir()
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_id_filename(entity_id: int, filename: str) -> None:
    """上传成功后记录 id -> 文件名"""
    data = _load()
    data[str(entity_id)] = filename
    _save(data)


def get_id_to_filename() -> Dict[str, str]:
    """返回完整 id -> filename 映射（key 为字符串 id）"""
    return _load()


def get_filenames_for_ids(ids: list) -> Dict[str, str]:
    """只返回给定 id 列表对应的 id -> filename 子集"""
    data = _load()
    return {str(i): data[str(i)] for i in ids if str(i) in data}
