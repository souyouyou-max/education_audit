"""
数据模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """单张上传响应（保留兼容）"""
    id: int
    message: str
    has_face: bool


class BatchUploadItem(BaseModel):
    """批量上传中单条结果"""
    filename: str
    id: Optional[int] = None
    has_face: bool = False
    success: bool = True
    error: Optional[str] = None


class BatchUploadResponse(BaseModel):
    """批量上传响应"""
    items: List[BatchUploadItem]
    success_count: int
    fail_count: int
    message: str


class SearchRequest(BaseModel):
    """搜索请求"""
    id: Optional[int] = None
    image_vector: Optional[List[float]] = None
    face_vector: Optional[List[float]] = None
    template_vector: Optional[List[float]] = None
    vector_field: str = Field(default="image_vector", description="搜索的向量字段")
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResultItem(BaseModel):
    """搜索结果项"""
    id: int
    score: float
    distance: float
    abnormal: bool = False


class SearchResponse(BaseModel):
    """搜索响应"""
    query_id: Optional[int] = None
    results: List[SearchResultItem]
    total: int


class ClusterGroup(BaseModel):
    """聚类组"""
    group_id: Any
    items: List[str]  # 字符串 ID，防止 JS 大整数精度丢失
    count: int


class AbnormalGroup(BaseModel):
    """异常组"""
    group_id: Any
    items: List[str]  # 字符串 ID，防止 JS 大整数精度丢失
    count: int
    type: str


class ClusterParamsUsed(BaseModel):
    """聚类实际使用的参数（便于核对与调试）"""
    eps: float
    min_samples: int


class ClusterResponse(BaseModel):
    """聚类响应"""
    groups: List[ClusterGroup]
    abnormal_groups: List[AbnormalGroup]
    total_items: int
    total_groups: int
    message: Optional[str] = None
    params_used: Optional[ClusterParamsUsed] = None
    id_to_filename: Optional[Dict[str, str]] = None  # 返回的 id 对应的源文件名，key 为字符串 id


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None

