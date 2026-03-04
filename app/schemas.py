"""
数据模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """单张上传响应（保留兼容）"""
    id: str          # 字符串 ID，防止 JS 大整数精度丢失（Milvus ID 超过 Number.MAX_SAFE_INTEGER）
    message: str
    has_face: bool


class BatchUploadItem(BaseModel):
    """批量上传中单条结果"""
    filename: str
    id: Optional[str] = None  # 字符串 ID，防止 JS 大整数精度丢失
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
    id: Optional[str] = None  # 接受字符串 ID（前端用字符串防精度丢失）；服务端内部用 int(id) 转换
    image_vector: Optional[List[float]] = None
    face_vector: Optional[List[float]] = None
    template_vector: Optional[List[float]] = None
    vector_field: str = Field(default="image_vector", description="搜索的向量字段")
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResultItem(BaseModel):
    """搜索结果项"""
    id: str      # 字符串 ID，防止 JS 大整数精度丢失（Milvus ID 超过 Number.MAX_SAFE_INTEGER）
    score: float
    distance: float
    abnormal: bool = False


class SearchResponse(BaseModel):
    """搜索响应"""
    query_id: Optional[str] = None  # 字符串 ID，防止 JS 大整数精度丢失
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
    eps: Optional[float] = None          # DBSCAN / Agglomerative（旧方案兼容）
    min_samples: Optional[int] = None    # 同上
    hdbscan_epsilon: Optional[float] = None  # HDBSCAN cluster_selection_epsilon
    kmeans_n: Optional[int] = None           # KMeans 预分组数


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


# ── 欺诈检测相关模型 ──────────────────────────────────────────────

class FaceInfo(BaseModel):
    """人脸属性信息"""
    age: Optional[int] = None
    gender: Optional[str] = None   # '男' / '女'
    det_score: Optional[float] = None
    bbox: Optional[List[float]] = None


class FraudIssue(BaseModel):
    """单条欺诈/风险问题"""
    rule: str                        # 规则名称
    detail: str                      # 详细描述
    severity: str                    # '高' / '中' / '低'
    related_ids: List[str] = []      # 关联的证件ID


class OcrFields(BaseModel):
    """OCR提取的结构化字段"""
    school: Optional[str] = None
    principal: Optional[str] = None
    grad_year: Optional[str] = None
    enrollment_year: Optional[str] = None
    issue_year: Optional[str] = None
    cert_no: Optional[str] = None
    gender: Optional[str] = None
    birth_year: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """单张证件全面分析响应"""
    entity_id: str  # 字符串 ID，防止 JS 大整数精度丢失
    filename: Optional[str] = None
    ocr_fields: Optional[OcrFields] = None
    face_info: Optional[FaceInfo] = None
    forensic: Optional[Dict[str, Any]] = None   # ELA / 照片边缘 / 印章
    rule_issues: List[FraudIssue] = []
    risk_level: str = "低"           # '高' / '中' / '低'
    risk_flags: List[str] = []


class CrossValidateResponse(BaseModel):
    """批量交叉核验响应"""
    total_certs: int                  # 参与核验的证件总数
    ocr_success: int                  # OCR成功数
    issues: List[FraudIssue] = []    # 所有发现的问题
    issue_count: int = 0
    high_severity_count: int = 0
    message: Optional[str] = None


# ── OCR 清单 ──────────────────────────────────────────────────────

class OcrListItem(BaseModel):
    """OCR 清单中单条证件"""
    id: str                                   # 字符串 id，防止 JS 大整数精度丢失
    filename: Optional[str] = None
    ocr_done: bool = False                    # 是否已运行过 OCR
    fields: Optional[Dict[str, Any]] = None  # 结构化字段（不含 _raw_text）
    raw_text: Optional[str] = None           # OCR 原始识别文本
    ps_data: Optional[Dict[str, Any]] = None # PS 鉴别结果（若已检测）


class OcrListResponse(BaseModel):
    """OCR 清单响应"""
    items: List[OcrListItem]
    total: int
    ocr_done_count: int

