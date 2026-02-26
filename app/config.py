"""
配置文件
"""
import os
from typing import Optional

class Settings:
    """应用配置"""
    
    # Milvus 配置
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER: Optional[str] = os.getenv("MILVUS_USER", None)
    MILVUS_PASSWORD: Optional[str] = os.getenv("MILVUS_PASSWORD", None)
    
    # Collection 配置（旧版 Milvus 不支持单集合多向量，拆分为三个集合）
    COLLECTION_NAME: str = "education_certificate"
    COLLECTION_IMAGE: str = "education_certificate_image"
    COLLECTION_FACE: str = "education_certificate_face"
    COLLECTION_TEMPLATE: str = "education_certificate_template"
    
    # 向量维度配置
    IMAGE_VECTOR_DIM: int = 512  # CLIP clip-vit-base-patch32 输出 512 维
    FACE_VECTOR_DIM: int = 512   # InsightFace 向量维度
    TEMPLATE_VECTOR_DIM: int = 576  # 模板向量维度（CLIP 512维 + HSV直方图 64维）
    
    # 索引配置
    INDEX_TYPE: str = "HNSW"
    METRIC_TYPE: str = "COSINE"
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    
    # CLIP 模型配置
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    
    # InsightFace 配置
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    
    # 相似度阈值
    SIMILARITY_THRESHOLD: float = 0.92
    
    # 聚类配置
    DBSCAN_EPS: float = float(os.getenv("DBSCAN_EPS", "0.085"))  # 余弦距离半径（0.08 约相似度>0.92，使 5/6/7、8/9、10/11 等能成簇，见 docs/expected_similar_groups.md）
    DBSCAN_MIN_SAMPLES: int = 2  # 形成簇的最小样本数（2 条相似即可成簇，如可能重复）
    ABNORMAL_CLUSTER_MIN_SIZE: int = 2  # 簇内数量 >= 此值即标记为异常（2 条相似就值得人工核查）

    # 模板聚类专用参数（模板特征相似度更高，eps 可适当放宽）
    TEMPLATE_DBSCAN_EPS: float = float(os.getenv("TEMPLATE_DBSCAN_EPS", "0.09"))
    TEMPLATE_BLUR_RADIUS: int = 2        # 高斯模糊半径：轻度模糊，保留院校名称可识别度（配合 CLIP_MASK_RATIO 遮罩个人内容）
    TEMPLATE_BORDER_RATIO: float = 0.13  # HSV 直方图采样的边框宽度（只采最外框）
    TEMPLATE_CLIP_MASK_RATIO: float = 0.20  # CLIP 遮罩保留比例：保留外侧 20%（含院校名/证书标题），遮掉中心收件人内容
    
    # 搜索配置
    SEARCH_TOP_K: int = 5
    
    # 文件上传配置（使用项目根下的 uploads，避免因运行目录不同导致存/取路径不一致）
    _project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(_project_root, "uploads")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp"}

settings = Settings()

