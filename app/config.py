"""
配置文件
"""
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

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
    TEMPLATE_VECTOR_DIM: int = 1088  # 模板向量维度（DINOv2-large 1024维 + HSV直方图 64维）
    
    # 索引配置
    INDEX_TYPE: str = "HNSW"
    METRIC_TYPE: str = "COSINE"
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    
    # CLIP 模型配置（用于全局图像向量）
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # DINOv2 模型配置（用于模板向量，纯视觉结构特征）
    DINO_MODEL_NAME: str = "facebook/dinov2-large"
    
    # InsightFace 配置
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    
    # 相似度阈值
    SIMILARITY_THRESHOLD: float = 0.92
    
    # 聚类配置
    DBSCAN_EPS: float = float(os.getenv("DBSCAN_EPS", "0.085"))  # 余弦距离半径（0.08 约相似度>0.92，使 5/6/7、8/9、10/11 等能成簇，见 docs/expected_similar_groups.md）
    DBSCAN_MIN_SAMPLES: int = 2  # 形成簇的最小样本数（2 条相似即可成簇，如可能重复）
    ABNORMAL_CLUSTER_MIN_SIZE: int = 2  # 簇内数量 >= 此值即标记为异常（2 条相似就值得人工核查）

    # 模板聚类专用参数（全局 HDBSCAN + DINOv2/LAB 特征融合，替代原 KMeans 预分组方案）
    TEMPLATE_DBSCAN_EPS: float = float(os.getenv("TEMPLATE_DBSCAN_EPS", "0.09"))  # 保留兼容，不再使用
    TEMPLATE_HDBSCAN_EPSILON: float = float(os.getenv("TEMPLATE_HDBSCAN_EPSILON", "0.05"))  # HDBSCAN cluster_selection_epsilon（融合特征空间）
    TEMPLATE_KMEANS_N: int = int(os.getenv("TEMPLATE_KMEANS_N", "5"))  # 保留兼容，不再使用
    TEMPLATE_MERGE_THRESHOLD: float = float(os.getenv("TEMPLATE_MERGE_THRESHOLD", "0.37"))  # DINOv2 质心欧氏距离跨簇合并阈值（同模板≈0.28~0.36，跨模板≈0.48+）
    TEMPLATE_COLOR_WEIGHT: float = float(os.getenv("TEMPLATE_COLOR_WEIGHT", "0.3"))   # LAB 颜色特征融合权重（0=纯DINOv2，1=纯颜色；0.3 在颜色相近模板中有效区分细微结构差异）
    TEMPLATE_NOISE_PAIR_THRESHOLD: float = float(os.getenv("TEMPLATE_NOISE_PAIR_THRESHOLD", "0.30"))  # 噪声点互配对阈值（combined 欧氏距离，0.30 可覆盖大多数同模板孤立点）
    TEMPLATE_BLUR_RADIUS: int = 2        # 高斯模糊半径：轻度模糊，消除扫描噪点
    TEMPLATE_BORDER_RATIO: float = 0.13  # HSV 直方图采样的边框宽度（只采最外框）
    
    # 搜索配置
    SEARCH_TOP_K: int = 5
    
    # 文件上传配置（使用项目根下的 uploads，避免因运行目录不同导致存/取路径不一致）
    _project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(_project_root, "uploads"))
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp"}

    # ZhengYan 多模态 OCR API 配置
    ZHENGYAN_ENDPOINT_URL: str = os.getenv("ZHENGYAN_ENDPOINT_URL", "")
    ZHENGYAN_ACCESS_TOKEN: str = os.getenv("ZHENGYAN_ACCESS_TOKEN", "")
    ZHENGYAN_APP_ID: str = os.getenv("ZHENGYAN_APP_ID", "")
    ZHENGYAN_MODEL_TYPE: str = os.getenv("ZHENGYAN_MODEL_TYPE", "")
    ZHENGYAN_USER_ID: str = os.getenv("ZHENGYAN_USER_ID", "")
    ZHENGYAN_USER_NAME: str = os.getenv("ZHENGYAN_USER_NAME", "")
    ZHENGYAN_USER_DEPT_NAME: str = os.getenv("ZHENGYAN_USER_DEPT_NAME", "")
    ZHENGYAN_USER_COMPANY: str = os.getenv("ZHENGYAN_USER_COMPANY", "")

    # MySQL 配置
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "123456")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "education_audit")

    # Admin 后台配置
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "education-audit-secret-key-change-in-production")

    @property
    def MYSQL_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

settings = Settings()

