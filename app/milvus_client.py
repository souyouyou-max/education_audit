"""
Milvus 客户端管理
（旧版 Milvus 不支持单集合多向量字段，拆分为 image/face/template 三个集合，用 entity_id 关联）
"""
import logging
from typing import List, Optional, Dict, Any
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException
)
from app.config import settings

logger = logging.getLogger(__name__)

# 向量字段名到 (集合名, 向量列名, 维度) 的映射
VECTOR_FIELD_CONFIG = {
    "image_vector": (settings.COLLECTION_IMAGE, "image_vector", settings.IMAGE_VECTOR_DIM),
    "face_vector": (settings.COLLECTION_FACE, "face_vector", settings.FACE_VECTOR_DIM),
    "template_vector": (settings.COLLECTION_TEMPLATE, "template_vector", settings.TEMPLATE_VECTOR_DIM),
}


class MilvusClient:
    """Milvus 客户端封装（多集合版）"""

    def __init__(self):
        self.col_image: Optional[Collection] = None
        self.col_face: Optional[Collection] = None
        self.col_template: Optional[Collection] = None
        self._connected = False

    def _get_collection(self, vector_field: str) -> Optional[Collection]:
        """根据向量字段名获取对应 collection"""
        if vector_field == "image_vector":
            return self.col_image
        if vector_field == "face_vector":
            return self.col_face
        if vector_field == "template_vector":
            return self.col_template
        return None

    def connect(self):
        """连接 Milvus"""
        if self._connected:
            return

        try:
            connection_params = {
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT,
            }

            if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
                connection_params["user"] = settings.MILVUS_USER
                connection_params["password"] = settings.MILVUS_PASSWORD

            connections.connect("default", **connection_params)
            self._connected = True
            logger.info(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self):
        """断开 Milvus 连接"""
        if self._connected:
            try:
                connections.disconnect("default")
                self._connected = False
                logger.info("Disconnected from Milvus")
            except Exception as e:
                logger.error(f"Error disconnecting from Milvus: {e}")

    def create_collection_if_not_exists(self):
        """创建三个 collection（每个仅一个向量字段），若不存在则创建"""
        self.connect()

        def ensure_collection(name: str, vec_name: str, dim: int, with_entity_id: bool):
            if utility.has_collection(name):
                col = Collection(name)
                # 校验向量维度是否匹配，不匹配则删除重建
                for field in col.schema.fields:
                    if field.name == vec_name and field.params.get("dim") != dim:
                        old_dim = field.params.get("dim")
                        logger.warning(
                            f"Collection {name} dim mismatch: "
                            f"existing={old_dim}, expected={dim}. Dropping and recreating."
                        )
                        utility.drop_collection(name)
                        break
                else:
                    logger.info(f"Collection {name} already exists (dim={dim})")
                    return col
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=vec_name, dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            if with_entity_id:
                fields.insert(1, FieldSchema(name="entity_id", dtype=DataType.INT64, description="关联主 ID"))
            schema = CollectionSchema(fields=fields, description=f"Education certificate - {vec_name}")
            col = Collection(name=name, schema=schema)
            index_params = {
                "metric_type": settings.METRIC_TYPE,
                "index_type": settings.INDEX_TYPE,
                "params": {"M": settings.HNSW_M, "efConstruction": settings.HNSW_EF_CONSTRUCTION},
            }
            col.create_index(vec_name, index_params)
            logger.info(f"Collection {name} created with index on {vec_name}")
            return col

        self.col_image = ensure_collection(
            settings.COLLECTION_IMAGE, "image_vector", settings.IMAGE_VECTOR_DIM, False
        )
        self.col_face = ensure_collection(
            settings.COLLECTION_FACE, "face_vector", settings.FACE_VECTOR_DIM, True
        )
        self.col_template = ensure_collection(
            settings.COLLECTION_TEMPLATE, "template_vector", settings.TEMPLATE_VECTOR_DIM, True
        )

    def insert(self, image_vector: List[float], face_vector: List[float],
               template_vector: List[float]) -> int:
        """插入向量数据，返回实体 ID（以 image 集合的主键为准）"""
        self.create_collection_if_not_exists()

        # 先插入 image，得到 entity_id（auto_id 时 insert 返回的 primary_keys 即为新 id）
        # row-based 插入时每字段为该行的值，向量为 list[float] 而非 list of vectors
        res = self.col_image.insert({"image_vector": image_vector})
        self.col_image.flush()
        entity_id = res.primary_keys[0]

        self.col_face.insert({"entity_id": entity_id, "face_vector": face_vector})
        self.col_face.flush()
        self.col_template.insert({"entity_id": entity_id, "template_vector": template_vector})
        self.col_template.flush()

        return entity_id

    def search(self, vector_field: str, query_vector: List[float],
               top_k: int = None, expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """相似度搜索，返回的 id 为 entity_id"""
        self.create_collection_if_not_exists()
        col = self._get_collection(vector_field)
        if not col:
            raise ValueError(f"Unknown vector_field: {vector_field}")

        _, vec_name, _ = VECTOR_FIELD_CONFIG[vector_field]
        col.load()
        top_k = top_k or settings.SEARCH_TOP_K
        search_params = {"metric_type": settings.METRIC_TYPE, "params": {"ef": 50}}
        output = ["entity_id"] if vector_field != "image_vector" else ["id"]
        results = col.search(
            data=[query_vector],
            anns_field=vec_name,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output,
        )
        formatted_results = []
        for hits in results:
            for hit in hits:
                if vector_field == "image_vector":
                    eid = hit.id
                else:
                    eid = getattr(hit, "entity", {}).get("entity_id") or getattr(hit, "entity_id", None)
                    if eid is None and hasattr(hit, "id"):
                        eid = hit.id
                formatted_results.append({
                    "id": eid,
                    "distance": float(hit.distance),
                    "score": 1.0 - float(hit.distance),
                })
        return formatted_results

    def get_all_vectors(self, vector_field: str) -> Dict[int, List[float]]:
        """获取指定向量字段的全部向量，key 为 entity_id"""
        self.create_collection_if_not_exists()
        col = self._get_collection(vector_field)
        if not col:
            raise ValueError(f"Unknown vector_field: {vector_field}")

        _, vec_name, _ = VECTOR_FIELD_CONFIG[vector_field]
        col.load()
        id_field = "id" if vector_field == "image_vector" else "entity_id"
        results = col.query(expr="id >= 0", output_fields=[id_field, vec_name])
        return {r[id_field]: r[vec_name] for r in results}

    def get_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """根据 entity_id 获取三条向量"""
        self.create_collection_if_not_exists()
        self.col_image.load()
        self.col_face.load()
        self.col_template.load()

        img = self.col_image.query(expr=f"id == {id}", output_fields=["id", "image_vector"])
        face = self.col_face.query(expr=f"entity_id == {id}", output_fields=["entity_id", "face_vector"])
        tpl = self.col_template.query(expr=f"entity_id == {id}", output_fields=["entity_id", "template_vector"])

        if not img:
            return None
        out = {"id": img[0]["id"], "image_vector": img[0]["image_vector"]}
        out["face_vector"] = face[0]["face_vector"] if face else []
        out["template_vector"] = tpl[0]["template_vector"] if tpl else []
        return out

    def get_collection_stats(self) -> Dict[str, Any]:
        """统计以 image 集合实体数为准"""
        self.create_collection_if_not_exists()
        num_entities = self.col_image.num_entities
        return {
            "collection_name": settings.COLLECTION_NAME,
            "collections": [
                settings.COLLECTION_IMAGE,
                settings.COLLECTION_FACE,
                settings.COLLECTION_TEMPLATE,
            ],
            "num_entities": num_entities,
        }


# 全局客户端实例
milvus_client = MilvusClient()
