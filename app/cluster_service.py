"""
聚类服务
"""
import logging
from typing import List, Dict, Any
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from app.config import settings
from app.milvus_client import milvus_client
from app.id_filename_store import get_filenames_for_ids

logger = logging.getLogger(__name__)


class ClusterService:
    """聚类服务"""
    
    def __init__(self):
        pass
    
    def cluster_by_image_vector(self) -> Dict[str, Any]:
        """基于图像向量进行 DBSCAN 聚类"""
        try:
            # 获取所有图像向量
            vectors_dict = milvus_client.get_all_vectors("image_vector")
            
            if len(vectors_dict) == 0:
                return {
                    "groups": [],
                    "abnormal_groups": [],
                    "total_items": 0,
                    "total_groups": 0,
                    "params_used": {"eps": settings.DBSCAN_EPS, "min_samples": settings.DBSCAN_MIN_SAMPLES},
                    "id_to_filename": {},
                }
            
            # 转换为 numpy array
            ids = list(vectors_dict.keys())
            vectors = np.array([vectors_dict[id] for id in ids])
            
            # DBSCAN 聚类
            clustering = DBSCAN(
                eps=settings.DBSCAN_EPS,
                min_samples=settings.DBSCAN_MIN_SAMPLES,
                metric='cosine'
            )
            labels = clustering.fit_predict(vectors)
            
            # 组织结果
            groups = {}
            for idx, label in enumerate(labels):
                group_id = int(label) if label != -1 else None
                item_id = ids[idx]
                
                if group_id is None:
                    # 噪声点，单独成组
                    group_id = f"noise_{item_id}"
                
                if group_id not in groups:
                    groups[group_id] = []
                
                groups[group_id].append(str(item_id))
            
            # 识别异常组：仅当簇内数量 >= ABNORMAL_CLUSTER_MIN_SIZE 时标记（避免少量相似就报异常）
            abnormal_groups = []
            for group_id, items in groups.items():
                if isinstance(group_id, int) and len(items) >= settings.ABNORMAL_CLUSTER_MIN_SIZE:
                    abnormal_groups.append({
                        "group_id": group_id,
                        "items": items,
                        "count": len(items),
                        "type": "cluster_abnormal"
                    })
            
            id_to_filename = get_filenames_for_ids(ids)
            return {
                "groups": [
                    {
                        "group_id": group_id,
                        "items": items,
                        "count": len(items)
                    }
                    for group_id, items in groups.items()
                ],
                "abnormal_groups": abnormal_groups,
                "total_items": len(ids),
                "total_groups": len(groups),
                "params_used": {
                    "eps": settings.DBSCAN_EPS,
                    "min_samples": settings.DBSCAN_MIN_SAMPLES,
                },
                "id_to_filename": id_to_filename,
            }
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            raise
    
    def cluster_by_face_vector(self) -> Dict[str, Any]:
        """基于人脸向量进行 DBSCAN 聚类"""
        try:
            # 获取所有人脸向量
            vectors_dict = milvus_client.get_all_vectors("face_vector")
            
            if len(vectors_dict) == 0:
                return {
                    "groups": [],
                    "abnormal_groups": [],
                    "total_items": 0,
                    "total_groups": 0,
                    "params_used": {"eps": settings.DBSCAN_EPS, "min_samples": settings.DBSCAN_MIN_SAMPLES},
                    "id_to_filename": {},
                }
            
            # 过滤掉全0向量（未检测到人脸）
            filtered_dict = {
                id: vec for id, vec in vectors_dict.items()
                if not all(v == 0.0 for v in vec)
            }
            
            if len(filtered_dict) == 0:
                return {
                    "groups": [],
                    "abnormal_groups": [],
                    "total_items": 0,
                    "total_groups": 0,
                    "message": "No face vectors available",
                    "params_used": {"eps": settings.DBSCAN_EPS, "min_samples": settings.DBSCAN_MIN_SAMPLES},
                    "id_to_filename": {},
                }
            
            # 转换为 numpy array
            ids = list(filtered_dict.keys())
            vectors = np.array([filtered_dict[id] for id in ids])
            
            # DBSCAN 聚类
            clustering = DBSCAN(
                eps=settings.DBSCAN_EPS,
                min_samples=settings.DBSCAN_MIN_SAMPLES,
                metric='cosine'
            )
            labels = clustering.fit_predict(vectors)
            
            # 组织结果
            groups = {}
            for idx, label in enumerate(labels):
                group_id = int(label) if label != -1 else None
                item_id = ids[idx]
                
                if group_id is None:
                    group_id = f"noise_{item_id}"
                
                if group_id not in groups:
                    groups[group_id] = []
                
                groups[group_id].append(str(item_id))
            
            # 识别异常组：仅当簇内数量 >= ABNORMAL_CLUSTER_MIN_SIZE 时标记
            abnormal_groups = []
            for group_id, items in groups.items():
                if isinstance(group_id, int) and len(items) >= settings.ABNORMAL_CLUSTER_MIN_SIZE:
                    abnormal_groups.append({
                        "group_id": group_id,
                        "items": items,
                        "count": len(items),
                        "type": "cluster_abnormal"
                    })
            
            id_to_filename = get_filenames_for_ids(ids)
            return {
                "groups": [
                    {
                        "group_id": group_id,
                        "items": items,
                        "count": len(items)
                    }
                    for group_id, items in groups.items()
                ],
                "abnormal_groups": abnormal_groups,
                "total_items": len(ids),
                "total_groups": len(groups),
                "params_used": {
                    "eps": settings.DBSCAN_EPS,
                    "min_samples": settings.DBSCAN_MIN_SAMPLES,
                },
                "id_to_filename": id_to_filename,
            }
        except Exception as e:
            logger.error(f"Error in face clustering: {e}")
            raise

    def cluster_by_template_vector(self) -> Dict[str, Any]:
        """基于模板向量进行 DBSCAN 聚类（使用模板专用参数）

        模板向量已在提取阶段去除了人脸和文字等个性化内容，
        因此聚类结果反映的是证书模板（边框、背景、布局）的分组。
        """
        try:
            vectors_dict = milvus_client.get_all_vectors("template_vector")

            if len(vectors_dict) == 0:
                return {
                    "groups": [],
                    "abnormal_groups": [],
                    "total_items": 0,
                    "total_groups": 0,
                    "params_used": {
                        "eps": settings.TEMPLATE_DBSCAN_EPS,
                        "min_samples": settings.DBSCAN_MIN_SAMPLES,
                    },
                    "id_to_filename": {},
                }

            ids = list(vectors_dict.keys())
            vectors = np.array([vectors_dict[id] for id in ids])

            # 过滤全零向量（模板提取失败的记录）
            valid_mask = np.any(vectors != 0, axis=1)
            if not np.any(valid_mask):
                return {
                    "groups": [],
                    "abnormal_groups": [],
                    "total_items": 0,
                    "total_groups": 0,
                    "message": "No valid template vectors available",
                    "params_used": {
                        "eps": settings.TEMPLATE_DBSCAN_EPS,
                        "min_samples": settings.DBSCAN_MIN_SAMPLES,
                    },
                    "id_to_filename": {},
                }

            valid_ids = [ids[i] for i in range(len(ids)) if valid_mask[i]]
            valid_vectors = vectors[valid_mask]

            # 使用完全连接层次聚类（Complete Linkage Agglomerative Clustering）
            # 相比 DBSCAN，完全连接要求组内所有两两距离都 ≤ 阈值，彻底避免链式合并
            dist_matrix = cosine_distances(valid_vectors)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=settings.TEMPLATE_DBSCAN_EPS,
                metric='precomputed',
                linkage='complete'
            )
            labels = clustering.fit_predict(dist_matrix)

            # 统计每个簇的大小，小于 min_samples 的视为噪声
            label_counts = Counter(labels.tolist())

            groups = {}
            for idx, label in enumerate(labels):
                item_id = valid_ids[idx]
                if label_counts[label] < settings.DBSCAN_MIN_SAMPLES:
                    group_id = f"noise_{item_id}"
                else:
                    group_id = int(label)

                if group_id not in groups:
                    groups[group_id] = []

                groups[group_id].append(str(item_id))

            abnormal_groups = []
            for group_id, items in groups.items():
                if isinstance(group_id, int) and len(items) >= settings.ABNORMAL_CLUSTER_MIN_SIZE:
                    abnormal_groups.append({
                        "group_id": group_id,
                        "items": items,
                        "count": len(items),
                        "type": "template_cluster"
                    })

            id_to_filename = get_filenames_for_ids(valid_ids)
            return {
                "groups": [
                    {
                        "group_id": group_id,
                        "items": items,
                        "count": len(items)
                    }
                    for group_id, items in groups.items()
                ],
                "abnormal_groups": abnormal_groups,
                "total_items": len(valid_ids),
                "total_groups": len(groups),
                "params_used": {
                    "eps": settings.TEMPLATE_DBSCAN_EPS,
                    "min_samples": settings.DBSCAN_MIN_SAMPLES,
                },
                "id_to_filename": id_to_filename,
            }
        except Exception as e:
            logger.error(f"Error in template clustering: {e}")
            raise


# 全局服务实例
cluster_service = ClusterService()

