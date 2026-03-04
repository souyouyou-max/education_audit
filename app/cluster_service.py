"""
聚类服务
"""
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
import cv2
import torch
from PIL import Image
from app.config import settings
from app.milvus_client import milvus_client
from app.utils import get_filenames_for_ids, get_image_path_by_id

logger = logging.getLogger(__name__)


class ClusterService:
    """聚类服务"""

    def _cluster_by_vector(
        self,
        vector_field: str,
        filter_zero: bool = False,
    ) -> Dict[str, Any]:
        """通用 DBSCAN 聚类逻辑（image_vector / face_vector 共用）"""
        vectors_dict = milvus_client.get_all_vectors(vector_field)
        empty_result: Dict[str, Any] = {
            "groups": [],
            "abnormal_groups": [],
            "total_items": 0,
            "total_groups": 0,
            "params_used": {
                "eps": settings.DBSCAN_EPS,
                "min_samples": settings.DBSCAN_MIN_SAMPLES,
            },
            "id_to_filename": {},
        }

        if not vectors_dict:
            return empty_result

        if filter_zero:
            vectors_dict = {
                k: v for k, v in vectors_dict.items()
                if not all(x == 0.0 for x in v)
            }
            if not vectors_dict:
                return {**empty_result, "message": "No non-zero vectors available"}

        ids = list(vectors_dict.keys())
        vectors = np.array([vectors_dict[id_] for id_ in ids])

        labels = DBSCAN(
            eps=settings.DBSCAN_EPS,
            min_samples=settings.DBSCAN_MIN_SAMPLES,
            metric="cosine",
        ).fit_predict(vectors)

        groups: Dict = {}
        for idx, label in enumerate(labels):
            group_id = int(label) if label != -1 else f"noise_{ids[idx]}"
            groups.setdefault(group_id, []).append(str(ids[idx]))

        abnormal_groups = [
            {
                "group_id": gid,
                "items": items,
                "count": len(items),
                "type": "cluster_abnormal",
            }
            for gid, items in groups.items()
            if isinstance(gid, int) and len(items) >= settings.ABNORMAL_CLUSTER_MIN_SIZE
        ]

        return {
            "groups": [
                {"group_id": gid, "items": items, "count": len(items)}
                for gid, items in groups.items()
            ],
            "abnormal_groups": abnormal_groups,
            "total_items": len(ids),
            "total_groups": len(groups),
            "params_used": {
                "eps": settings.DBSCAN_EPS,
                "min_samples": settings.DBSCAN_MIN_SAMPLES,
            },
            "id_to_filename": get_filenames_for_ids(ids),
        }

    def cluster_by_image_vector(self) -> Dict[str, Any]:
        """基于图像向量进行 DBSCAN 聚类"""
        try:
            return self._cluster_by_vector("image_vector", filter_zero=False)
        except Exception as e:
            logger.error("Error in image clustering: %s", e)
            raise

    def cluster_by_face_vector(self) -> Dict[str, Any]:
        """基于人脸向量进行 DBSCAN 聚类（过滤未检测到人脸的全零向量）"""
        try:
            return self._cluster_by_vector("face_vector", filter_zero=True)
        except Exception as e:
            logger.error("Error in face clustering: %s", e)
            raise

    @staticmethod
    def _compute_edge_asymmetry(img_path: str) -> float:
        """计算左右半边的边缘密度不对称度。"""
        try:
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return 0.0
            h, w = gray.shape
            edges = cv2.Canny(gray, 50, 150)
            left_density = edges[:, :w // 2].mean()
            right_density = edges[:, w // 2:].mean()
            return abs(left_density - right_density) / (left_density + right_density + 1e-8)
        except Exception:
            return 0.0

    @staticmethod
    def _get_color_red_feature(img_path: str) -> np.ndarray:
        """提取图片的主色(BGR均值) + 红色印章比例，用于 KMeans 预分组。"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return np.array([128 / 255.0, 128 / 255.0, 128 / 255.0, 0.0])

            img = cv2.medianBlur(img, 5)
            mask = cv2.inRange(img, np.array([20, 20, 20]), np.array([235, 235, 235]))
            masked_pixels = img.reshape(-1, 3)[mask.reshape(-1) > 0]
            mean_color = (
                np.mean(masked_pixels, axis=0) / 255.0
                if len(masked_pixels) > 0
                else np.mean(img.reshape(-1, 3), axis=0) / 255.0
            )
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_red = (
                cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                + cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            )
            red_ratio = np.sum(mask_red > 0) / (img.shape[0] * img.shape[1]) * 100 / 10.0
            return np.concatenate([mean_color, [red_ratio]])
        except Exception:
            return np.array([128 / 255.0, 128 / 255.0, 128 / 255.0, 0.0])

    def cluster_by_template_vector(self) -> Dict[str, Any]:
        """两阶段模板聚类：KMeans 颜色预分组 → 组内 HDBSCAN 精细聚类。"""
        from app.vector_service import vector_service

        try:
            upload_dir = settings.UPLOAD_DIR
            all_entity_ids = sorted(
                int(os.path.splitext(fname)[0])
                for fname in os.listdir(upload_dir)
                if os.path.splitext(fname)[1].lower() in settings.ALLOWED_EXTENSIONS
                and os.path.splitext(fname)[0].isdigit()
            )

            if not all_entity_ids:
                return {
                    "groups": [], "abnormal_groups": [],
                    "total_items": 0, "total_groups": 0,
                    "params_used": {
                        "hdbscan_epsilon": settings.TEMPLATE_HDBSCAN_EPSILON,
                        "kmeans_n": settings.TEMPLATE_KMEANS_N,
                    },
                    "id_to_filename": {},
                }

            vector_service._ensure_models_loaded()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            dino_features: List[np.ndarray] = []
            color_features: List[np.ndarray] = []
            valid_ids: List[int] = []
            img_paths: List[str] = []
            batch_size = 4

            for start in range(0, len(all_entity_ids), batch_size):
                batch_ids = all_entity_ids[start:start + batch_size]
                batch_imgs, batch_paths, batch_ids_ok = [], [], []
                for eid in batch_ids:
                    img_path = get_image_path_by_id(int(eid))
                    if not img_path:
                        continue
                    try:
                        batch_imgs.append(Image.open(img_path).convert("RGB"))
                        batch_paths.append(img_path)
                        batch_ids_ok.append(eid)
                    except Exception as e:
                        logger.warning("Skip entity %s: %s", eid, e)

                if not batch_imgs:
                    continue

                with torch.no_grad():
                    inputs = vector_service.dino_processor(images=batch_imgs, return_tensors="pt")
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    outputs = vector_service.dino_model(**inputs)
                    feats = outputs.last_hidden_state[:, 0, :]
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    dino_features.extend(feats.cpu().numpy())

                for img_path in batch_paths:
                    color_features.append(self._get_color_red_feature(img_path))
                    img_paths.append(img_path)
                valid_ids.extend(batch_ids_ok)

            if not valid_ids:
                return {
                    "groups": [], "abnormal_groups": [],
                    "total_items": 0, "total_groups": 0,
                    "message": "No images found in uploads/",
                    "params_used": {
                        "hdbscan_epsilon": settings.TEMPLATE_HDBSCAN_EPSILON,
                        "kmeans_n": settings.TEMPLATE_KMEANS_N,
                    },
                    "id_to_filename": {},
                }

            dino_arr = np.array(dino_features)
            color_arr = np.array(color_features)
            n = len(valid_ids)

            # 第一阶段：KMeans 颜色预分组
            n_pre = min(settings.TEMPLATE_KMEANS_N, n)
            pre_labels = KMeans(n_clusters=n_pre, n_init=10, random_state=42).fit_predict(color_arr)
            pre_groups: Dict[int, List[int]] = defaultdict(list)
            for i, lbl in enumerate(pre_labels):
                pre_groups[int(lbl)].append(i)

            # 第二阶段：组内 HDBSCAN 精细聚类
            all_labels = np.full(n, -1, dtype=int)
            cluster_counter = 0
            for indices in pre_groups.values():
                if len(indices) < settings.DBSCAN_MIN_SAMPLES:
                    continue
                sub_labels = hdbscan.HDBSCAN(
                    min_cluster_size=settings.DBSCAN_MIN_SAMPLES,
                    min_samples=1,
                    metric="euclidean",
                    cluster_selection_epsilon=settings.TEMPLATE_HDBSCAN_EPSILON,
                    cluster_selection_method="leaf",
                ).fit_predict(dino_arr[indices])
                for local_idx, global_idx in enumerate(indices):
                    if sub_labels[local_idx] != -1:
                        all_labels[global_idx] = cluster_counter + int(sub_labels[local_idx])
                cluster_counter += len(set(sub_labels) - {-1})

            # 第三阶段：布局分裂
            LAYOUT_SPLIT_THRESHOLD = 0.05
            edge_asymmetries = np.array([
                self._compute_edge_asymmetry(img_paths[i]) for i in range(n)
            ])
            new_labels = all_labels.copy()
            next_label = int(max(all_labels)) + 1 if n > 0 else 0
            for lbl in sorted(set(all_labels) - {-1}):
                member_idx = np.where(all_labels == lbl)[0]
                if len(member_idx) < 2 * settings.DBSCAN_MIN_SAMPLES:
                    continue
                asym_vals = edge_asymmetries[member_idx]
                low_mask = asym_vals < LAYOUT_SPLIT_THRESHOLD
                if (int(np.sum(low_mask)) >= settings.DBSCAN_MIN_SAMPLES
                        and int(np.sum(~low_mask)) >= settings.DBSCAN_MIN_SAMPLES):
                    for local_i, global_i in enumerate(member_idx):
                        if not low_mask[local_i]:
                            new_labels[global_i] = next_label
                    next_label += 1
            all_labels = new_labels

            # 第四阶段：跨预组簇合并（布局兼容）
            unique_labels = sorted(set(all_labels) - {-1})
            if len(unique_labels) > 1:
                centroids: Dict[int, np.ndarray] = {}
                cluster_layout_type: Dict[int, str] = {}
                for lbl in unique_labels:
                    member_idx = np.where(all_labels == lbl)[0]
                    centroids[lbl] = dino_arr[member_idx].mean(axis=0)
                    cluster_layout_type[lbl] = (
                        "folded" if edge_asymmetries[member_idx].mean() >= LAYOUT_SPLIT_THRESHOLD
                        else "single"
                    )

                label_list = list(centroids.keys())
                centroid_matrix = np.array([centroids[l] for l in label_list])
                dist_matrix = squareform(pdist(centroid_matrix, metric="euclidean"))

                parent = {l: l for l in label_list}

                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                merge_threshold = settings.TEMPLATE_MERGE_THRESHOLD
                for i in range(len(label_list)):
                    for j in range(i + 1, len(label_list)):
                        if dist_matrix[i, j] < merge_threshold:
                            li, lj = label_list[i], label_list[j]
                            if cluster_layout_type[li] != cluster_layout_type[lj]:
                                continue
                            ri, rj = find(li), find(lj)
                            if ri != rj:
                                parent[rj] = ri

                root_to_new: Dict[int, int] = {}
                new_counter = 0
                merged_labels = np.full(n, -1, dtype=int)
                for idx in range(n):
                    if all_labels[idx] == -1:
                        continue
                    root = find(all_labels[idx])
                    if root not in root_to_new:
                        root_to_new[root] = new_counter
                        new_counter += 1
                    merged_labels[idx] = root_to_new[root]
                all_labels = merged_labels

            # 第五阶段：噪声点回收
            noise_idx = np.where(all_labels == -1)[0]
            cluster_labels_now = sorted(set(all_labels) - {-1})
            if len(noise_idx) > 0 and cluster_labels_now:
                centroids_final = {
                    lbl: dino_arr[np.where(all_labels == lbl)[0]].mean(axis=0)
                    for lbl in cluster_labels_now
                }
                centroid_labels_list = list(centroids_final.keys())
                centroid_vecs = np.array([centroids_final[l] for l in centroid_labels_list])
                for ni in noise_idx:
                    dists = np.linalg.norm(centroid_vecs - dino_arr[ni], axis=1)
                    best_j = int(np.argmin(dists))
                    if dists[best_j] < settings.TEMPLATE_MERGE_THRESHOLD:
                        all_labels[ni] = centroid_labels_list[best_j]

            # 整理输出
            groups: Dict = {}
            for idx, label in enumerate(all_labels):
                item_id = valid_ids[idx]
                group_id = int(label) if label != -1 else f"noise_{item_id}"
                groups.setdefault(group_id, []).append(str(item_id))

            abnormal_groups = [
                {"group_id": gid, "items": items, "count": len(items), "type": "template_cluster"}
                for gid, items in groups.items()
                if isinstance(gid, int) and len(items) >= settings.ABNORMAL_CLUSTER_MIN_SIZE
            ]

            return {
                "groups": [
                    {"group_id": gid, "items": items, "count": len(items)}
                    for gid, items in groups.items()
                ],
                "abnormal_groups": abnormal_groups,
                "total_items": n,
                "total_groups": len(groups),
                "params_used": {
                    "hdbscan_epsilon": settings.TEMPLATE_HDBSCAN_EPSILON,
                    "kmeans_n": n_pre,
                    "merge_threshold": settings.TEMPLATE_MERGE_THRESHOLD,
                    "layout_split_threshold": LAYOUT_SPLIT_THRESHOLD,
                },
                "id_to_filename": get_filenames_for_ids(valid_ids),
            }
        except Exception as e:
            logger.error("Error in template clustering: %s", e)
            raise


# 全局服务实例
cluster_service = ClusterService()
