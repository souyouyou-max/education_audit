"""
聚类服务
"""
import logging
from typing import List, Dict, Any
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import cv2
import torch
from PIL import Image
from app.config import settings
from app.milvus_client import milvus_client
from app.id_filename_store import get_filenames_for_ids
from app.utils import get_image_path_by_id

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

    @staticmethod
    def _compute_edge_asymmetry(img_path: str) -> float:
        """计算左右半边的边缘密度不对称度。
        单页证书左右对称（< 0.05），对折双页证书左右内容不同（> 0.06）。
        """
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
        """提取图片的主色(BGR均值) + 红色印章比例，用于 KMeans 预分组。
        逻辑与 cluster_images_dinov2.py 中的 get_dominant_color_and_red_ratio 保持一致。
        """
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
        """两阶段模板聚类：KMeans 颜色预分组 → 组内 HDBSCAN 精细聚类。

        聚类直接从 uploads/ 图片提取纯 DINOv2 特征（不使用 Milvus 存储的混合向量），
        与 cluster_images_dinov2.py 逻辑对齐，避免 HSV 混合干扰聚类精度。
        Milvus 中的 template_vector 仍用于 /search 按模板风格搜索。
        """
        # 延迟导入避免循环依赖
        from app.vector_service import vector_service

        try:
            # 直接扫描 uploads/ 目录获取所有图片，不依赖 Milvus 集合
            import os
            upload_dir = settings.UPLOAD_DIR
            all_entity_ids = []
            for fname in os.listdir(upload_dir):
                name, ext = os.path.splitext(fname)
                if ext.lower() in settings.ALLOWED_EXTENSIONS:
                    try:
                        all_entity_ids.append(int(name))
                    except ValueError:
                        pass
            # 排序以保证与独立脚本一致的处理顺序
            all_entity_ids.sort()

            if not all_entity_ids:
                return {
                    "groups": [], "abnormal_groups": [],
                    "total_items": 0, "total_groups": 0,
                    "params_used": {"hdbscan_epsilon": settings.TEMPLATE_HDBSCAN_EPSILON,
                                    "kmeans_n": settings.TEMPLATE_KMEANS_N},
                    "id_to_filename": {},
                }

            # 确保 DINOv2 模型已加载
            vector_service._ensure_models_loaded()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # --- 从图片批量提取纯 DINOv2 特征 + 颜色特征 ---
            dino_features = []
            color_features = []
            valid_ids = []
            img_paths = []  # 保存路径，供布局分裂步骤使用
            batch_size = 4

            for start in range(0, len(all_entity_ids), batch_size):
                batch_ids = all_entity_ids[start:start + batch_size]
                batch_imgs, batch_paths, batch_ids_ok = [], [], []
                for eid in batch_ids:
                    img_path = get_image_path_by_id(int(eid))
                    if not img_path:
                        continue
                    try:
                        img = Image.open(img_path).convert("RGB")
                        batch_imgs.append(img)
                        batch_paths.append(img_path)
                        batch_ids_ok.append(eid)
                    except Exception as e:
                        logger.warning(f"Skip {eid}: {e}")

                if not batch_imgs:
                    continue

                # DINOv2 批量推理
                with torch.no_grad():
                    inputs = vector_service.dino_processor(images=batch_imgs, return_tensors="pt")
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
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
                    "params_used": {"hdbscan_epsilon": settings.TEMPLATE_HDBSCAN_EPSILON,
                                    "kmeans_n": settings.TEMPLATE_KMEANS_N},
                    "id_to_filename": {},
                }

            dino_features = np.array(dino_features)
            color_features = np.array(color_features)
            n = len(valid_ids)

            # --- 第一阶段：KMeans 颜色预分组 ---
            # 与独立脚本对齐：固定使用 TEMPLATE_KMEANS_N，图片数不足时退化为图片数
            n_pre = min(settings.TEMPLATE_KMEANS_N, n)
            pre_labels = KMeans(n_clusters=n_pre, n_init=10, random_state=42).fit_predict(color_features)
            pre_groups: Dict[int, list] = defaultdict(list)
            for i, lbl in enumerate(pre_labels):
                pre_groups[int(lbl)].append(i)

            # --- 第二阶段：组内 HDBSCAN 精细聚类 ---
            all_labels = np.full(n, -1, dtype=int)
            cluster_counter = 0
            for indices in pre_groups.values():
                if len(indices) < settings.DBSCAN_MIN_SAMPLES:
                    continue
                sub_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=settings.DBSCAN_MIN_SAMPLES,
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_epsilon=settings.TEMPLATE_HDBSCAN_EPSILON,
                    cluster_selection_method='leaf',
                )
                sub_labels = sub_clusterer.fit_predict(dino_features[indices])
                for local_idx, global_idx in enumerate(indices):
                    if sub_labels[local_idx] != -1:
                        all_labels[global_idx] = cluster_counter + int(sub_labels[local_idx])
                cluster_counter += len(set(sub_labels)) - (1 if -1 in sub_labels else 0)

            # --- 第三阶段：布局分裂（在跨组合并之前执行） ---
            # DINOv2 无法区分单页 vs 对折双页证书（特征距离重叠）。
            # 先拆分，避免混合布局的簇产生误导性质心导致后续错误合并。
            LAYOUT_SPLIT_THRESHOLD = 0.05
            edge_asymmetries = np.array([
                self._compute_edge_asymmetry(img_paths[i]) for i in range(n)
            ])
            new_labels = all_labels.copy()
            next_label = max(all_labels) + 1 if len(all_labels) > 0 else 0
            for lbl in sorted(set(all_labels) - {-1}):
                member_idx = np.where(all_labels == lbl)[0]
                if len(member_idx) < 2 * settings.DBSCAN_MIN_SAMPLES:
                    continue  # 太小的簇不值得拆
                asym_vals = edge_asymmetries[member_idx]
                low_mask = asym_vals < LAYOUT_SPLIT_THRESHOLD
                high_mask = ~low_mask
                n_low = int(np.sum(low_mask))
                n_high = int(np.sum(high_mask))
                if n_low >= settings.DBSCAN_MIN_SAMPLES and n_high >= settings.DBSCAN_MIN_SAMPLES:
                    # 簇内同时存在单页和对折 → 拆分
                    for local_i, global_i in enumerate(member_idx):
                        if high_mask[local_i]:
                            new_labels[global_i] = next_label
                    next_label += 1
            all_labels = new_labels

            # --- 第四阶段：跨预组簇合并（布局兼容） ---
            # KMeans 按颜色分组后，同模板但色调不同的图可能被分到不同预组。
            # 计算每个簇的 DINOv2 质心，距离小于阈值的簇合并。
            # 额外约束：只合并布局类型相同的簇（都是单页或都是对折），
            # 防止单页证书和对折证书因 DINOv2 距离接近而错误合并。
            unique_labels = sorted(set(all_labels) - {-1})
            if len(unique_labels) > 1:
                centroids = {}
                cluster_layout_type = {}  # 每个簇的布局类型：'single' or 'folded'
                for lbl in unique_labels:
                    member_idx = np.where(all_labels == lbl)[0]
                    centroids[lbl] = dino_features[member_idx].mean(axis=0)
                    mean_asym = edge_asymmetries[member_idx].mean()
                    cluster_layout_type[lbl] = 'folded' if mean_asym >= LAYOUT_SPLIT_THRESHOLD else 'single'

                merge_threshold = settings.TEMPLATE_MERGE_THRESHOLD
                label_list = list(centroids.keys())
                centroid_matrix = np.array([centroids[l] for l in label_list])
                from scipy.spatial.distance import pdist, squareform
                dist_matrix = squareform(pdist(centroid_matrix, metric='euclidean'))

                # Union-Find 合并
                parent = {l: l for l in label_list}

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for i in range(len(label_list)):
                    for j in range(i + 1, len(label_list)):
                        if dist_matrix[i, j] < merge_threshold:
                            li, lj = label_list[i], label_list[j]
                            # 只合并布局类型相同的簇
                            if cluster_layout_type[li] != cluster_layout_type[lj]:
                                continue
                            ri, rj = find(li), find(lj)
                            if ri != rj:
                                parent[rj] = ri

                # 重映射标签
                root_to_new = {}
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

            # --- 第五阶段：噪声点回收 ---
            # 噪声点可能是被 KMeans 分到了缺乏同类的预组。
            # 把每个噪声点分配到最近的簇（如果与簇质心距离 < merge_threshold）。
            noise_idx = np.where(all_labels == -1)[0]
            cluster_labels_now = sorted(set(all_labels) - {-1})
            if len(noise_idx) > 0 and len(cluster_labels_now) > 0:
                # 重新计算合并后的簇质心
                centroids_final = {}
                for lbl in cluster_labels_now:
                    member_idx = np.where(all_labels == lbl)[0]
                    centroids_final[lbl] = dino_features[member_idx].mean(axis=0)
                centroid_labels = list(centroids_final.keys())
                centroid_vecs = np.array([centroids_final[l] for l in centroid_labels])

                rescue_threshold = settings.TEMPLATE_MERGE_THRESHOLD
                for ni in noise_idx:
                    dists = np.linalg.norm(centroid_vecs - dino_features[ni], axis=1)
                    best_j = int(np.argmin(dists))
                    if dists[best_j] < rescue_threshold:
                        all_labels[ni] = centroid_labels[best_j]

            # --- 整理输出 ---
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

            id_to_filename = get_filenames_for_ids(valid_ids)
            return {
                "groups": [{"group_id": gid, "items": items, "count": len(items)}
                           for gid, items in groups.items()],
                "abnormal_groups": abnormal_groups,
                "total_items": n,
                "total_groups": len(groups),
                "params_used": {
                    "hdbscan_epsilon": settings.TEMPLATE_HDBSCAN_EPSILON,
                    "kmeans_n": n_pre,
                    "merge_threshold": settings.TEMPLATE_MERGE_THRESHOLD,
                    "layout_split_threshold": 0.05,
                },
                "id_to_filename": id_to_filename,
            }
        except Exception as e:
            logger.error(f"Error in template clustering: {e}")
            raise


# 全局服务实例
cluster_service = ClusterService()

