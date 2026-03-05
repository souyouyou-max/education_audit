"""
聚类服务
"""
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import hdbscan
import cv2
import torch
from PIL import Image
from app.config import settings
from app.milvus_client import milvus_client
from app.utils import get_filenames_for_ids, get_image_path_by_id


def estimate_optimal_eps(vectors: np.ndarray, k: int = 4, metric: str = "cosine") -> float:
    """
    使用 k-距离图估计最佳 DBSCAN eps。
    
    原理：计算每个点到其第 k 近邻的距离，排序后寻找"拐点"(elbow/knee)。
    拐点之前的距离变化平缓（噪声区），之后急剧上升（簇边界）。
    
    Args:
        vectors: 向量矩阵 (n_samples, n_features)
        k: 近邻数（通常取 min_samples 或 min_samples-1）
        metric: 距离度量
    
    Returns:
        估计的最佳 eps 值
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(vectors) <= k:
        return settings.DBSCAN_EPS  # 数据太少，返回默认值
    
    neigh = NearestNeighbors(n_neighbors=k+1, metric=metric)  # +1 因为包含自己
    neigh.fit(vectors)
    distances, _ = neigh.kneighbors(vectors)
    
    # 取第 k 近邻的距离（第 0 列是自己，距离为 0）
    k_distances = np.sort(distances[:, k])
    
    # 使用 Kneedle 算法找拐点
    # 简化版：计算相邻点斜率变化最大的位置
    diffs = np.diff(k_distances)
    if len(diffs) < 2:
        return settings.DBSCAN_EPS
    
    # 找二阶差分最大的点（曲率最大）
    second_diffs = np.diff(diffs)
    knee_idx = np.argmax(second_diffs) + 1  # +1 因为二阶差分比原数组少2个元素
    
    # 边界检查，确保 knee 在合理范围内
    if knee_idx < 1 or knee_idx >= len(k_distances):
        knee_idx = len(k_distances) // 4  # 保守估计：取 25% 位置
    
    estimated_eps = float(k_distances[knee_idx])
    
    # 限制在合理范围内
    return max(0.03, min(0.15, estimated_eps))


def compute_clustering_debug_info(vectors: np.ndarray, metric: str = "cosine") -> Dict[str, Any]:
    """
    计算聚类调试信息，帮助用户理解数据分布并调参。
    
    Returns:
        {
            "k_distance_curve": [...],  # k-距离曲线，用于画折线图
            "eps_candidates": [...],    # 不同 eps 对应的簇数量
            "distance_histogram": {...}, # 距离分布直方图
            "suggested_eps": float,     # 建议的 eps 值
        }
    """
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter
    
    n = len(vectors)
    if n <= 2:
        return {"error": "Too few samples for clustering debug"}
    
    # 1. 计算 k-距离曲线
    k = min(4, n-1)
    neigh = NearestNeighbors(n_neighbors=k+1, metric=metric)
    neigh.fit(vectors)
    distances, _ = neigh.kneighbors(vectors)
    k_distances = np.sort(distances[:, k]).tolist()
    
    # 2. 计算距离直方图
    pd = pairwise_distances(vectors, metric=metric)
    # 只取上三角（排除对角线）
    triu_indices = np.triu_indices(n, k=1)
    all_distances = pd[triu_indices]
    
    hist, bin_edges = np.histogram(all_distances, bins=20)
    distance_histogram = {
        "bins": bin_edges.tolist(),
        "counts": hist.tolist(),
    }
    
    # 3. 不同 eps 下的聚类数量
    eps_candidates = []
    for eps in np.linspace(0.03, 0.15, 13):  # 0.03 到 0.15，步长 0.01
        labels = DBSCAN(eps=eps, min_samples=2, metric=metric).fit_predict(vectors)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        eps_candidates.append({
            "eps": round(eps, 3),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        })
    
    # 4. 建议 eps
    suggested_eps = estimate_optimal_eps(vectors, k=k, metric=metric)
    
    return {
        "k_distance_curve": k_distances,
        "eps_candidates": eps_candidates,
        "distance_histogram": distance_histogram,
        "suggested_eps": round(suggested_eps, 4),
        "sample_count": n,
    }


# 添加 pairwise_distances 导入
from sklearn.metrics.pairwise import pairwise_distances

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

        # 自适应 eps：如果数据量足够，自动估计最佳 eps
        use_adaptive_eps = len(vectors) >= 10
        if use_adaptive_eps:
            optimal_eps = estimate_optimal_eps(vectors, k=settings.DBSCAN_MIN_SAMPLES, metric="cosine")
            logger.info("Adaptive eps estimation: %.4f (default was %.4f)", optimal_eps, settings.DBSCAN_EPS)
        else:
            optimal_eps = settings.DBSCAN_EPS
        
        labels = DBSCAN(
            eps=optimal_eps,
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

    @staticmethod
    def _extract_lab_color_features(img_path: str) -> np.ndarray:
        """提取 LAB 颜色直方图特征（52维）：L/A/B 各16 bin + LAB均值3维 + 红印比例1维。

        相比原 BGR均值+红印比例（4维），LAB直方图对扫描亮度变化更鲁棒，
        对同模板不同色调证书（如印刷年代导致的泛黄）具有更细粒度区分能力。
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                return np.zeros(52)
            img = cv2.medianBlur(img, 5)

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
            hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
            hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()
            hist_l /= hist_l.sum() + 1e-8
            hist_a /= hist_a.sum() + 1e-8
            hist_b /= hist_b.sum() + 1e-8
            mean_lab = lab.mean(axis=(0, 1)) / 255.0

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_red = (
                cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                + cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            )
            red_ratio = np.sum(mask_red > 0) / (img.shape[0] * img.shape[1])

            return np.concatenate([hist_l, hist_a, hist_b, mean_lab, [red_ratio]])
        except Exception:
            return np.zeros(52)

    def cluster_by_template_vector(self) -> Dict[str, Any]:
        """模板聚类（全局 HDBSCAN + DINOv2/LAB 特征融合）。

        改进点（相比原 KMeans 预分组方案）：
        1. 去除 KMeans 预分组 —— KMeans 按颜色粗分会把同模板不同扫描色调的证书分到不同预组，
           导致后续 HDBSCAN 无法发现它们属于同一模板；也会把视觉相近但模板不同的证书分到同一
           预组，HDBSCAN 因共享密度而错误合并。
        2. 全局 HDBSCAN（eom方法）—— 直接在全量特征上运行，避免预分组引入的边界误差。
        3. DINOv2 + LAB颜色特征融合 —— 纯 DINOv2 靠视觉结构区分模板，对颜色无感；
           加入 LAB直方图（52维）后，颜色差异对相似度有独立贡献，
           有效区分"结构相近、颜色不同"的不同模板证书。
        4. 噪声点互配对 —— 被孤立为噪声的同模板证书（如只有1张），
           通过两两 DINOv2 距离判断是否可以配对成新簇，减少单点丢失。

        阶段：
          1. 特征提取（DINOv2 1024D + LAB 52D）
          2. 特征融合（加权归一化拼接）
          3. 全局 HDBSCAN（eom，epsilon=TEMPLATE_HDBSCAN_EPSILON）
          4. 布局分裂（单页 vs 对折）
          5. 跨簇合并（DINOv2 质心距离 < TEMPLATE_MERGE_THRESHOLD，布局兼容）
          6. 噪声回收（最近质心 + 噪声互配对）
        """
        from app.vector_service import vector_service

        LAYOUT_SPLIT_THRESHOLD = 0.05

        def _empty(msg: str = ""):
            result = {
                "groups": [], "abnormal_groups": [],
                "total_items": 0, "total_groups": 0,
                "params_used": {
                    "hdbscan_epsilon": settings.TEMPLATE_HDBSCAN_EPSILON,
                    "color_weight": settings.TEMPLATE_COLOR_WEIGHT,
                    "merge_threshold": settings.TEMPLATE_MERGE_THRESHOLD,
                    "noise_pair_threshold": settings.TEMPLATE_NOISE_PAIR_THRESHOLD,
                    "layout_split_threshold": LAYOUT_SPLIT_THRESHOLD,
                },
                "id_to_filename": {},
            }
            if msg:
                result["message"] = msg
            return result

        try:
            upload_dir = settings.UPLOAD_DIR
            all_entity_ids = sorted(
                int(os.path.splitext(fname)[0])
                for fname in os.listdir(upload_dir)
                if os.path.splitext(fname)[1].lower() in settings.ALLOWED_EXTENSIONS
                and os.path.splitext(fname)[0].isdigit()
            )
            if not all_entity_ids:
                return _empty("No images found in uploads/")

            vector_service._ensure_models_loaded()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # ── 阶段 1：特征提取 ──────────────────────────────────────────
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
                    color_features.append(self._extract_lab_color_features(img_path))
                    img_paths.append(img_path)
                valid_ids.extend(batch_ids_ok)

            if not valid_ids:
                return _empty("No images found in uploads/")

            dino_arr = np.array(dino_features)   # (N, 1024) L2归一化
            color_arr = np.array(color_features)  # (N, 52)
            n = len(valid_ids)

            # ── 阶段 2：特征融合 ─────────────────────────────────────────
            # dino_arr: (N, 1024), color_arr: (N, 52) — 维度不同，不能直接相加
            # 正确做法：各自 L2 归一化后加权拼接，再整体归一化
            # 拼接后维度: (N, 1024+52) = (N, 1076)，HDBSCAN 用欧氏距离聚类
            dino_norm = dino_arr / (np.linalg.norm(dino_arr, axis=1, keepdims=True) + 1e-8)
            color_norm = color_arr / (np.linalg.norm(color_arr, axis=1, keepdims=True) + 1e-8)
            cw = settings.TEMPLATE_COLOR_WEIGHT
            # 加权拼接：dino 占 (1-cw) 权重，颜色占 cw 权重（通过缩放体现）
            combined = np.concatenate([
                (1.0 - cw) * dino_norm,   # (N, 1024)
                cw * color_norm,           # (N, 52)
            ], axis=1)  # → (N, 1076)
            logger.info(
                "Feature fusion: dino(1024) + LAB_color(52) → combined(%d), color_weight=%.2f",
                combined.shape[1], cw,
            )
            # ── 阶段 3：全局 HDBSCAN（eom 方法）────────────────────────────
            # eom（Excess of Mass）比 leaf 更稳健：优先选择层级树中持久性强的大簇，
            # 减少因密度微小波动导致大簇被错误拆分。
            all_labels = hdbscan.HDBSCAN(
                min_cluster_size=settings.DBSCAN_MIN_SAMPLES,
                min_samples=1,
                metric="euclidean",
                cluster_selection_epsilon=settings.TEMPLATE_HDBSCAN_EPSILON,
                cluster_selection_method="eom",
            ).fit_predict(combined)
            logger.info(
                "Template clustering HDBSCAN: n=%d, clusters=%d, noise=%d",
                n, len(set(all_labels) - {-1}), int(np.sum(all_labels == -1)),
            )

            # ── 阶段 4：布局分裂（单页 vs 对折双页） ────────────────────────
            edge_asymmetries = np.array([self._compute_edge_asymmetry(p) for p in img_paths])
            new_labels = all_labels.copy()
            cur_max = int(all_labels.max()) if (all_labels >= 0).any() else -1
            next_label = cur_max + 1
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

            # ── 阶段 5：跨簇合并（布局兼容 + DINOv2 质心距离） ────────────────
            unique_labels = sorted(set(all_labels) - {-1})
            if len(unique_labels) > 1:
                centroids: Dict[int, np.ndarray] = {}
                cluster_layout_type: Dict[int, str] = {}
                for lbl in unique_labels:
                    midx = np.where(all_labels == lbl)[0]
                    centroids[lbl] = dino_arr[midx].mean(axis=0)  # DINOv2 质心
                    cluster_layout_type[lbl] = (
                        "folded" if edge_asymmetries[midx].mean() >= LAYOUT_SPLIT_THRESHOLD
                        else "single"
                    )

                label_list = list(centroids.keys())
                centroid_matrix = np.array([centroids[l] for l in label_list])
                dist_matrix = squareform(pdist(centroid_matrix, metric="euclidean"))

                parent: Dict[int, int] = {l: l for l in label_list}

                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for i in range(len(label_list)):
                    for j in range(i + 1, len(label_list)):
                        if dist_matrix[i, j] < settings.TEMPLATE_MERGE_THRESHOLD:
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

            # ── 阶段 6：噪声回收（两轮） ───────────────────────────────────
            # 6a. 严格回收：噪声点到已有簇质心距离 < MERGE_THRESHOLD
            cluster_labels_now = sorted(set(all_labels) - {-1})
            noise_idx = np.where(all_labels == -1)[0]

            if len(noise_idx) > 0 and cluster_labels_now:
                # 用 combined 特征计算质心（和 HDBSCAN 输入一致）
                centroids_final = {
                    lbl: combined[np.where(all_labels == lbl)[0]].mean(axis=0)
                    for lbl in cluster_labels_now
                }
                centroid_labels_list = list(centroids_final.keys())
                centroid_vecs = np.array([centroids_final[l] for l in centroid_labels_list])
                for ni in noise_idx:
                    dists = np.linalg.norm(centroid_vecs - combined[ni], axis=1)
                    best_j = int(np.argmin(dists))
                    if dists[best_j] < settings.TEMPLATE_MERGE_THRESHOLD:
                        all_labels[ni] = centroid_labels_list[best_j]

            # 6b. 噪声互配对：剩余噪声点两两比较 combined 距离，生成新簇
            # 阈值用 NOISE_PAIR_THRESHOLD（比 MERGE_THRESHOLD 更宽松，专门给孤立点用）
            noise_idx = np.where(all_labels == -1)[0]
            if len(noise_idx) >= 2:
                pair_threshold = settings.TEMPLATE_NOISE_PAIR_THRESHOLD
                noise_parent: Dict[int, int] = {int(i): int(i) for i in noise_idx}

                def find_n(x: int) -> int:
                    while noise_parent[x] != x:
                        noise_parent[x] = noise_parent[noise_parent[x]]
                        x = noise_parent[x]
                    return x

                for ii_pos, ia in enumerate(noise_idx):
                    for ib in noise_idx[ii_pos + 1:]:
                        # 用 combined 特征距离（与 HDBSCAN 一致）
                        d = np.linalg.norm(combined[ia] - combined[ib])
                        if d < pair_threshold:
                            ra, rb = find_n(int(ia)), find_n(int(ib))
                            if ra != rb:
                                noise_parent[rb] = ra

                root_members: Dict[int, List[int]] = defaultdict(list)
                for ni in noise_idx:
                    root_members[find_n(int(ni))].append(int(ni))

                cur_max2 = int(all_labels.max()) if (all_labels >= 0).any() else -1
                next_lbl = cur_max2 + 1
                for root, members in root_members.items():
                    if len(members) >= settings.DBSCAN_MIN_SAMPLES:
                        for mi in members:
                            all_labels[mi] = next_lbl
                        logger.debug("Noise pairing: new cluster %d, members=%d", next_lbl, len(members))
                        next_lbl += 1

            # 6c. 软回收：对仍为噪声的点，放宽阈值（MERGE_THRESHOLD * 1.5）就近并入
            # 目的：尽量减少孤立噪点，将边缘样本归入最近的合理簇
            noise_idx = np.where(all_labels == -1)[0]
            if len(noise_idx) > 0 and cluster_labels_now:
                soft_threshold = settings.TEMPLATE_MERGE_THRESHOLD * 1.5
                # 重新计算质心（已有新簇加入）
                updated_labels = sorted(set(all_labels) - {-1})
                if updated_labels:
                    centroids_soft = {
                        lbl: combined[np.where(all_labels == lbl)[0]].mean(axis=0)
                        for lbl in updated_labels
                    }
                    soft_label_list = list(centroids_soft.keys())
                    soft_centroid_vecs = np.array([centroids_soft[l] for l in soft_label_list])
                    for ni in noise_idx:
                        dists = np.linalg.norm(soft_centroid_vecs - combined[ni], axis=1)
                        best_j = int(np.argmin(dists))
                        if dists[best_j] < soft_threshold:
                            all_labels[ni] = soft_label_list[best_j]
                            logger.debug(
                                "Soft recovery: noise idx=%d → cluster %d (dist=%.4f)",
                                ni, soft_label_list[best_j], dists[best_j]
                            )

            final_noise = int(np.sum(all_labels == -1))
            logger.info("Noise recovery done: %d remaining noise points (was %d)", final_noise, len(noise_idx) if len(noise_idx) > 0 else 0)

            # ── 整理输出 ─────────────────────────────────────────────────
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

            n_clusters = len([g for g in groups if isinstance(g, int)])
            n_noise = len([g for g in groups if not isinstance(g, int)])
            logger.info(
                "Template clustering done: n=%d, clusters=%d, noise_singletons=%d",
                n, n_clusters, n_noise,
            )

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
                    "color_weight": settings.TEMPLATE_COLOR_WEIGHT,
                    "merge_threshold": settings.TEMPLATE_MERGE_THRESHOLD,
                    "noise_pair_threshold": settings.TEMPLATE_NOISE_PAIR_THRESHOLD,
                    "layout_split_threshold": LAYOUT_SPLIT_THRESHOLD,
                },
                "id_to_filename": get_filenames_for_ids(valid_ids),
            }
        except Exception as e:
            logger.error("Error in template clustering: %s", e)
            raise


# 全局服务实例
cluster_service = ClusterService()
