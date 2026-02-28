#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终预聚类版：先按颜色+红印粗分，再每个组内细聚类
- 先 KMeans 粗分 5 个视觉大类（颜色 + 红印比例）
- 然后每个粗组内独立跑 HDBSCAN（epsilon 0.20~0.25）
- 这样能更好区分学校/印章/纸张变体
"""

import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="四川/重庆中学毕业证模板分组（预聚类版）")
    parser.add_argument("--image_dir", type=str, default="/Users/songyangyang/Desktop/education_audit/pic",
                        help="图片文件夹路径")
    parser.add_argument("--model", type=str, default="facebook/dinov2-large",
                        help="DINOv2 模型")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--min_cluster_size", type=int, default=2,
                        help="最小簇大小（≥2）")
    parser.add_argument("--fine_epsilon", type=float, default=0.12,
                        help="每个粗组内的 HDBSCAN epsilon（0.12可区分高中/中学，过大会混淆）")
    parser.add_argument("--output_dir", type=str, default="sichuan_chongqing_diplomas_grouped_precluster",
                        help="输出目录")
    parser.add_argument("--action", choices=["copy", "move"], default="copy",
                        help="copy 或 move")
    parser.add_argument("--use_mean_pool", action="store_true", default=False,
                        help="使用 mean pool（默认 CLS）")
    parser.add_argument("--merge_threshold", type=float, default=0.37,
                        help="跨预组簇合并+噪声回收阈值（同模板质心距≈0.28~0.36，跨模板≈0.48+）")
    return parser.parse_args()


def compute_edge_asymmetry(img_path):
    """计算左右半边的边缘密度不对称度。
    单页证书左右对称（< 0.05），对折双页证书左右内容不同（> 0.06）。
    """
    try:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        left_density = edges[:, :w // 2].mean()
        right_density = edges[:, w // 2:].mean()
        return abs(left_density - right_density) / (left_density + right_density + 1e-8)
    except Exception:
        return 0.0


def get_dominant_color_and_red_ratio(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return np.array([128, 128, 128]), 0.0

        img = cv2.medianBlur(img, 5)
        mask = cv2.inRange(img, np.array([20,20,20]), np.array([235,235,235]))
        masked_pixels = img.reshape(-1, 3)[mask.reshape(-1) > 0]
        mean_color = np.mean(masked_pixels, axis=0).astype(int) if len(masked_pixels) > 0 else np.mean(img.reshape(-1, 3), axis=0).astype(int)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        red_ratio = np.sum(mask_red > 0) / (img.shape[0] * img.shape[1]) * 100
        return mean_color, red_ratio
    except:
        return np.array([128, 128, 128]), 0.0


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        print(f"错误：文件夹不存在 -> {image_dir}")
        return

    output_base = Path(args.output_dir).expanduser().resolve()
    output_base.mkdir(exist_ok=True, parents=True)

    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    image_paths = sorted(p for ext in exts for p in image_dir.rglob(ext))

    print(f"找到 {len(image_paths)} 张毕业证图片")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    print(f"加载模型：{args.model}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    # 预计算颜色 + 红印
    print("预计算颜色 + 红印比例...")
    pre_features = []
    color_red_list = []
    for p in tqdm(image_paths, desc="预计算"):
        col, red = get_dominant_color_and_red_ratio(p)
        color_red_list.append((col, red))
        pre_features.append(np.concatenate([col / 255.0, [red / 10.0]]))  # 归一化

    pre_features = np.array(pre_features)

    # KMeans 粗分 5 个视觉大类
    n_pre_clusters = 5
    kmeans = KMeans(n_clusters=n_pre_clusters, n_init=10, random_state=42)
    pre_labels = kmeans.fit_predict(pre_features)

    print("\n预聚类结果（5 个视觉大类）：")
    pre_groups = defaultdict(list)
    for i, lbl in enumerate(pre_labels):
        pre_groups[lbl].append(i)
    for lbl in range(n_pre_clusters):
        idx = pre_groups[lbl]
        if len(idx) == 0: continue
        mean_col = np.mean([color_red_list[i][0] for i in idx], axis=0).astype(int)
        mean_red = np.mean([color_red_list[i][1] for i in idx])
        print(f"预组 {lbl} ({len(idx)}张): 主色 RGB{mean_col}, 平均红印 {mean_red:.1f}%")

    # 提取 DINOv2 特征（全局提取一次）
    def extract_features(paths, bs):
        feats_list = []
        for start in tqdm(range(0, len(paths), bs), desc="提取特征"):
            batch = paths[start:start + bs]
            try:
                images = [Image.open(p).convert("RGB") for p in batch]
            except Exception as e:
                print(f"跳过损坏图片：{e}")
                continue
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                feat = outputs.last_hidden_state[:, 1:].mean(dim=1) if args.use_mean_pool else outputs.last_hidden_state[:, 0]
            feats_list.append(feat.cpu().numpy())
        if not feats_list:
            return np.array([])
        return normalize(np.concatenate(feats_list))

    print("开始提取 DINOv2 特征...")
    embeddings = extract_features(image_paths, args.batch_size)
    if len(embeddings) == 0:
        return
    print(f"特征形状：{embeddings.shape}")

    # 在每个预组内独立聚类
    all_labels = np.full(len(image_paths), -1, dtype=int)  # 初始化为噪声
    cluster_counter = 0

    for pre_lbl in range(n_pre_clusters):
        idx = pre_groups[pre_lbl]
        if len(idx) < args.min_cluster_size:
            # 小组直接噪声
            continue

        sub_embeddings = embeddings[idx]
        sub_paths = [image_paths[i] for i in idx]

        print(f"\n预组 {pre_lbl} 内细聚类 ({len(idx)} 张)...")
        sub_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=args.fine_epsilon,
            cluster_selection_method='leaf',
        )
        sub_labels = sub_clusterer.fit_predict(sub_embeddings)

        # 映射回全局标签
        for local_lbl, global_idx in enumerate(idx):
            if sub_labels[local_lbl] != -1:
                all_labels[global_idx] = cluster_counter + sub_labels[local_lbl]
            # 噪声保持 -1

        # 更新计数器
        sub_n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        cluster_counter += sub_n_clusters

    # --- 布局分裂（在跨组合并之前执行） ---
    # DINOv2 无法区分单页 vs 对折双页证书。
    # 先拆分，避免混合布局的簇产生误导性质心导致后续错误合并。
    LAYOUT_SPLIT_THRESHOLD = 0.05
    edge_asymmetries = np.array([compute_edge_asymmetry(p) for p in image_paths])
    new_labels = all_labels.copy()
    next_label = max(all_labels) + 1 if len(all_labels) > 0 else 0
    split_count = 0
    for lbl in sorted(set(all_labels) - {-1}):
        member_idx = np.where(all_labels == lbl)[0]
        if len(member_idx) < 2 * args.min_cluster_size:
            continue
        asym_vals = edge_asymmetries[member_idx]
        low_mask = asym_vals < LAYOUT_SPLIT_THRESHOLD
        high_mask = ~low_mask
        n_low = int(np.sum(low_mask))
        n_high = int(np.sum(high_mask))
        if n_low >= args.min_cluster_size and n_high >= args.min_cluster_size:
            for local_i, global_i in enumerate(member_idx):
                if high_mask[local_i]:
                    new_labels[global_i] = next_label
            next_label += 1
            split_count += 1
    if split_count > 0:
        all_labels = new_labels
        print(f"布局分裂：{split_count} 个簇被拆分（单页 vs 对折）")

    # --- 跨预组簇合并（布局兼容） ---
    # KMeans 按颜色分组后，同模板但色调不同的图可能被分到不同预组。
    # 计算每个簇的 DINOv2 质心，距离小于阈值的簇合并。
    # 额外约束：只合并布局类型相同的簇（都是单页或都是对折）。
    from scipy.spatial.distance import pdist, squareform

    unique_labels = sorted(set(all_labels) - {-1})
    if len(unique_labels) > 1:
        centroids = {}
        cluster_layout_type = {}  # 每个簇的布局类型
        for lbl in unique_labels:
            member_idx = np.where(all_labels == lbl)[0]
            centroids[lbl] = embeddings[member_idx].mean(axis=0)
            mean_asym = edge_asymmetries[member_idx].mean()
            cluster_layout_type[lbl] = 'folded' if mean_asym >= LAYOUT_SPLIT_THRESHOLD else 'single'

        merge_threshold = args.merge_threshold
        label_list = list(centroids.keys())
        centroid_matrix = np.array([centroids[l] for l in label_list])
        dist_matrix = squareform(pdist(centroid_matrix, metric='euclidean'))

        # Union-Find 合并
        parent = {l: l for l in label_list}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged_count = 0
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
                        merged_count += 1

        if merged_count > 0:
            root_to_new = {}
            new_counter = 0
            merged_labels = np.full(len(image_paths), -1, dtype=int)
            for idx_i in range(len(image_paths)):
                if all_labels[idx_i] == -1:
                    continue
                root = find(all_labels[idx_i])
                if root not in root_to_new:
                    root_to_new[root] = new_counter
                    new_counter += 1
                merged_labels[idx_i] = root_to_new[root]
            all_labels = merged_labels
            print(f"\n跨预组合并：{merged_count} 次合并")

    # --- 噪声点回收 ---
    # 噪声点可能是被 KMeans 分到了缺乏同类的预组。
    # 把每个噪声点分配到最近的簇（如果与簇质心距离 < merge_threshold）。
    noise_idx = np.where(all_labels == -1)[0]
    cluster_labels_now = sorted(set(all_labels) - {-1})
    if len(noise_idx) > 0 and len(cluster_labels_now) > 0:
        centroids_final = {}
        for lbl in cluster_labels_now:
            member_idx = np.where(all_labels == lbl)[0]
            centroids_final[lbl] = embeddings[member_idx].mean(axis=0)
        centroid_labels = list(centroids_final.keys())
        centroid_vecs = np.array([centroids_final[l] for l in centroid_labels])

        rescue_threshold = args.merge_threshold
        rescued = 0
        for ni in noise_idx:
            dists = np.linalg.norm(centroid_vecs - embeddings[ni], axis=1)
            best_j = int(np.argmin(dists))
            if dists[best_j] < rescue_threshold:
                all_labels[ni] = centroid_labels[best_j]
                rescued += 1
        if rescued > 0:
            print(f"噪声回收：{rescued} 张噪声点被回收到最近的簇")

    # 统计最终簇
    final_labels = all_labels
    n_final_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
    n_noise = np.sum(final_labels == -1)
    print(f"\n最终发现 {n_final_clusters} 个模板簇，{n_noise} 张独特/噪声图片")

    # 计算最终统计
    print("\n最终各簇统计：")
    for label in sorted(set(final_labels)):
        if label == -1:
            continue
        idx = np.where(final_labels == label)[0]
        if len(idx) == 0: continue
        color_red_pairs = [get_dominant_color_and_red_ratio(image_paths[i]) for i in idx]
        mean_color = np.mean([p[0] for p in color_red_pairs], axis=0).astype(int)
        mean_red = np.mean([p[1] for p in color_red_pairs])
        color_name = "偏黄/老纸" if mean_color[0] > 180 and mean_color[1] > 150 else \
                      "偏灰/冷调" if mean_color[0] < 170 and mean_color[1] < 170 else "其他"
        print(f"  最终群 {label:02d} ({len(idx)}张): 主色 RGB{mean_color} ({color_name}), 平均红印 {mean_red:.1f}%")

    # 保存分组
    groups = defaultdict(list)
    for path, label in zip(image_paths, final_labels):
        groups[label].append(path)

    for label, paths in sorted(groups.items(), key=lambda x: -len(x[1])):
        if label == -1:
            cluster_dir = output_base / "unique_or_noise"
            title = f"独特/噪声图片 ({len(paths)} 张)"
        else:
            cluster_dir = output_base / f"template_{label:02d}"
            title = f"模板群 {label} ({len(paths)} 张)"

        cluster_dir.mkdir(exist_ok=True)
        print(f"\n{title}")

        for p in paths:
            dest = cluster_dir / p.name
            if args.action == "copy":
                shutil.copy2(p, dest)
            else:
                shutil.move(p, dest)
            print(f"  {p.name}")

    print(f"\n分组完成！结果保存在：{output_base}")
    print("建议：")
    print("1. 查看 template_xx + unique_or_noise")
    print("2. 如果仍不理想 → 调整 --fine_epsilon 或增加预组数（修改 n_pre_clusters=6）")
    print("3. 告诉我结果，我再调")


if __name__ == "__main__":
    main()