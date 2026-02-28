#!/usr/bin/env python3
"""诊断：模拟完整 5 阶段管道，跟踪每一步的聚类变化"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
import hdbscan
import cv2

IMAGE_DIR = Path("/Users/songyangyang/Desktop/education_audit/pic")
MODEL_NAME = "facebook/dinov2-large"

# 参数（与 cluster_images_dinov2.py 对齐）
KMEANS_N = 5
HDBSCAN_EPSILON = 0.12
MIN_CLUSTER_SIZE = 2
MERGE_THRESHOLD = 0.30
LAYOUT_SPLIT_THRESHOLD = 0.05

# 获取所有图片
exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
image_paths = sorted(p for ext in exts for p in IMAGE_DIR.rglob(ext))
print(f"找到 {len(image_paths)} 张图片")

# 关注的图片
WATCH_GROUP10 = {"图片 1.png", "图片 2.png", "图片 3.png", "图片 4.png",
                 "图片 5.png", "图片 6.png", "图片 7.png", "图片 8.png",
                 "图片 16.png", "图片 22.png", "图片 30.png", "图片 34.png",
                 "图片 41.png", "图片 43.png", "图片 44.png", "图片 46.png"}
WATCH_SHOULD_MERGE = {"图片 18.png", "图片 28.png", "图片 36.png", "图片 40.png",
                      "图片 20.png", "图片 37.png", "图片 38.png", "图片 42.png"}

def get_color_red_feature(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return np.array([128/255.0, 128/255.0, 128/255.0, 0.0])
        img = cv2.medianBlur(img, 5)
        mask = cv2.inRange(img, np.array([20,20,20]), np.array([235,235,235]))
        masked_pixels = img.reshape(-1, 3)[mask.reshape(-1) > 0]
        mean_color = (np.mean(masked_pixels, axis=0) / 255.0
                      if len(masked_pixels) > 0
                      else np.mean(img.reshape(-1, 3), axis=0) / 255.0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red = (cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                    + cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255])))
        red_ratio = np.sum(mask_red > 0) / (img.shape[0] * img.shape[1]) * 100 / 10.0
        return np.concatenate([mean_color, [red_ratio]])
    except:
        return np.array([128/255.0, 128/255.0, 128/255.0, 0.0])

def compute_edge_asymmetry(img_path):
    try:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        left_d = edges[:, :w//2].mean()
        right_d = edges[:, w//2:].mean()
        return abs(left_d - right_d) / (left_d + right_d + 1e-8)
    except:
        return 0.0

def show_watch_groups(labels, paths, stage_name, watch_set):
    """显示关注图片在哪个组"""
    fname_to_label = {}
    label_to_fnames = defaultdict(list)
    for i, p in enumerate(paths):
        if p.name in watch_set:
            fname_to_label[p.name] = labels[i]
            label_to_fnames[labels[i]].append(p.name)

    print(f"\n  [{stage_name}] 关注图片的分组:")
    for lbl in sorted(label_to_fnames.keys()):
        fnames = sorted(label_to_fnames[lbl])
        tag = "噪声" if lbl == -1 else f"簇{lbl}"
        print(f"    {tag}: {fnames}")

# --- 提取特征 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

print("提取颜色特征...")
color_features = np.array([get_color_red_feature(p) for p in image_paths])

print("提取 DINOv2 特征...")
dino_features = []
batch_size = 4
for start in range(0, len(image_paths), batch_size):
    batch = image_paths[start:start + batch_size]
    try:
        images = [Image.open(p).convert("RGB") for p in batch]
    except Exception as e:
        print(f"跳过: {e}")
        continue
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 0]
    dino_features.append(feats.cpu().numpy())

dino_features = normalize(np.concatenate(dino_features))
n = len(image_paths)
print(f"特征形状: {dino_features.shape}")

# === 第一阶段：KMeans 颜色预分组 ===
print("\n" + "="*60)
print("第一阶段：KMeans 颜色预分组")
n_pre = min(KMEANS_N, n)
pre_labels = KMeans(n_clusters=n_pre, n_init=10, random_state=42).fit_predict(color_features)
pre_groups = defaultdict(list)
for i, lbl in enumerate(pre_labels):
    pre_groups[int(lbl)].append(i)

for lbl in sorted(pre_groups.keys()):
    idx = pre_groups[lbl]
    fnames = [image_paths[i].name for i in idx]
    g10_in = [f for f in fnames if f in WATCH_GROUP10]
    merge_in = [f for f in fnames if f in WATCH_SHOULD_MERGE]
    print(f"  预组{lbl} ({len(idx)}张): {fnames[:5]}{'...' if len(fnames) > 5 else ''}")
    if g10_in:
        print(f"    → 含模板群10: {sorted(g10_in)}")
    if merge_in:
        print(f"    → 含应合并组: {sorted(merge_in)}")

# === 第二阶段：组内 HDBSCAN ===
print("\n" + "="*60)
print("第二阶段：组内 HDBSCAN 精细聚类")
all_labels = np.full(n, -1, dtype=int)
cluster_counter = 0
for pre_lbl in sorted(pre_groups.keys()):
    indices = pre_groups[pre_lbl]
    if len(indices) < MIN_CLUSTER_SIZE:
        continue
    sub_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=1,
        metric='euclidean',
        cluster_selection_epsilon=HDBSCAN_EPSILON,
        cluster_selection_method='eom',
    )
    sub_labels = sub_clusterer.fit_predict(dino_features[indices])
    n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    n_noise = sum(1 for s in sub_labels if s == -1)

    # 映射到全局
    for local_idx, global_idx in enumerate(indices):
        if sub_labels[local_idx] != -1:
            all_labels[global_idx] = cluster_counter + int(sub_labels[local_idx])

    # 详细输出
    sub_groups = defaultdict(list)
    for local_idx, global_idx in enumerate(indices):
        sub_groups[sub_labels[local_idx]].append(image_paths[global_idx].name)

    g10_in_pre = [image_paths[i].name for i in indices if image_paths[i].name in WATCH_GROUP10]
    merge_in_pre = [image_paths[i].name for i in indices if image_paths[i].name in WATCH_SHOULD_MERGE]

    if g10_in_pre or merge_in_pre:
        print(f"\n  预组{pre_lbl}: {n_sub}簇 + {n_noise}噪声")
        for slbl in sorted(sub_groups.keys()):
            fnames = sorted(sub_groups[slbl])
            tag = "噪声" if slbl == -1 else f"局部簇{slbl}→全局簇{cluster_counter + slbl}"
            highlight = [f for f in fnames if f in WATCH_GROUP10 or f in WATCH_SHOULD_MERGE]
            if highlight:
                print(f"    {tag}: {fnames}")

    cluster_counter += n_sub

show_watch_groups(all_labels, image_paths, "HDBSCAN后", WATCH_GROUP10)
show_watch_groups(all_labels, image_paths, "HDBSCAN后", WATCH_SHOULD_MERGE)

# === 第三阶段：跨预组簇合并 ===
print("\n" + "="*60)
print("第三阶段：跨预组簇合并")
from scipy.spatial.distance import pdist, squareform

unique_labels = sorted(set(all_labels) - {-1})
print(f"  合并前: {len(unique_labels)} 个簇")

if len(unique_labels) > 1:
    centroids = {}
    for lbl in unique_labels:
        member_idx = np.where(all_labels == lbl)[0]
        centroids[lbl] = dino_features[member_idx].mean(axis=0)

    label_list = list(centroids.keys())
    centroid_matrix = np.array([centroids[l] for l in label_list])
    dist_matrix = squareform(pdist(centroid_matrix, metric='euclidean'))

    # 查看关注组的簇之间的距离
    g10_labels = set()
    merge_labels = set()
    for i, p in enumerate(image_paths):
        if p.name in WATCH_GROUP10 and all_labels[i] != -1:
            g10_labels.add(all_labels[i])
        if p.name in WATCH_SHOULD_MERGE and all_labels[i] != -1:
            merge_labels.add(all_labels[i])

    print(f"  模板群10涉及的簇: {sorted(g10_labels)}")
    print(f"  应合并组涉及的簇: {sorted(merge_labels)}")

    # 打印这些簇之间的距离
    all_watch_labels = sorted(g10_labels | merge_labels)
    if len(all_watch_labels) > 1:
        print(f"\n  关注簇之间的质心距离（merge_threshold={MERGE_THRESHOLD}）:")
        for i, l1 in enumerate(all_watch_labels):
            for j, l2 in enumerate(all_watch_labels):
                if j > i:
                    idx1 = label_list.index(l1)
                    idx2 = label_list.index(l2)
                    d = dist_matrix[idx1, idx2]
                    will_merge = "← 会合并!" if d < MERGE_THRESHOLD else ""
                    # 找出每个簇包含哪些关注图片
                    members1 = [p.name for k, p in enumerate(image_paths) if all_labels[k] == l1 and (p.name in WATCH_GROUP10 or p.name in WATCH_SHOULD_MERGE)]
                    members2 = [p.name for k, p in enumerate(image_paths) if all_labels[k] == l2 and (p.name in WATCH_GROUP10 or p.name in WATCH_SHOULD_MERGE)]
                    print(f"    簇{l1}{members1} ↔ 簇{l2}{members2}: {d:.4f} {will_merge}")

    # Union-Find 合并
    parent = {l: l for l in label_list}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merge_count = 0
    merge_details = []
    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            if dist_matrix[i, j] < MERGE_THRESHOLD:
                ri, rj = find(label_list[i]), find(label_list[j])
                if ri != rj:
                    parent[rj] = ri
                    merge_count += 1
                    # 检查是否涉及关注图片
                    m1 = [p.name for k, p in enumerate(image_paths) if all_labels[k] == label_list[i]]
                    m2 = [p.name for k, p in enumerate(image_paths) if all_labels[k] == label_list[j]]
                    g10_m = [f for f in m1+m2 if f in WATCH_GROUP10]
                    if g10_m:
                        merge_details.append(f"    簇{label_list[i]} + 簇{label_list[j]} (dist={dist_matrix[i,j]:.4f}), 含: {g10_m}")

    print(f"\n  合并次数: {merge_count}")
    if merge_details:
        print("  涉及模板群10的合并:")
        for detail in merge_details:
            print(detail)

    # 重映射
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

show_watch_groups(all_labels, image_paths, "合并后", WATCH_GROUP10)
show_watch_groups(all_labels, image_paths, "合并后", WATCH_SHOULD_MERGE)

# === 第四阶段：噪声点回收 ===
print("\n" + "="*60)
print("第四阶段：噪声点回收")
noise_idx = np.where(all_labels == -1)[0]
cluster_labels_now = sorted(set(all_labels) - {-1})
print(f"  噪声点: {len(noise_idx)}, 现有簇: {len(cluster_labels_now)}")

# 列出噪声中的关注图片
noise_watch = [image_paths[i].name for i in noise_idx if image_paths[i].name in WATCH_GROUP10 or image_paths[i].name in WATCH_SHOULD_MERGE]
if noise_watch:
    print(f"  关注图片中的噪声: {sorted(noise_watch)}")

if len(noise_idx) > 0 and len(cluster_labels_now) > 0:
    centroids_final = {}
    for lbl in cluster_labels_now:
        member_idx = np.where(all_labels == lbl)[0]
        centroids_final[lbl] = dino_features[member_idx].mean(axis=0)
    centroid_labels = list(centroids_final.keys())
    centroid_vecs = np.array([centroids_final[l] for l in centroid_labels])

    rescued_details = []
    for ni in noise_idx:
        dists = np.linalg.norm(centroid_vecs - dino_features[ni], axis=1)
        best_j = int(np.argmin(dists))
        if dists[best_j] < MERGE_THRESHOLD:
            target_lbl = centroid_labels[best_j]
            fname = image_paths[ni].name
            if fname in WATCH_GROUP10 or fname in WATCH_SHOULD_MERGE:
                # 找目标簇包含的关注图片
                target_members = [image_paths[k].name for k in range(n) if all_labels[k] == target_lbl]
                rescued_details.append(f"    {fname} → 簇{target_lbl} (dist={dists[best_j]:.4f}), 簇成员: {target_members[:5]}...")
            all_labels[ni] = target_lbl

    if rescued_details:
        print("  关注图片的回收详情:")
        for d in rescued_details:
            print(d)

show_watch_groups(all_labels, image_paths, "回收后", WATCH_GROUP10)
show_watch_groups(all_labels, image_paths, "回收后", WATCH_SHOULD_MERGE)

# === 第五阶段：布局分裂 ===
print("\n" + "="*60)
print("第五阶段：布局分裂")
edge_asymmetries = np.array([compute_edge_asymmetry(p) for p in image_paths])
new_labels = all_labels.copy()
next_label = max(all_labels) + 1 if len(all_labels) > 0 else 0
for lbl in sorted(set(all_labels) - {-1}):
    member_idx = np.where(all_labels == lbl)[0]
    if len(member_idx) < 2 * MIN_CLUSTER_SIZE:
        continue
    asym_vals = edge_asymmetries[member_idx]
    low_mask = asym_vals < LAYOUT_SPLIT_THRESHOLD
    high_mask = ~low_mask
    n_low = int(np.sum(low_mask))
    n_high = int(np.sum(high_mask))
    if n_low >= MIN_CLUSTER_SIZE and n_high >= MIN_CLUSTER_SIZE:
        fnames = [image_paths[i].name for i in member_idx]
        g10_in = [f for f in fnames if f in WATCH_GROUP10]
        merge_in = [f for f in fnames if f in WATCH_SHOULD_MERGE]
        if g10_in or merge_in:
            low_names = [image_paths[member_idx[k]].name for k in range(len(member_idx)) if low_mask[k]]
            high_names = [image_paths[member_idx[k]].name for k in range(len(member_idx)) if high_mask[k]]
            print(f"  簇{lbl} 拆分: 低不对称({n_low}): {sorted(low_names)[:5]}..., 高不对称({n_high}): {sorted(high_names)[:5]}...")
        for local_i, global_i in enumerate(member_idx):
            if high_mask[local_i]:
                new_labels[global_i] = next_label
        next_label += 1
all_labels = new_labels

show_watch_groups(all_labels, image_paths, "布局分裂后", WATCH_GROUP10)
show_watch_groups(all_labels, image_paths, "布局分裂后", WATCH_SHOULD_MERGE)

# === 最终结果 ===
print("\n" + "="*60)
print("最终结果")
final_groups = defaultdict(list)
for i, lbl in enumerate(all_labels):
    final_groups[lbl].append(image_paths[i].name)

for lbl in sorted(final_groups.keys()):
    fnames = sorted(final_groups[lbl])
    tag = "噪声" if lbl == -1 else f"模板群{lbl:02d}"
    g10_in = [f for f in fnames if f in WATCH_GROUP10]
    merge_in = [f for f in fnames if f in WATCH_SHOULD_MERGE]
    marker = ""
    if g10_in:
        marker += " ★G10"
    if merge_in:
        marker += " ★MERGE"
    print(f"  {tag} ({len(fnames)}张){marker}: {fnames}")
