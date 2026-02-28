#!/usr/bin/env python3
"""诊断：对比 HDBSCAN eom vs leaf 在预组4的聚类效果"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
import hdbscan
import cv2

IMAGE_DIR = Path("/Users/songyangyang/Desktop/education_audit/pic")
MODEL_NAME = "facebook/dinov2-large"

exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
image_paths = sorted(p for ext in exts for p in IMAGE_DIR.rglob(ext))

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

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

color_features = np.array([get_color_red_feature(p) for p in image_paths])

dino_features = []
batch_size = 4
for start in range(0, len(image_paths), batch_size):
    batch = image_paths[start:start + batch_size]
    images = [Image.open(p).convert("RGB") for p in batch]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 0]
    dino_features.append(feats.cpu().numpy())
dino_features = normalize(np.concatenate(dino_features))

# KMeans 预分组
from collections import defaultdict
pre_labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(color_features)
pre_groups = defaultdict(list)
for i, lbl in enumerate(pre_labels):
    pre_groups[int(lbl)].append(i)

# 找到包含模板群10图片的预组
target_group = None
for lbl, indices in pre_groups.items():
    fnames = [image_paths[i].name for i in indices]
    if "图片 1.png" in fnames:
        target_group = lbl
        break

print(f"目标预组: {target_group}")
indices = pre_groups[target_group]
fnames = [image_paths[i].name for i in indices]
print(f"包含 {len(indices)} 张图片: {sorted(fnames)}")

sub_features = dino_features[indices]

# 测试不同 HDBSCAN 配置
print("\n" + "="*60)

for method in ['eom', 'leaf']:
    for epsilon in [0.06, 0.08, 0.10, 0.12]:
        for min_cs in [2, 3]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cs,
                min_samples=1,
                metric='euclidean',
                cluster_selection_epsilon=epsilon,
                cluster_selection_method=method,
            )
            labels = clusterer.fit_predict(sub_features)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = sum(1 for l in labels if l == -1)

            groups = defaultdict(list)
            for i, lbl in enumerate(labels):
                groups[lbl].append(fnames[i])

            # 只显示有意义的结果
            if n_clusters > 1 or method == 'eom':
                print(f"\n{method} eps={epsilon} min_cs={min_cs}: {n_clusters}簇 + {n_noise}噪声")
                for lbl in sorted(groups.keys()):
                    tag = "噪声" if lbl == -1 else f"簇{lbl}"
                    print(f"  {tag}: {sorted(groups[lbl])}")

# 也测试 Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
print("\n" + "="*60)
print("Agglomerative Clustering (complete linkage):")
for threshold in [0.35, 0.40, 0.45, 0.50, 0.55]:
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='euclidean',
        linkage='complete'
    )
    labels = agg.fit_predict(sub_features)
    n_clusters = len(set(labels))

    groups = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(fnames[i])

    print(f"\n  threshold={threshold:.2f}: {n_clusters} 组")
    for lbl in sorted(groups.keys()):
        print(f"    组{lbl}: {sorted(groups[lbl])}")

# 测试合并阈值对应合并组的影响
print("\n" + "="*60)
print("跨预组合并：图片18/28/36/40 和 图片20/37/38/42 的质心距离")

# 找到两个组的预组和成员
group_a_indices = [i for i in range(len(image_paths)) if image_paths[i].name in {"图片 18.png", "图片 28.png", "图片 36.png", "图片 40.png"}]
group_b_indices = [i for i in range(len(image_paths)) if image_paths[i].name in {"图片 20.png", "图片 37.png", "图片 38.png", "图片 42.png"}]

centroid_a = dino_features[group_a_indices].mean(axis=0)
centroid_b = dino_features[group_b_indices].mean(axis=0)
dist_ab = np.linalg.norm(centroid_a - centroid_b)
print(f"  质心距离: {dist_ab:.4f}")

# 成员之间的距离
print("\n  成员间距离:")
for i in group_a_indices:
    for j in group_b_indices:
        d = np.linalg.norm(dino_features[i] - dino_features[j])
        print(f"    {image_paths[i].name} ↔ {image_paths[j].name}: {d:.4f}")
