#!/usr/bin/env python3
"""诊断：用 leaf 方法后，所有簇之间的质心距离，找出合适的合并阈值"""
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

KMEANS_N = 5
HDBSCAN_EPSILON = 0.12
MIN_CLUSTER_SIZE = 2

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
n = len(image_paths)

# Stage 1: KMeans
pre_labels = KMeans(n_clusters=min(KMEANS_N, n), n_init=10, random_state=42).fit_predict(color_features)
pre_groups = defaultdict(list)
for i, lbl in enumerate(pre_labels):
    pre_groups[int(lbl)].append(i)

# Stage 2: HDBSCAN leaf
all_labels = np.full(n, -1, dtype=int)
cluster_counter = 0
for indices in pre_groups.values():
    if len(indices) < MIN_CLUSTER_SIZE:
        continue
    sub_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=1,
        metric='euclidean',
        cluster_selection_epsilon=HDBSCAN_EPSILON,
        cluster_selection_method='leaf',
    )
    sub_labels = sub_clusterer.fit_predict(dino_features[indices])
    for local_idx, global_idx in enumerate(indices):
        if sub_labels[local_idx] != -1:
            all_labels[global_idx] = cluster_counter + int(sub_labels[local_idx])
    cluster_counter += len(set(sub_labels)) - (1 if -1 in sub_labels else 0)

# 打印 HDBSCAN 后的簇
print("=== HDBSCAN leaf 后的簇 ===")
hdbscan_groups = defaultdict(list)
for i, lbl in enumerate(all_labels):
    hdbscan_groups[lbl].append(image_paths[i].name)

for lbl in sorted(hdbscan_groups.keys()):
    fnames = sorted(hdbscan_groups[lbl])
    tag = "噪声" if lbl == -1 else f"簇{lbl:02d}"
    print(f"  {tag} ({len(fnames)}张): {fnames}")

# 计算所有簇对之间的质心距离
from scipy.spatial.distance import pdist, squareform
unique_labels = sorted(set(all_labels) - {-1})
centroids = {}
for lbl in unique_labels:
    member_idx = np.where(all_labels == lbl)[0]
    centroids[lbl] = dino_features[member_idx].mean(axis=0)

label_list = list(centroids.keys())
centroid_matrix = np.array([centroids[l] for l in label_list])
dist_matrix = squareform(pdist(centroid_matrix, metric='euclidean'))

# 列出所有 < 0.40 的簇对
print(f"\n=== 所有簇对质心距离 < 0.40 ===")
pairs = []
for i in range(len(label_list)):
    for j in range(i+1, len(label_list)):
        d = dist_matrix[i, j]
        if d < 0.40:
            m1 = sorted(hdbscan_groups[label_list[i]])
            m2 = sorted(hdbscan_groups[label_list[j]])
            pairs.append((d, label_list[i], label_list[j], m1, m2))

pairs.sort()
for d, l1, l2, m1, m2 in pairs:
    print(f"  簇{l1:02d} ↔ 簇{l2:02d}: {d:.4f}")
    print(f"    簇{l1:02d}: {m1}")
    print(f"    簇{l2:02d}: {m2}")
