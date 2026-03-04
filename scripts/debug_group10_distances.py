#!/usr/bin/env python3
"""诊断：模板群10 中 16 张图片的 DINOv2 距离矩阵 + 颜色特征"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel
import cv2

IMAGE_DIR = Path("/Users/songyangyang/Desktop/education_audit/pic")
MODEL_NAME = "facebook/dinov2-large"

# 模板群10 的 16 张图片
FILES = [
    "图片 1.png", "图片 2.png", "图片 3.png", "图片 4.png",
    "图片 5.png", "图片 6.png", "图片 7.png", "图片 8.png",
    "图片 16.png", "图片 22.png", "图片 30.png", "图片 34.png",
    "图片 41.png", "图片 43.png", "图片 44.png", "图片 46.png",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

# 过滤存在的文件
valid_files = []
for f in FILES:
    if (IMAGE_DIR / f).exists():
        valid_files.append(f)
    else:
        print(f"文件不存在: {f}")

paths = [IMAGE_DIR / f for f in valid_files]
names = [f.replace("图片 ", "").replace(".png", "") for f in valid_files]

# 提取 DINOv2 特征
print("提取 DINOv2 特征...")
all_feats = []
batch_size = 4
for start in range(0, len(paths), batch_size):
    batch_paths = paths[start:start + batch_size]
    images = [Image.open(p).convert("RGB") for p in batch_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 0]
    all_feats.append(feats.cpu().numpy())

feats = normalize(np.concatenate(all_feats))
n = len(names)

# 打印距离矩阵
print(f"\n=== DINOv2 距离矩阵 ({n} 张图片) ===")
print(f"{'':>6s}", end="")
for name in names:
    print(f"{name:>8s}", end="")
print()

dist_matrix = np.zeros((n, n))
for i in range(n):
    print(f"{names[i]:>6s}", end="")
    for j in range(n):
        d = np.linalg.norm(feats[i] - feats[j])
        dist_matrix[i, j] = d
        print(f"{d:8.4f}", end="")
    print()

# 分析聚类结构
print("\n=== 距离统计 ===")
upper_dists = []
for i in range(n):
    for j in range(i+1, n):
        upper_dists.append((dist_matrix[i, j], names[i], names[j]))

upper_dists.sort()
print(f"\n最近的 10 对:")
for d, a, b in upper_dists[:10]:
    print(f"  {a} ↔ {b}: {d:.4f}")

print(f"\n最远的 10 对:")
for d, a, b in upper_dists[-10:]:
    print(f"  {a} ↔ {b}: {d:.4f}")

print(f"\n距离分布: min={upper_dists[0][0]:.4f} max={upper_dists[-1][0]:.4f} mean={np.mean([x[0] for x in upper_dists]):.4f}")

# 每张图片的平均距离
print(f"\n=== 每张图片到其他图片的平均距离 ===")
for i in range(n):
    others = [dist_matrix[i, j] for j in range(n) if i != j]
    print(f"  {names[i]:>6s}: avg={np.mean(others):.4f} min={np.min(others):.4f} max={np.max(others):.4f}")

# 颜色特征
print(f"\n=== 颜色/红印特征 ===")
for i, p in enumerate(paths):
    img = cv2.imread(str(p))
    if img is None:
        continue
    img_blur = cv2.medianBlur(img, 5)
    mask = cv2.inRange(img_blur, np.array([20,20,20]), np.array([235,235,235]))
    masked_pixels = img_blur.reshape(-1, 3)[mask.reshape(-1) > 0]
    mean_color = np.mean(masked_pixels, axis=0).astype(int) if len(masked_pixels) > 0 else np.mean(img_blur.reshape(-1, 3), axis=0).astype(int)

    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mask_red = (cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                + cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255])))
    red_ratio = np.sum(mask_red > 0) / (img.shape[0] * img.shape[1]) * 100

    # 边缘不对称
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    left_d = edges[:, :w//2].mean()
    right_d = edges[:, w//2:].mean()
    asym = abs(left_d - right_d) / (left_d + right_d + 1e-8)

    print(f"  {names[i]:>6s}: BGR={mean_color}, 红印={red_ratio:.1f}%, 边缘不对称={asym:.4f}, 尺寸={w}x{h}")

# 尝试不同阈值下的聚类效果
print(f"\n=== 不同阈值下的子聚类 ===")
from sklearn.cluster import AgglomerativeClustering

for threshold in [0.20, 0.25, 0.28, 0.30, 0.35, 0.40]:
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='euclidean',
        linkage='complete'
    )
    labels = agg.fit_predict(feats)
    n_clusters = len(set(labels))
    groups = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(names[i])
    print(f"\n  threshold={threshold:.2f} → {n_clusters} 组:")
    for lbl in sorted(groups.keys()):
        print(f"    组{lbl}: {groups[lbl]}")
