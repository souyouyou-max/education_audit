#!/usr/bin/env python3
"""诊断：图片 26/29/33/39 + 18/28/36/40 之间的 DINOv2 距离"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

IMAGE_DIR = Path("/Users/songyangyang/Desktop/education_audit/pic")
MODEL_NAME = "facebook/dinov2-large"

# 高中单页: 26, 33  |  中学对折: 29, 39, 18, 28, 36, 40
FILES = [
    "图片 26.png", "图片 33.png",       # 高中
    "图片 29.png", "图片 39.png",       # 中学（同预组）
    "图片 18.png", "图片 28.png",       # 中学（不同预组）
    "图片 36.png", "图片 40.png",       # 中学（不同预组）
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

paths = [IMAGE_DIR / f for f in FILES]
images = [Image.open(p).convert("RGB") for p in paths]
inputs = processor(images=images, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    feats = outputs.last_hidden_state[:, 0]
feats = normalize(feats.cpu().numpy())

names = [f.replace("图片 ", "").replace(".png", "") for f in FILES]

print(f"\n{'':>6s}", end="")
for n in names:
    print(f"{n:>8s}", end="")
print()
for i in range(len(names)):
    print(f"{names[i]:>6s}", end="")
    for j in range(len(names)):
        d = np.linalg.norm(feats[i] - feats[j])
        print(f"{d:8.4f}", end="")
    print()

# 高中 vs 中学 的平均距离
gaoz = feats[:2]   # 26, 33
zhong_same = feats[2:4]  # 29, 39
zhong_other = feats[4:]  # 18, 28, 36, 40

print("\n--- 距离统计 ---")
# 高中内部
d_gz = np.linalg.norm(gaoz[0] - gaoz[1])
print(f"高中内部 (26↔33): {d_gz:.4f}")

# 中学内部（同预组）
d_zs = np.linalg.norm(zhong_same[0] - zhong_same[1])
print(f"中学内部-同预组 (29↔39): {d_zs:.4f}")

# 高中↔中学
cross = []
for g in gaoz:
    for z in np.concatenate([zhong_same, zhong_other]):
        cross.append(np.linalg.norm(g - z))
print(f"高中↔中学 min={min(cross):.4f} max={max(cross):.4f} mean={np.mean(cross):.4f}")

# 中学全体内部
all_zhong = np.concatenate([zhong_same, zhong_other])
zhong_dists = []
for i in range(len(all_zhong)):
    for j in range(i+1, len(all_zhong)):
        zhong_dists.append(np.linalg.norm(all_zhong[i] - all_zhong[j]))
print(f"中学全体内部 min={min(zhong_dists):.4f} max={max(zhong_dists):.4f} mean={np.mean(zhong_dists):.4f}")
