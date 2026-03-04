#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze pairwise distances between key images to understand what's happening
"""
import torch
from PIL import Image
from pathlib import Path
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from sklearn.preprocessing import normalize

image_dir = Path("/Users/souyouyou/.openclaw/workspace/education_audit/pic")
model_name = "facebook/dinov2-large"

# Focus images
focus_files = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png",
               "8.png", "9.png", "10.png", "11.png", "12.png", "13.png", "14.png"]
focus_paths = [image_dir / f for f in focus_files]

device = "cpu"
print(f"Loading model {model_name}...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()

print("Extracting features...")
feats_list = []
for p in focus_paths:
    img = Image.open(p).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feat = outputs.last_hidden_state[:, 0]  # CLS token
    feats_list.append(feat.cpu().numpy())

embeddings = normalize(np.concatenate(feats_list))
print(f"Features shape: {embeddings.shape}")

# Compute pairwise distances
from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(embeddings, metric='euclidean'))

print("\nPairwise Euclidean distances (normalized embeddings):")
print(f"{'':12}", end="")
for f in focus_files:
    print(f"{f:>8}", end="")
print()

for i, fi in enumerate(focus_files):
    print(f"{fi:12}", end="")
    for j, fj in enumerate(focus_files):
        print(f"{dist_matrix[i,j]:8.3f}", end="")
    print()

print("\n--- Key distances ---")
groups = {
    "A(1,2)": (0,1), "A(1,3)": (0,2), "A(2,3)": (1,2),
    "B(5,6)": (4,5), "B(5,7)": (4,6), "B(6,7)": (5,6),
    "C(8,9)": (7,8),
    "D(10,11)": (9,10),
    "E(12,13)": (11,12),
    "4-10": (3,9), "4-11": (3,10), "4-14": (3,13),
    "14-A1": (13,0), "14-D10": (13,9),
}
for name, (i,j) in groups.items():
    print(f"  dist({name}): {dist_matrix[i,j]:.4f}")

# What's closest to 4.png?
print("\nClosest to 4.png:")
dists_4 = [(dist_matrix[3,j], focus_files[j]) for j in range(len(focus_files)) if j != 3]
dists_4.sort()
for d, f in dists_4[:5]:
    print(f"  {f}: {d:.4f}")

# What's closest to 14.png?
print("\nClosest to 14.png:")
dists_14 = [(dist_matrix[13,j], focus_files[j]) for j in range(len(focus_files)) if j != 13]
dists_14.sort()
for d, f in dists_14[:5]:
    print(f"  {f}: {d:.4f}")
