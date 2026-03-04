#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
距离分析脚本 - 寻找最优 epsilon
"""
import torch
from PIL import Image
from pathlib import Path
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from sklearn.preprocessing import normalize
import hdbscan

image_dir = Path("/Users/souyouyou/.openclaw/workspace/education_audit/pic")
model_name = "facebook/dinov2-large"

focus_files = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png",
               "8.png", "9.png", "10.png", "11.png", "12.png", "13.png", "14.png"]
focus_paths = [image_dir / f for f in focus_files]

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()

feats_list = []
for p in focus_paths:
    img = Image.open(p).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feat = outputs.last_hidden_state[:, 0]
    feats_list.append(feat.cpu().float().numpy())

embeddings = normalize(np.concatenate(feats_list))

from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(embeddings, metric='euclidean'))

print("\nPairwise distances:")
for i in range(14):
    for j in range(i+1, 14):
        print(f"  dist({focus_files[i]}, {focus_files[j]}) = {dist_matrix[i,j]:.5f}")

# Expected groups
expected_groups = {
    'A': [0,1,2],   # 1,2,3
    'B': [4,5,6],   # 5,6,7
    'C': [7,8],     # 8,9
    'D': [9,10],    # 10,11
    'E': [11,12],   # 12,13
    'singleton_4': [3],
    'singleton_14': [13],
}

print("\n--- Within-group max distances ---")
for g, idxs in expected_groups.items():
    if len(idxs) > 1:
        max_d = max(dist_matrix[i,j] for i in idxs for j in idxs if i != j)
        min_d = min(dist_matrix[i,j] for i in idxs for j in idxs if i != j)
        print(f"  {g}: min={min_d:.5f}, max={max_d:.5f}")

print("\n--- Between-group min distances ---")
group_names = list(expected_groups.keys())
for gi in range(len(group_names)):
    for gj in range(gi+1, len(group_names)):
        name_i, name_j = group_names[gi], group_names[gj]
        idxs_i, idxs_j = expected_groups[name_i], expected_groups[name_j]
        min_d = min(dist_matrix[ii,jj] for ii in idxs_i for jj in idxs_j)
        if min_d < 0.12:
            print(f"  {name_i}-{name_j}: min_cross_dist={min_d:.5f}")

print("\n--- Trying different epsilon values ---")
for eps in [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09]:
    for method in ['eom', 'leaf']:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=eps,
            cluster_selection_method=method,
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Check if matches expected
        label_map = {f: labels[i] for i, f in enumerate(focus_files)}
        
        def same_cluster(files):
            lbls = [label_map[f] for f in files]
            return len(set(lbls)) == 1 and lbls[0] != -1
        
        def is_singleton(f):
            lbl = label_map[f]
            others_same = [g for g in focus_files if g != f and label_map[g] == lbl]
            return lbl == -1 or len(others_same) == 0
        
        def no_contamination(group_files, all_expected_files):
            lbl = label_map[group_files[0]]
            if lbl == -1:
                return False
            others_in_cluster = [g for g in focus_files if label_map[g] == lbl and g not in all_expected_files]
            return len(others_in_cluster) == 0
        
        all_expected_members = [f for g, idxs in expected_groups.items() 
                                 if g not in ('singleton_4', 'singleton_14')
                                 for f in [focus_files[i] for i in idxs]]
        
        a_ok = same_cluster(['1.png','2.png','3.png']) and no_contamination(['1.png','2.png','3.png'], ['1.png','2.png','3.png'])
        b_ok = same_cluster(['5.png','6.png','7.png']) and no_contamination(['5.png','6.png','7.png'], ['5.png','6.png','7.png'])
        c_ok = same_cluster(['8.png','9.png']) and no_contamination(['8.png','9.png'], ['8.png','9.png'])
        d_ok = same_cluster(['10.png','11.png']) and no_contamination(['10.png','11.png'], ['10.png','11.png'])
        e_ok = same_cluster(['12.png','13.png']) and no_contamination(['12.png','13.png'], ['12.png','13.png'])
        s4_ok = is_singleton('4.png')
        s14_ok = is_singleton('14.png')
        
        score = sum([a_ok, b_ok, c_ok, d_ok, e_ok, s4_ok, s14_ok])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(1 for l in labels if l == -1)
        
        markers = ''.join(['A' if a_ok else 'a', 'B' if b_ok else 'b', 'C' if c_ok else 'c',
                           'D' if d_ok else 'd', 'E' if e_ok else 'e',
                           '4' if s4_ok else '-', 'f' if s14_ok else '-'])
        
        if score >= 5:
            print(f"  eps={eps:.3f} {method:4s}: score={score}/7 groups={markers} n_clusters={n_clusters} n_noise={n_noise}")
            # Print label mapping
            for f in focus_files:
                print(f"    {f}: {label_map[f]}")
