#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script: compute pairwise DINOv2 (CLS token) euclidean distances
between specific images to understand why 1.png is not being rescued into
the {2.png, 3.png} cluster.

Also computes distances for {4.png, 10.png, 11.png} which ARE successfully
clustered together, as a baseline comparison.

Usage:
    python debug_distances.py
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel


IMAGE_DIR = Path("/Users/songyangyang/Desktop/education_audit/pic")
MODEL_NAME = "facebook/dinov2-large"

# Group A: the problematic group (1.png is noise, 2.png+3.png cluster together)
GROUP_A_FILES = ["1.png", "2.png", "3.png"]

# Group B: a successfully clustered group, for comparison
GROUP_B_FILES = ["4.png", "10.png", "11.png"]

ALL_FILES = GROUP_A_FILES + GROUP_B_FILES


def extract_features(paths, processor, model, device):
    """Extract CLS token features from DINOv2, then L2-normalize.

    This matches the default behavior of cluster_images_dinov2.py
    (use_mean_pool=False => CLS token, then sklearn normalize).
    """
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token is at position 0
        features = outputs.last_hidden_state[:, 0]
    features_np = features.cpu().numpy()
    # L2 normalize (same as the clustering script)
    features_np = normalize(features_np)
    return features_np


def pairwise_euclidean(features, names):
    """Compute and print pairwise euclidean distances."""
    n = len(names)
    print(f"\n{'':>10s}", end="")
    for name in names:
        print(f"{name:>10s}", end="")
    print()

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        print(f"{names[i]:>10s}", end="")
        for j in range(n):
            d = np.linalg.norm(features[i] - features[j])
            dist_matrix[i, j] = d
            print(f"{d:10.4f}", end="")
        print()

    return dist_matrix


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    # Verify images exist
    image_paths = []
    for fname in ALL_FILES:
        p = IMAGE_DIR / fname
        if not p.exists():
            print(f"WARNING: {p} does not exist, skipping")
        else:
            image_paths.append(p)

    if len(image_paths) == 0:
        print("No images found, exiting.")
        return

    found_names = [p.name for p in image_paths]
    print(f"\nImages found: {found_names}")

    # Load model
    print(f"\nLoading model {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print("Model loaded.")

    # Extract features
    print("\nExtracting CLS token features...")
    features = extract_features(image_paths, processor, model, device)
    print(f"Feature shape: {features.shape}")

    # Full pairwise distance matrix (all 6 images)
    print("\n" + "=" * 70)
    print("FULL PAIRWISE EUCLIDEAN DISTANCE MATRIX (L2-normalized CLS tokens)")
    print("=" * 70)
    pairwise_euclidean(features, found_names)

    # Focused: Group A (problematic)
    group_a_indices = [i for i, name in enumerate(found_names) if name in GROUP_A_FILES]
    if len(group_a_indices) >= 2:
        print("\n" + "=" * 70)
        print("GROUP A: {1.png, 2.png, 3.png} -- PROBLEMATIC (1.png not rescued)")
        print("=" * 70)
        group_a_feats = features[group_a_indices]
        group_a_names = [found_names[i] for i in group_a_indices]
        pairwise_euclidean(group_a_feats, group_a_names)

        # Compute centroid of {2.png, 3.png} and distance from 1.png to it
        idx_2 = [i for i, n in enumerate(group_a_names) if n == "2.png"]
        idx_3 = [i for i, n in enumerate(group_a_names) if n == "3.png"]
        idx_1 = [i for i, n in enumerate(group_a_names) if n == "1.png"]

        if idx_2 and idx_3 and idx_1:
            centroid_23 = (group_a_feats[idx_2[0]] + group_a_feats[idx_3[0]]) / 2.0
            # Re-normalize centroid (since mean of L2-normed vectors is not L2-normed)
            centroid_23_norm = centroid_23 / np.linalg.norm(centroid_23)

            dist_1_to_centroid = np.linalg.norm(group_a_feats[idx_1[0]] - centroid_23)
            dist_1_to_centroid_renorm = np.linalg.norm(group_a_feats[idx_1[0]] - centroid_23_norm)

            print(f"\nCentroid of {{2.png, 3.png}} (raw mean, NOT re-normalized):")
            print(f"  dist(1.png -> centroid_{{2,3}})          = {dist_1_to_centroid:.4f}")
            print(f"  dist(1.png -> centroid_{{2,3}} renormed) = {dist_1_to_centroid_renorm:.4f}")
            print(f"\n  Current merge_threshold / rescue_threshold = 0.22")
            if dist_1_to_centroid < 0.22:
                print(f"  => 1.png WOULD be rescued (dist {dist_1_to_centroid:.4f} < 0.22)")
            else:
                print(f"  => 1.png would NOT be rescued (dist {dist_1_to_centroid:.4f} >= 0.22)")
                print(f"  Suggested threshold to rescue: > {dist_1_to_centroid:.4f}")

    # Focused: Group B (successfully clustered)
    group_b_indices = [i for i, name in enumerate(found_names) if name in GROUP_B_FILES]
    if len(group_b_indices) >= 2:
        print("\n" + "=" * 70)
        print("GROUP B: {4.png, 10.png, 11.png} -- BASELINE (successfully clustered)")
        print("=" * 70)
        group_b_feats = features[group_b_indices]
        group_b_names = [found_names[i] for i in group_b_indices]
        pairwise_euclidean(group_b_feats, group_b_names)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if len(group_a_indices) >= 2:
        group_a_feats = features[group_a_indices]
        group_a_names = [found_names[i] for i in group_a_indices]
        max_a = 0
        for i in range(len(group_a_indices)):
            for j in range(i + 1, len(group_a_indices)):
                d = np.linalg.norm(group_a_feats[i] - group_a_feats[j])
                max_a = max(max_a, d)
        print(f"Group A max pairwise dist: {max_a:.4f}")

    if len(group_b_indices) >= 2:
        group_b_feats = features[group_b_indices]
        group_b_names = [found_names[i] for i in group_b_indices]
        max_b = 0
        for i in range(len(group_b_indices)):
            for j in range(i + 1, len(group_b_indices)):
                d = np.linalg.norm(group_b_feats[i] - group_b_feats[j])
                max_b = max(max_b, d)
        print(f"Group B max pairwise dist: {max_b:.4f}")

    print(f"\nCurrent merge_threshold = 0.22")
    print("If Group A max dist >> 0.22, raising the threshold may help but could")
    print("also cause false merges in other groups.")
    print("\nConsider also checking:")
    print("  - Which KMeans pre-cluster each image lands in (color-based)")
    print("  - Whether 1.png is in a different pre-cluster than 2.png/3.png")
    print("  - The HDBSCAN sub-cluster labels before noise rescue")


if __name__ == "__main__":
    main()
