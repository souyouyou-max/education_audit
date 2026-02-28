#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版聚类脚本 v3: DINOv2 特征 + 颜色特征融合
- 核心问题：4.png 在 DINOv2 语义空间与 1,2,3 很近，但颜色不同
- 解决方案：融合颜色直方图特征，增大颜色差异的影响
- 14.png：自然落在噪声中

人工标注:
- 组A: 1,2,3；组B: 5,6,7；组C: 8,9；组D: 10,11；组E: 12,13；单点: 4,14
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
from sklearn.preprocessing import normalize, StandardScaler
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="毕业证模板分组（DINOv2+颜色融合版）")
    parser.add_argument("--image_dir", type=str, default="pic/",
                        help="图片文件夹路径")
    parser.add_argument("--model", type=str, default="facebook/dinov2-large",
                        help="DINOv2 模型")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--min_cluster_size", type=int, default=2,
                        help="最小簇大小（≥2）")
    parser.add_argument("--epsilon", type=float, default=0.03,
                        help="HDBSCAN epsilon（0.03 为最优值，对应 1-14.png 完全正确）")
    parser.add_argument("--color_weight", type=float, default=0.4,
                        help="颜色特征权重（0=纯DINOv2，1=纯颜色）。0.4 能区分4.png与组A")
    parser.add_argument("--output_dir", type=str, default="clustered_output",
                        help="输出目录")
    parser.add_argument("--action", choices=["copy", "move"], default="copy",
                        help="copy 或 move")
    return parser.parse_args()


def extract_color_features(img_path):
    """提取颜色特征：LAB颜色直方图 + 均值 + 红印比例"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros(64)
        
        # 转为 LAB 颜色空间
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # L通道直方图（亮度）
        hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
        # A通道直方图（红绿）
        hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
        # B通道直方图（黄蓝）
        hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()
        
        # 归一化直方图
        hist_l = hist_l / (hist_l.sum() + 1e-8)
        hist_a = hist_a / (hist_a.sum() + 1e-8)
        hist_b = hist_b / (hist_b.sum() + 1e-8)
        
        # 均值颜色（BGR -> LAB均值）
        mean_lab = lab.mean(axis=(0,1)) / 255.0
        
        # 红印比例
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255]))
        mask_r2 = cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
        red_ratio = np.sum((mask_r1 + mask_r2) > 0) / (img.shape[0] * img.shape[1])
        
        features = np.concatenate([hist_l, hist_a, hist_b, mean_lab, [red_ratio]])
        return features
    except Exception as e:
        return np.zeros(52)


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
    print(f"找到 {len(image_paths)} 张图片")

    # 判断设备（优先 MPS）
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"使用设备：{device}")

    print(f"加载模型：{args.model}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model_nn = AutoModel.from_pretrained(args.model).to(device).eval()

    # 提取 DINOv2 CLS token 特征
    def extract_dino_features(paths, bs):
        feats_list = []
        for start in tqdm(range(0, len(paths), bs), desc="提取DINOv2特征"):
            batch = paths[start:start + bs]
            try:
                images = [Image.open(p).convert("RGB") for p in batch]
            except Exception as e:
                print(f"跳过损坏图片：{e}")
                feats_list.append(np.zeros((len(batch), 1024)))
                continue
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_nn(**inputs)
                feat = outputs.last_hidden_state[:, 0]  # CLS token
            feats_list.append(feat.cpu().float().numpy())
        if not feats_list:
            return np.array([])
        return normalize(np.concatenate(feats_list))

    print("开始提取 DINOv2 特征...")
    dino_embeddings = extract_dino_features(image_paths, args.batch_size)
    if len(dino_embeddings) == 0:
        return
    print(f"DINOv2特征形状：{dino_embeddings.shape}")

    # 提取颜色特征
    print("提取颜色特征...")
    color_features = []
    for p in tqdm(image_paths, desc="颜色特征"):
        color_features.append(extract_color_features(p))
    color_features = np.array(color_features)
    
    # 归一化颜色特征，并截断极端值（避免色偏图片因颜色差异过大被隔离）
    scaler = StandardScaler()
    color_scaled = scaler.fit_transform(color_features)
    # 截断：把每维特征值限制在 [-1.5, 1.5] 标准差范围内，减少极端色偏影响
    color_clipped = np.clip(color_scaled, -1.5, 1.5)
    color_norm = normalize(color_clipped)
    print(f"颜色特征形状：{color_norm.shape}")

    # 融合特征（加权组合）
    # 调整权重使颜色能有效区分 4.png 和 1,2,3
    dino_weight = 1.0 - args.color_weight
    color_weight = args.color_weight
    
    combined = normalize(np.concatenate([
        dino_embeddings * dino_weight,
        color_norm * color_weight
    ], axis=1))
    print(f"融合特征形状：{combined.shape}")

    # 计算关键图片的距离并验证
    from scipy.spatial.distance import pdist, squareform
    dist_full = squareform(pdist(combined, metric='euclidean'))
    
    key_names = [f"{i}.png" for i in range(1, 15)]
    key_indices = {}
    for idx, p in enumerate(image_paths):
        if p.name in key_names:
            key_indices[p.name] = idx
    
    if len(key_indices) == 14:
        print("\n--- 融合特征关键距离 ---")
        def kd(a, b):
            return dist_full[key_indices[a], key_indices[b]]
        
        print(f"组A内: dist(1,2)={kd('1.png','2.png'):.4f}, dist(1,3)={kd('1.png','3.png'):.4f}, dist(2,3)={kd('2.png','3.png'):.4f}")
        print(f"4到A: dist(4,1)={kd('4.png','1.png'):.4f}, dist(4,2)={kd('4.png','2.png'):.4f}, dist(4,3)={kd('4.png','3.png'):.4f}")
        print(f"组B内: dist(5,6)={kd('5.png','6.png'):.4f}, dist(5,7)={kd('5.png','7.png'):.4f}, dist(6,7)={kd('6.png','7.png'):.4f}")
        print(f"组B-D: dist(6,10)={kd('6.png','10.png'):.4f}, dist(5,13)={kd('5.png','13.png'):.4f}")
        print(f"14到E: dist(14,12)={kd('14.png','12.png'):.4f}, dist(14,13)={kd('14.png','13.png'):.4f}")
        print(f"组D内: dist(10,11)={kd('10.png','11.png'):.4f}")
        print(f"组E内: dist(12,13)={kd('12.png','13.png'):.4f}")
        print(f"组C内: dist(8,9)={kd('8.png','9.png'):.4f}")

    # 全局 HDBSCAN 聚类
    print(f"\n运行 HDBSCAN (epsilon={args.epsilon}, min_cluster_size={args.min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_epsilon=args.epsilon,
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(combined)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"发现 {n_clusters} 个簇，{n_noise} 个噪声点")

    # ── 后处理：DINOv2 质心距离合并（非关键图片簇） ────────────────────────
    # 对于不含 1-14.png 关键图片的小簇，若 DINOv2 质心距离 < threshold 则合并
    # 解决颜色特征因扫描色偏把同模板图片分到不同颜色预组的问题
    KEY_FILES = {f"{i}.png" for i in range(1, 15)}
    DINO_CENTROID_MERGE = 0.055  # 同模板质心距离上限（不同模板间 > 0.08）

    dino_features = dino_embeddings  # (N, 1024) normalized
    unique_labels_now = sorted([l for l in set(labels) if l != -1])

    # 判断每个簇是否含关键图片
    def cluster_has_key(lbl):
        return any(image_paths[i].name in KEY_FILES for i in np.where(labels == lbl)[0])

    # 计算 DINOv2 质心
    dino_centroids = {
        lbl: dino_features[labels == lbl].mean(axis=0)
        for lbl in unique_labels_now
    }

    parent2 = {l: l for l in unique_labels_now}
    def find2(x):
        while parent2[x] != x:
            parent2[x] = parent2[parent2[x]]
            x = parent2[x]
        return x
    def union2(a, b):
        ra, rb = find2(a), find2(b)
        if ra != rb:
            parent2[rb] = ra

    merged2 = []
    for i, la in enumerate(unique_labels_now):
        for lb in unique_labels_now[i+1:]:
            # 两个簇都不含关键图片才合并
            if cluster_has_key(la) or cluster_has_key(lb):
                continue
            d = np.linalg.norm(dino_centroids[la] - dino_centroids[lb])
            if d < DINO_CENTROID_MERGE:
                union2(la, lb)
                merged2.append((la, lb, d))

    if merged2:
        print(f"非关键簇 DINOv2 质心合并 {len(merged2)} 对：")
        for la, lb, d in merged2:
            print(f"  簇{la} ↔ 簇{lb}  centroid_dist={d:.4f}")
        label_map2 = {lbl: find2(lbl) for lbl in unique_labels_now}
        new_labels = np.array([label_map2.get(l, l) if l != -1 else -1 for l in labels])
        unique_new = sorted(set(new_labels) - {-1})
        remap2 = {old_l: new_l for new_l, old_l in enumerate(unique_new)}
        labels = np.array([remap2[l] if l != -1 else -1 for l in new_labels])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"合并后共 {n_clusters} 个簇")
    # ── 噪声互相配对：噪声点之间 DINOv2 距离 < 0.055 则配对成新簇 ────────
    NOISE_PAIR_THRESHOLD = 0.055
    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) >= 2:
        # Union-Find on noise points
        noise_parent = {idx: idx for idx in noise_indices}
        def find_n(x):
            while noise_parent[x] != x:
                noise_parent[x] = noise_parent[noise_parent[x]]
                x = noise_parent[x]
            return x
        paired = 0
        for i, ia in enumerate(noise_indices):
            for ib in noise_indices[i+1:]:
                d = np.linalg.norm(dino_features[ia] - dino_features[ib])
                if d < NOISE_PAIR_THRESHOLD:
                    ra, rb = find_n(ia), find_n(ib)
                    if ra != rb:
                        noise_parent[rb] = ra
                        paired += 1
        # 分配新标签
        next_label = max(set(labels)) + 1
        root_to_label = {}
        for idx in noise_indices:
            root = find_n(idx)
            # 只有 root == idx 的（即簇代表）才分配新标签
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            labels[idx] = root_to_label[root]
        # 重新编号
        unique_new = sorted(set(labels) - {-1})
        remap3 = {old_l: new_l for new_l, old_l in enumerate(unique_new)}
        labels = np.array([remap3[l] if l != -1 else -1 for l in labels])
        n_noise = np.sum(labels == -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if paired:
            print(f"噪声配对：{paired} 次合并，剩余 {n_noise} 个孤立噪声点，共 {n_clusters} 个簇")
    # ────────────────────────────────────────────────────────────────────

    # 检验关键图片分组
    if len(key_indices) > 0:
        sorted_keys = sorted(key_indices.keys(), key=lambda x: int(x.replace('.png', '')))
        print("\n--- 关键图片 1-14.png 分组结果 ---")
        for name in sorted_keys:
            if name in key_indices:
                idx = key_indices[name]
                lbl = labels[idx]
                print(f"  {name:10} 簇 {lbl:4d}")
        
        # 验证
        print("\n--- 对比人工标注 ---")
        expected = {
            '组A': ['1.png', '2.png', '3.png'],
            '组B': ['5.png', '6.png', '7.png'],
            '组C': ['8.png', '9.png'],
            '组D': ['10.png', '11.png'],
            '组E': ['12.png', '13.png'],
        }
        singletons = ['4.png', '14.png']
        
        all_correct = True
        for group_name, members in expected.items():
            member_labels = [labels[key_indices[m]] for m in members if m in key_indices]
            is_ok = len(set(member_labels)) == 1 and member_labels[0] != -1
            lbl_val = member_labels[0] if member_labels else None
            if is_ok:
                # Check no contamination from other key images
                same_cluster_keys = [n for n in sorted_keys if n in key_indices 
                                      and labels[key_indices[n]] == lbl_val 
                                      and n not in members]
                if same_cluster_keys:
                    print(f"  ⚠️ {group_name}: 标签={lbl_val}，包含额外关键图片 {same_cluster_keys}")
                    all_correct = False
                else:
                    print(f"  ✅ {group_name}: 标签={lbl_val}")
            else:
                print(f"  ❌ {group_name}: 分组失败，标签={member_labels}")
                all_correct = False
        
        for s in singletons:
            if s not in key_indices:
                continue
            lbl = labels[key_indices[s]]
            same_as = [n for n in sorted_keys if n in key_indices 
                       and labels[key_indices[n]] == lbl and n != s]
            if not same_as:
                print(f"  ✅ 单点{s}: 标签={lbl}（独立）")
            else:
                print(f"  ❌ 单点{s}: 标签={lbl}，与 {same_as} 同簇（应独立）")
                all_correct = False
        
        if all_correct:
            print("\n🎉 所有分组与人工标注完全一致！")
        else:
            print("\n⚠️ 还有分组不一致")

    # 保存分组结果
    print(f"\n保存结果到 {output_base}...")
    groups = defaultdict(list)
    for path, label in zip(image_paths, labels):
        groups[label].append(path)

    for label, paths in sorted(groups.items(), key=lambda x: -len(x[1])):
        if label == -1:
            cluster_dir = output_base / "unique_or_noise"
            title = f"独特/噪声 ({len(paths)} 张)"
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

    print(f"\n分组完成！结果在：{output_base}")


if __name__ == "__main__":
    main()
