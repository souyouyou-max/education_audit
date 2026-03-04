#!/usr/bin/env python3
"""诊断：用宽高比 + 左右不对称度区分单页/对折证书"""
import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path("/Users/songyangyang/Desktop/education_audit/pic")

TEST_FILES = {
    "高中-26": "图片 26.png",
    "高中-33": "图片 33.png",
    "中学-29": "图片 29.png",
    "中学-39": "图片 39.png",
    "中学-18": "图片 18.png",
    "中学-28": "图片 28.png",
    "中学-36": "图片 36.png",
    "中学-40": "图片 40.png",
    "宁夏-1":  "1.png",
    "宁夏-2":  "2.png",
    "宁夏-3":  "3.png",
    "其他-4":  "4.png",
    "其他-5":  "5.png",
    "其他-8":  "8.png",
    "其他-9":  "9.png",
    "其他-12": "12.png",
    "其他-13": "13.png",
}


def compute_layout_features(img_path: str):
    """计算布局特征：宽高比 + 左右不对称度"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. 宽高比
    aspect_ratio = w / h

    # 2. 左右灰度直方图差异（归一化直方图的卡方距离）
    left_half = gray[:, :w // 2]
    right_half = gray[:, w // 2:]
    hist_left = cv2.calcHist([left_half], [0], None, [64], [0, 256]).flatten()
    hist_right = cv2.calcHist([right_half], [0], None, [64], [0, 256]).flatten()
    hist_left = hist_left / (hist_left.sum() + 1e-8)
    hist_right = hist_right / (hist_right.sum() + 1e-8)
    chi2 = np.sum((hist_left - hist_right) ** 2 / (hist_left + hist_right + 1e-8)) / 2

    # 3. 左右边缘密度差异
    edges = cv2.Canny(gray, 50, 150)
    left_edge_density = edges[:, :w // 2].mean()
    right_edge_density = edges[:, w // 2:].mean()
    edge_asymmetry = abs(left_edge_density - right_edge_density) / (left_edge_density + right_edge_density + 1e-8)

    # 4. 左右亮度差
    left_brightness = left_half.mean()
    right_brightness = right_half.mean()
    brightness_diff = abs(left_brightness - right_brightness) / 255.0

    return {
        "aspect_ratio": aspect_ratio,
        "hist_chi2": chi2,
        "edge_asymmetry": edge_asymmetry,
        "brightness_diff": brightness_diff,
        "size": f"{w}x{h}",
    }


print(f"{'名称':>10s}  {'尺寸':>12s}  {'宽高比':>6s}  {'直方图χ²':>8s}  {'边缘不对称':>10s}  {'亮度差':>6s}")
print("-" * 72)

for label, fname in TEST_FILES.items():
    img_path = IMAGE_DIR / fname
    if not img_path.exists():
        print(f"{label:>10s}  不存在: {fname}")
        continue

    f = compute_layout_features(str(img_path))
    if f is None:
        continue

    print(f"{label:>10s}  {f['size']:>12s}  {f['aspect_ratio']:6.3f}  {f['hist_chi2']:8.4f}  {f['edge_asymmetry']:10.4f}  {f['brightness_diff']:6.4f}")
