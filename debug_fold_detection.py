#!/usr/bin/env python3
"""诊断：测试对折线检测算法，看能否区分 高中(单页) vs 中学(对折)"""
import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path("/Users/songyangyang/Desktop/education_audit/pic")

# 高中单页: 26, 33  |  中学对折: 29, 39, 18, 28, 36, 40
TEST_FILES = {
    "高中-26": "图片 26.png",
    "高中-33": "图片 33.png",
    "中学-29": "图片 29.png",
    "中学-39": "图片 39.png",
    "中学-18": "图片 18.png",
    "中学-28": "图片 28.png",
    "中学-36": "图片 36.png",
    "中学-40": "图片 40.png",
    # 再测几张其他模板看看是否有误判
    "宁夏-1":  "1.png",
    "宁夏-2":  "2.png",
    "其他-4":  "4.png",
    "其他-5":  "5.png",
}


def compute_fold_score(img_path: str) -> float:
    """检测图片中心是否有垂直对折线。

    方法：用 Sobel 检测水平梯度（垂直边缘），
    比较中心竖条区域的边缘强度 vs 图片整体。
    比值 > 1 说明中心边缘密集（折痕信号）。
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    h, w = img.shape

    # 水平梯度 → 捕捉垂直边缘（折痕是垂直线）
    sobel_x = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))

    # 中心竖条：±2.5% 宽度
    center = w // 2
    strip_half = max(int(w * 0.025), 3)
    center_strip = sobel_x[:, center - strip_half:center + strip_half]
    center_intensity = center_strip.mean()

    # 两侧（排除中心条）
    left = sobel_x[:, :center - strip_half]
    right = sobel_x[:, center + strip_half:]
    rest_intensity = np.concatenate([left, right], axis=1).mean()

    if rest_intensity < 1e-6:
        return 0.0

    return center_intensity / rest_intensity


def compute_brightness_dip(img_path: str) -> float:
    """检测中心亮度凹陷（折痕处通常有阴影）。

    返回凹陷比例：0 表示无凹陷，正值越大凹陷越明显。
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    h, w = img.shape

    # 每列平均亮度
    profile = img.mean(axis=0)

    # 中心区域
    center = w // 2
    strip = max(int(w * 0.03), 3)
    center_brightness = profile[center - strip:center + strip].mean()

    # 左右两侧平均亮度（排除边缘 10%）
    margin = int(w * 0.1)
    left_brightness = profile[margin:center - strip].mean()
    right_brightness = profile[center + strip:w - margin].mean()
    side_avg = (left_brightness + right_brightness) / 2

    if side_avg < 1e-6:
        return 0.0

    return (side_avg - center_brightness) / side_avg


# 测试所有图片
print(f"{'名称':>10s}  {'边缘比':>8s}  {'亮度凹陷':>8s}  {'判定':>6s}")
print("-" * 45)

for label, fname in TEST_FILES.items():
    img_path = IMAGE_DIR / fname
    if not img_path.exists():
        print(f"{label:>10s}  文件不存在: {fname}")
        continue

    edge_ratio = compute_fold_score(str(img_path))
    brightness_dip = compute_brightness_dip(str(img_path))

    # 简单判定：边缘比 > 1.3 或 亮度凹陷 > 0.03 → 对折
    is_fold = edge_ratio > 1.3 or brightness_dip > 0.03
    verdict = "对折" if is_fold else "单页"

    print(f"{label:>10s}  {edge_ratio:8.3f}  {brightness_dip:8.4f}  {verdict:>6s}")
