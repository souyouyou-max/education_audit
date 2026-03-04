"""
图像取证服务 - 检测PS伪造痕迹（纯本地，无需外部数据）

检测项目：
- ELA（误差级别分析）：识别被PS修改的区域
- 照片边缘锐利度：贴片照片边缘过于整齐
- 印章真实性：打印印章颜色过于均匀
- 感知哈希：跨证件区域相似度对比
"""
import io
import logging
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ForensicService:

    def ela_analysis(self, image: Image.Image, quality: int = 90) -> Dict[str, Any]:
        """
        误差级别分析（Error Level Analysis）
        原理：JPEG重压缩后，未修改区域误差小，PS贴入的区域误差大（更亮）
        """
        img_rgb = image.convert("RGB")
        buf = io.BytesIO()
        img_rgb.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")

        orig = np.array(img_rgb, dtype=np.float32)
        recomp_arr = np.array(recomp, dtype=np.float32)
        ela = np.abs(orig - recomp_arr) * 10
        ela = np.clip(ela, 0, 255).astype(np.uint8)

        ela_std = float(np.std(ela))
        ela_mean = float(np.mean(ela))
        bright_ratio = float(np.mean(ela > 50))

        # 高方差 或 大面积高亮 → 可疑
        is_suspicious = ela_std > 25 or bright_ratio > 0.15

        return {
            "ela_std": round(ela_std, 2),
            "ela_mean": round(ela_mean, 2),
            "bright_area_ratio": round(bright_ratio, 4),
            "is_suspicious": is_suspicious,
            "description": "发现异常高亮区域，可能存在PS修改" if is_suspicious
                           else "图像整体一致，未发现明显PS痕迹",
        }

    def analyze_photo_edge(
        self,
        image: Image.Image,
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        分析证件照区域边缘锐利度
        贴片照片特征：边缘梯度极大，与背景噪声特征差异明显
        """
        img_arr = np.array(image.convert("RGB"))
        h, w = img_arr.shape[:2]
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

        if face_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in face_bbox)
            pad = 30
            rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
            rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
        else:
            # 默认假设证件照在右上角区域
            rx1, ry1 = int(w * 0.6), 0
            rx2, ry2 = w, int(h * 0.45)

        photo = gray[ry1:ry2, rx1:rx2]
        if photo.size == 0:
            return {"error": "无法定位照片区域"}

        # Sobel梯度（边缘锐利度）
        sx = cv2.Sobel(photo, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(photo, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sx ** 2 + sy ** 2)
        max_grad = float(np.max(gradient))
        mean_grad = float(np.mean(gradient))

        # 照片区域与背景噪声比值
        bg = gray[:, : int(w * 0.55)]
        photo_noise = float(np.std(photo))
        bg_noise = float(np.std(bg)) if bg.size > 0 else photo_noise
        noise_ratio = photo_noise / (bg_noise + 1e-6)

        is_suspicious = max_grad > 200 and noise_ratio > 1.5

        return {
            "max_gradient": round(max_grad, 2),
            "mean_gradient": round(mean_grad, 2),
            "noise_ratio": round(noise_ratio, 4),
            "is_suspicious": is_suspicious,
            "description": "照片边缘异常锐利，可能为贴片照片" if is_suspicious
                           else "照片边缘过渡自然",
        }

    def analyze_seal(self, image: Image.Image) -> Dict[str, Any]:
        """
        印章真实性分析
        打印印章：红色像素颜色方差极小（机器喷墨，均匀一致）
        真实盖章：油墨渗透，颜色方差较大，边缘有晕染
        """
        img_arr = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)

        # 两段红色HSV区间（红色在HSV中横跨0°和180°两端）
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_count = int(np.sum(red_mask > 0))
        total = img_arr.shape[0] * img_arr.shape[1]

        if red_count < 200:
            return {"has_red_seal": False, "description": "未检测到红色印章区域"}

        red_pixels = img_arr[red_mask > 0].astype(float)
        color_variance = float(np.var(red_pixels))

        # 打印：方差 < 600（颜色极其均匀）
        is_printed = color_variance < 600

        return {
            "has_red_seal": True,
            "red_pixel_ratio": round(red_count / total, 4),
            "color_variance": round(color_variance, 2),
            "is_printed_seal": is_printed,
            "description": "颜色均匀，疑似打印印章（非实体盖章）" if is_printed
                           else "颜色自然，印章特征正常",
        }

    def compute_phash(self, image: Image.Image, region: str = "full") -> str:
        """
        计算图像区域感知哈希（16×16 = 256位）
        region: 'full' | 'photo'（右上角） | 'seal'（下部） | 'bottom'（最底部）
        """
        img_arr = np.array(image.convert("RGB"))
        h, w = img_arr.shape[:2]

        if region == "photo":
            crop = img_arr[0: int(h * 0.5), int(w * 0.55):]
        elif region == "seal":
            crop = img_arr[int(h * 0.55):, :]
        elif region == "bottom":
            crop = img_arr[int(h * 0.8):, :]
        else:
            crop = img_arr

        if crop.size == 0:
            crop = img_arr

        resized = Image.fromarray(crop).resize((16, 16)).convert("L")
        arr = np.array(resized)
        mean_val = arr.mean()
        bits = (arr > mean_val).flatten()
        hash_int = int("".join("1" if b else "0" for b in bits), 2)
        return format(hash_int, "x").zfill(64)

    @staticmethod
    def phash_distance(h1: str, h2: str) -> int:
        """计算两个感知哈希的汉明距离（0=完全相同，256=完全不同）"""
        try:
            i1 = int(h1, 16)
            i2 = int(h2, 16)
            return bin(i1 ^ i2).count("1")
        except Exception:
            return 256

    def full_analysis(
        self,
        image: Image.Image,
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """对单张证件图像进行完整取证分析，汇总风险等级"""
        results: Dict[str, Any] = {}

        try:
            results["ela"] = self.ela_analysis(image)
        except Exception as e:
            results["ela"] = {"error": str(e)}

        try:
            results["photo_edge"] = self.analyze_photo_edge(image, face_bbox)
        except Exception as e:
            results["photo_edge"] = {"error": str(e)}

        try:
            results["seal"] = self.analyze_seal(image)
        except Exception as e:
            results["seal"] = {"error": str(e)}

        risk_flags = []
        if results.get("ela", {}).get("is_suspicious"):
            risk_flags.append("ELA检测到可疑修改区域")
        if results.get("photo_edge", {}).get("is_suspicious"):
            risk_flags.append("照片边缘异常锐利，疑似贴片")
        if results.get("seal", {}).get("is_printed_seal"):
            risk_flags.append("印章疑似打印，非实体盖章")

        results["risk_flags"] = risk_flags
        results["risk_level"] = (
            "高" if len(risk_flags) >= 2 else ("中" if len(risk_flags) == 1 else "低")
        )
        return results

    def compare_regions_batch(
        self, id_image_map: Dict[int, Image.Image]
    ) -> list:
        """
        批量比对所有证件的关键区域感知哈希
        返回相似度异常高的证件对（可能是同一模板伪造）
        """
        hashes: Dict[int, Dict[str, str]] = {}
        for eid, img in id_image_map.items():
            try:
                hashes[eid] = {
                    "photo": self.compute_phash(img, "photo"),
                    "seal": self.compute_phash(img, "seal"),
                    "bottom": self.compute_phash(img, "bottom"),
                }
            except Exception as e:
                logger.warning("phash failed for id=%s: %s", eid, e)

        issues = []
        ids = list(hashes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                for region in ("photo", "seal", "bottom"):
                    dist = self.phash_distance(
                        hashes[id1].get(region, ""),
                        hashes[id2].get(region, ""),
                    )
                    # 汉明距离 < 10 视为高度相似（256位哈希）
                    if dist < 10:
                        issues.append({
                            "rule": f"不同证件{region}区域高度雷同",
                            "detail": (
                                f"ID {id1} 与 ID {id2} 的【"
                                + {"photo": "照片", "seal": "印章", "bottom": "底部"}[region]
                                + f"】区域感知哈希汉明距离={dist}（几乎相同）"
                            ),
                            "severity": "高",
                            "related_ids": [str(id1), str(id2)],
                        })
        return issues


forensic_service = ForensicService()
