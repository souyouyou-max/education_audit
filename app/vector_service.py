"""
向量提取服务 - 改进版（2026年推荐实践）
"""
import logging
import numpy as np
from typing import List, Tuple, Optional
import torch
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
import insightface
from app.config import settings

logger = logging.getLogger(__name__)

class FaceQualityError(Exception):
    """自定义异常：人脸质量不足或未检测到有效人脸"""
    pass

class VectorService:
    """向量提取服务 - 优化版"""
    
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_processor = None
        self.face_app = None
        self._models_loaded = False
    
    def _ensure_models_loaded(self):
        if self._models_loaded:
            return
        
        self._init_models()
        self._models_loaded = True
    
    def _init_models(self):
        try:
            # CLIP（建议使用 openai/clip-vit-large-patch14 或更新版本）
            logger.info("Loading CLIP model: %s", settings.CLIP_MODEL_NAME)
            self.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME)
            self.clip_processor = CLIPProcessor.from_pretrained(
                settings.CLIP_MODEL_NAME, use_fast=True
            )
            self.clip_model.eval()
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.error("Failed to load CLIP: %s", e, exc_info=True)
            raise
        
        try:
            # DINOv2（纯视觉自监督模型，用于模板特征提取）
            logger.info("Loading DINOv2 model: %s", settings.DINO_MODEL_NAME)
            self.dino_processor = AutoImageProcessor.from_pretrained(settings.DINO_MODEL_NAME)
            self.dino_model = AutoModel.from_pretrained(settings.DINO_MODEL_NAME)
            self.dino_model.eval()
            if torch.cuda.is_available():
                self.dino_model = self.dino_model.cuda()
            logger.info("DINOv2 model loaded")
        except Exception as e:
            logger.error("Failed to load DINOv2: %s", e, exc_info=True)
            raise

        try:
            model_name = settings.INSIGHTFACE_MODEL_NAME
            logger.info("Loading InsightFace model: %s", model_name)
            self.face_app = insightface.app.FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            # det_size 建议根据场景调整：越大越准但越慢
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace loaded")
        except Exception as e:
            logger.error("Failed to load InsightFace: %s", e, exc_info=True)
            raise
    
    def extract_image_vector(self, image: Image.Image) -> List[float]:
        """提取整体图像向量（CLIP）"""
        self._ensure_models_loaded()
        try:
            # 强制加载像素数据：ImageFile 的延迟加载可能在 fp=None 时触发 AssertionError
            if hasattr(image, 'tile') and image.tile:
                image.load()
            image = image.convert("RGB")
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v 
                              for k, v in inputs.items()}
                
                features = self.clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                
                vector = features[0].cpu().numpy().tolist()
                
                expected_dim = settings.IMAGE_VECTOR_DIM
                if len(vector) != expected_dim:
                    raise ValueError(f"CLIP vector dim mismatch: got {len(vector)}, expected {expected_dim}")

                return vector
        except Exception as e:
            logger.error("Image vector extraction failed: %s", e, exc_info=True)
            raise
    
    def extract_face_vector(self, image: Image.Image, min_quality_score: float = 0.5,
                            _faces=None) -> Tuple[List[float], bool]:
        """提取人脸向量（InsightFace），增加质量过滤

        Args:
            _faces: 预先检测好的人脸列表（由 extract_all_vectors 传入以避免重复检测）
        Returns:
            (vector, success): 成功时返回向量+True，失败返回None+False
        """
        self._ensure_models_loaded()
        try:
            img_array = np.array(image.convert("RGB"))
            faces = _faces if _faces is not None else self.face_app.get(img_array)
            
            if not faces:
                logger.debug("No face detected")
                return None, False
            
            # 取置信度最高的人脸
            best_face = max(faces, key=lambda f: f.det_score)
            
            if best_face.det_score < min_quality_score:
                logger.debug("Face quality too low: score=%.3f", best_face.det_score)
                return None, False

            # 检查人脸大小（像素太小通常不可靠）
            bbox = best_face.bbox.astype(int)
            face_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if face_size < 80:
                logger.debug("Face too small: %dpx", face_size)
                return None, False
            
            embedding = best_face.embedding
            norm = np.linalg.norm(embedding)
            if norm < 1e-6:  # 防止除0
                raise ValueError("Zero norm embedding")
            embedding = embedding / norm
            
            vector = embedding.tolist()
            
            expected_dim = settings.FACE_VECTOR_DIM
            if len(vector) != expected_dim:
                raise ValueError(f"Face vector dim mismatch: got {len(vector)}, expected {expected_dim}")
            
            return vector, True
        
        except Exception as e:
            logger.error("Face vector extraction failed: %s", e, exc_info=True)
            return None, False
    
    def _extract_border_hsv_histogram(self, img_array: np.ndarray,
                                      h_bins: int = 32, s_bins: int = 16,
                                      v_bins: int = 16) -> np.ndarray:
        """提取边框区域的 HSV 颜色直方图特征（64 维）"""
        h, w = img_array.shape[:2]
        br = settings.TEMPLATE_BORDER_RATIO
        bh, bw = int(h * br), int(w * br)
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:bh, :] = True
        border_mask[h - bh:, :] = True
        border_mask[:, :bw] = True
        border_mask[:, w - bw:] = True

        hsv_img = np.array(Image.fromarray(img_array).convert("HSV"))
        pixels = hsv_img[border_mask].copy()

        # 归一化 V（亮度）通道，消除扫描/光照差异（不同扫描件亮度可能差异很大）
        v_mean = pixels[:, 2].mean()
        if v_mean > 1:
            pixels[:, 2] = np.clip(
                pixels[:, 2].astype(float) / v_mean * 128, 0, 255
            ).astype(np.uint8)

        h_hist, _ = np.histogram(pixels[:, 0], bins=h_bins,
                                 range=(0, 255), density=True)
        s_hist, _ = np.histogram(pixels[:, 1], bins=s_bins,
                                 range=(0, 255), density=True)
        v_hist, _ = np.histogram(pixels[:, 2], bins=v_bins,
                                 range=(0, 255), density=True)

        feat = np.concatenate([h_hist, s_hist, v_hist])
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat = feat / norm
        return feat

    def extract_template_vector(self, image: Image.Image,
                                _faces=None) -> Optional[List[float]]:
        """提取模板特征向量 —— DINOv2结构特征 + HSV颜色特征 混合

        策略：
        1. 遮盖检测到的人脸（灰色填充）
        2. DINOv2-large 提取纯视觉结构特征（1024维 CLS token）
           - 纯视觉自监督模型，不受"都是毕业证"文字语义干扰
           - 对边框样式、排版布局、颜色分布敏感
        3. HSV 提取边框区域颜色直方图（64维）
        4. HSV×1.5 加权拼接后 L2 归一化，输出 1088 维模板向量（1024 + 64）

        Args:
            _faces: 预先检测好的人脸列表（由 extract_all_vectors 传入以避免重复检测）
        """
        self._ensure_models_loaded()
        try:
            image = image.convert("RGB")
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            gray = np.array([128, 128, 128], dtype=np.uint8)

            # 只遮盖人脸（DINOv2无文字语义偏见，不需要遮内容区域）
            processed = img_array.copy()
            try:
                faces = _faces if _faces is not None else self.face_app.get(img_array)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    margin = int(max(fw, fh) * 0.6)
                    y1 = max(0, bbox[1] - margin)
                    y2 = min(h, bbox[3] + margin)
                    x1 = max(0, bbox[0] - margin)
                    x2 = min(w, bbox[2] + margin)
                    processed[y1:y2, x1:x2] = gray
            except Exception as e:
                logger.warning("Face masking failed: %s", e)

            processed_pil = Image.fromarray(processed)
            processed_pil = processed_pil.filter(
                ImageFilter.GaussianBlur(radius=settings.TEMPLATE_BLUR_RADIUS))

            if max(processed_pil.size) > 1024:
                processed_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            # --- 1. DINOv2 特征（CLS token，768维） ---
            with torch.no_grad():
                inputs = self.dino_processor(images=processed_pil, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {
                        k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                outputs = self.dino_model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                dino_vec = features[0].cpu().numpy()

            # --- 2. HSV 颜色直方图特征（边框区域，64维） ---
            hsv_vec = self._extract_border_hsv_histogram(img_array)

            # --- 3. 拼接并归一化（HSV 加权 1.5） ---
            combined = np.concatenate([dino_vec, hsv_vec * 1.5])
            norm = np.linalg.norm(combined)
            if norm > 1e-8:
                combined = combined / norm

            vector = combined.tolist()

            expected_dim = settings.TEMPLATE_VECTOR_DIM  # 1088 = DINOv2-large(1024) + HSV(64)
            if len(vector) != expected_dim:
                raise ValueError(
                    f"Template vector dim mismatch: got {len(vector)}, "
                    f"expected {expected_dim}")

            return vector
        except Exception as e:
            logger.error("Template vector extraction failed: %s", e, exc_info=True)
            return None
    
    def get_face_attributes(self, image: Image.Image) -> Optional[dict]:
        """
        获取人脸属性（年龄、性别、bbox）供规则引擎使用
        返回 {"age": int, "gender": str, "bbox": list, "det_score": float}
        未检测到人脸时返回 None
        """
        self._ensure_models_loaded()
        try:
            img_array = np.array(image.convert("RGB"))
            faces = self.face_app.get(img_array)
            if not faces:
                return None
            best = max(faces, key=lambda f: f.det_score)
            gender_raw = getattr(best, "sex", None)
            if isinstance(gender_raw, (int, float)):
                gender = "男" if gender_raw >= 0.5 else "女"
            elif isinstance(gender_raw, str):
                gender = "男" if gender_raw.upper() in ("M", "MALE") else "女"
            else:
                gender = None
            return {
                "age": int(best.age) if hasattr(best, "age") and best.age is not None else None,
                "gender": gender,
                "bbox": best.bbox.tolist(),
                "det_score": float(best.det_score),
            }
        except Exception as e:
            logger.warning("get_face_attributes failed: %s", e)
            return None

    def extract_all_vectors(self, image: Image.Image) -> Tuple[List[float], List[float], List[float]]:
        """提取所有向量，失败字段返回零向量（Milvus 不接受 None）

        并发优化：
        - CLIP 和 InsightFace 用两个线程同时跑（两者独立，都会释放 GIL）
        - 人脸检测结果共享给 face_vector 和 template_vector，只跑一次
        - DINOv2 在两者完成后串行执行（依赖人脸检测结果）
        """
        import threading
        self._ensure_models_loaded()

        # 预先将 PIL Image 转换为 numpy 数组（线程安全），避免多线程同时操作同一 PIL 对象
        # PIL Image 非线程安全：两个线程同时调用 .convert("RGB") 会导致 JPEG fp 状态竞争
        img_rgb   = image.convert("RGB")
        img_array = np.array(img_rgb)

        image_vec_box: List = [None]
        faces_box:     List = [[]]

        def _run_clip():
            try:
                # 每个线程独立创建 PIL Image，避免共享 fp 状态
                clip_img = Image.fromarray(img_array)
                image_vec_box[0] = self.extract_image_vector(clip_img)
            except Exception as e:
                logger.error("extract_image_vector failed: %s", e)

        def _run_face():
            try:
                faces_box[0] = self.face_app.get(img_array)
            except Exception as e:
                logger.warning("Face detection failed: %s", e)

        # CLIP 和 InsightFace 并行
        t_clip = threading.Thread(target=_run_clip, daemon=True)
        t_face = threading.Thread(target=_run_face, daemon=True)
        t_clip.start()
        t_face.start()
        t_clip.join()
        t_face.join()

        image_vec      = image_vec_box[0]
        detected_faces = faces_box[0]

        # 串行调用使用独立 PIL Image（fromarray，无 fp 依赖）
        face_img = Image.fromarray(img_array)

        # 人脸向量：从已检测结果直接提取，无需重跑
        face_vec, _ = self.extract_face_vector(face_img, _faces=detected_faces)

        # DINOv2 模板向量：依赖人脸遮盖，在人脸检测完成后串行
        template_vec = None
        try:
            template_vec = self.extract_template_vector(face_img, _faces=detected_faces)
        except Exception as e:
            logger.error("extract_template_vector failed: %s", e)

        # 保证返回合法向量（None → 零向量）
        if image_vec is None:
            image_vec = [0.0] * settings.IMAGE_VECTOR_DIM
        if face_vec is None:
            face_vec = [0.0] * settings.FACE_VECTOR_DIM
        if template_vec is None:
            template_vec = [0.0] * settings.TEMPLATE_VECTOR_DIM

        return image_vec, face_vec, template_vec


    def extract_all_vectors_batch(
        self, images: list
    ) -> list:
        """批量提取多张图片的所有向量（减少模型加载开销）

        Args:
            images: PIL.Image 列表

        Returns:
            list of (image_vec, face_vec, template_vec) tuples
        """
        self._ensure_models_loaded()
        return [self.extract_all_vectors(img) for img in images]


# 全局实例
vector_service = VectorService()