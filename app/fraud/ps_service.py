"""
PS/篡改鉴别服务
调用 ZhengYan 多模态 API，分析证件图片是否存在 Photoshop 或人工篡改痕迹。
结果持久化至 MySQL ps_detection_results 表。
"""
import base64
import io
import json
import logging
import re
from typing import Dict, Any, Optional

from PIL import Image

from app.fraud.ocr_service import _LegacyTLSAdapter, _make_session

logger = logging.getLogger(__name__)

# ── PS 鉴别提示词（专家级）──────────────────────────────────────

_PS_PROMPT = """你是一位拥有15年数字图像取证经验的资深专家，专精学历证件真伪鉴别。请对这张证件图片进行严格的取证分析。

在 <thinking> 标签内，必须逐一检查以下8个维度，不得遗漏：

<thinking>
1. 整体一致性：背景纹理、色调统一性，是否存在局部分辨率突变、拼接边界或克隆区域
2. 文字与字体：关键字段（姓名/证号/日期/学校）字体是否一致，笔画粗细是否异常，文字边缘是否有白色光晕、锯齿或模糊，是否存在重叠/遮盖痕迹
3. 照片/人像区：人像照片光照方向是否与证件整体环境匹配，照片边缘是否有切割/融合/羽化痕迹，照片与背景分辨率/噪声是否一致
4. 印章/公章：印章颜色是否自然（过于鲜艳或过于暗淡均可疑），文字是否均匀清晰，是否有截图/打印再扫描特征，钢印压痕是否真实
5. 纸张与物理痕迹：折痕、水印、污渍的分布是否自然，物理痕迹是否与内容区域冲突（如折痕穿过文字但文字无形变）
6. 噪声与频域：图像噪声分布是否均匀，不同区域是否有明显的JPEG压缩块效应差异，是否有高频伪影
7. 语义逻辑：证件信息是否自洽（入学年份+学制 vs 毕业年份，日期格式，编号位数和格式是否符合规律）
8. 其他常见PS痕迹：内容克隆/图章复制残留、选区羽化边缘、修复画笔/仿制图章痕迹、色彩空间不一致
</thinking>

根据以上分析，严格以 JSON 格式返回最终鉴别结论（不要包含 thinking 内容，不要 markdown 代码块）：
{
  "overall_authenticity": "真实 / 疑似修改 / 高度疑似PS / 几乎确定伪造",
  "confidence": 85,
  "tampering_probability": 20,
  "key_evidence": ["支持篡改的关键证据1", "关键证据2"],
  "no_tampering_evidence": ["支持真实的证据1", "支持真实的证据2"],
  "suspect_regions": [
    {"description": "可疑区域描述", "reason": "具体可疑原因"}
  ],
  "recommendation": "处置建议，如：建议人工复核证件原件 / 证件真实性较高，可正常使用"
}

判断标准：
- 真实：所有维度均无异常，整体真实性高（tampering_probability ≤ 20）
- 疑似修改：1-2处轻微可疑，建议人工复核（tampering_probability 21-50）
- 高度疑似PS：存在明显篡改痕迹，高度怀疑伪造（tampering_probability 51-80）
- 几乎确定伪造：多处确凿篡改证据，几乎可以确定为伪造（tampering_probability > 80）

confidence 为你对此次判断的把握程度（0-100整数），tampering_probability 为篡改概率（0-100整数）。"""


class PSDetectionService:

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """调用 ZhengYan API 对图片进行 PS 鉴别，返回结构化结果"""
        from app.config import settings

        if not settings.ZHENGYAN_ENDPOINT_URL:
            raise RuntimeError("ZhengYan API 未配置，请设置环境变量 ZHENGYAN_ENDPOINT_URL")

        # PIL Image → JPEG base64
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        headers = {"Content-Type": "application/json"}
        if settings.ZHENGYAN_ACCESS_TOKEN:
            headers["Authorization"] = settings.ZHENGYAN_ACCESS_TOKEN
        if settings.ZHENGYAN_APP_ID:
            headers["App-Id"] = settings.ZHENGYAN_APP_ID
        if settings.ZHENGYAN_MODEL_TYPE:
            headers["Model-Type"] = settings.ZHENGYAN_MODEL_TYPE

        body = {
            "text": _PS_PROMPT,
            "image": b64,
            "user_info": {
                "user_id": settings.ZHENGYAN_USER_ID,
                "user_name": settings.ZHENGYAN_USER_NAME,
                "user_dept_name": settings.ZHENGYAN_USER_DEPT_NAME,
                "user_company": settings.ZHENGYAN_USER_COMPANY,
            },
        }

        try:
            session = _make_session()
            resp = session.post(
                settings.ZHENGYAN_ENDPOINT_URL,
                json=body,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error("ZhengYan PS detection API call failed: %s", e)
            raise RuntimeError(f"PS detection API failed: {e}") from e

        data = resp.json()
        raw_text = self._extract_text(data)
        return self._parse_result(raw_text)

    def _extract_text(self, data: dict) -> str:
        """从 API 响应中提取文本内容"""
        choices = data.get("choices") or (data.get("data") or {}).get("choices")
        if choices and isinstance(choices, list):
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content:
                return str(content).strip()
        inner = data.get("data") or {}
        if isinstance(inner, dict):
            for key in ("content", "text", "result", "output"):
                val = inner.get(key)
                if val:
                    return str(val).strip()
        for key in ("content", "text", "result", "output"):
            val = data.get(key)
            if val:
                return str(val).strip()
        return ""

    _AUTH_TO_RISK = {
        "真实": "低",
        "疑似修改": "中",
        "高度疑似PS": "高",
        "几乎确定伪造": "高",
    }

    def _parse_result(self, text: str) -> Dict[str, Any]:
        """解析模型返回的 JSON（含 thinking 块），附带原始响应用于调试"""
        result: Dict[str, Any] = {
            "overall_authenticity": "真实",
            "risk_level": "低",
            "tampering_probability": None,
            "confidence": None,
            "key_evidence": [],
            "no_tampering_evidence": [],
            "suspect_regions": [],
            "recommendation": "",
            "_raw_response": text,
        }
        try:
            # 剥离 <thinking>...</thinking> 块（含多行，非贪婪）
            clean = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text, flags=re.IGNORECASE).strip()
            # 去除 markdown 代码块
            json_str = re.sub(r'^```(?:json)?\s*|\s*```$', '', clean, flags=re.MULTILINE).strip()
            data = json.loads(json_str)
            if isinstance(data, dict):
                auth = str(data.get("overall_authenticity") or "真实").strip()
                risk = self._AUTH_TO_RISK.get(auth, "低")

                def _jlist(val):
                    if isinstance(val, list):
                        return [str(x) for x in val if x]
                    if isinstance(val, str) and val:
                        return [val]
                    return []

                result["overall_authenticity"] = auth
                result["risk_level"] = risk
                result["tampering_probability"] = (
                    float(data["tampering_probability"])
                    if data.get("tampering_probability") is not None else None
                )
                result["confidence"] = (
                    float(data["confidence"])
                    if data.get("confidence") is not None else None
                )
                result["key_evidence"] = _jlist(data.get("key_evidence"))
                result["no_tampering_evidence"] = _jlist(data.get("no_tampering_evidence"))
                raw_regions = data.get("suspect_regions") or []
                if isinstance(raw_regions, list):
                    result["suspect_regions"] = [
                        r if isinstance(r, dict) else {"description": str(r), "reason": ""}
                        for r in raw_regions
                    ]
                result["recommendation"] = str(data.get("recommendation") or "").strip()
        except Exception:
            logger.debug("PS result JSON parse failed, raw=%s", text[:100])
        return result

    def detect_and_save(self, entity_id: int, image: Image.Image) -> Dict[str, Any]:
        """检测并写入 MySQL，已有缓存则直接返回"""
        from app.database import get_ps_result, save_ps_result
        cached = get_ps_result(entity_id)
        if cached is not None:
            return cached
        try:
            result = self.detect(image)
            save_ps_result(entity_id, result)
            logger.info("PS detection saved for entity_id=%s, risk=%s", entity_id, result.get("risk_level"))
            return {k: v for k, v in result.items() if not k.startswith("_")}
        except Exception as e:
            logger.error("PS detection failed for entity_id=%s: %s", entity_id, e)
            return {"risk_level": "未知", "reasons": [str(e)], "confidence": None, "summary": "检测失败"}

    def redetect_and_save(self, entity_id: int, image: Image.Image) -> Dict[str, Any]:
        """强制重新检测（忽略缓存）"""
        from app.database import save_ps_result, get_ps_result
        # 先删除旧记录
        try:
            from app.database import is_db_available, db_session
            from app.models import PSDetectionResult
            if is_db_available():
                with db_session() as db:
                    db.query(PSDetectionResult).filter(PSDetectionResult.entity_id == entity_id).delete()
        except Exception:
            pass
        try:
            result = self.detect(image)
            save_ps_result(entity_id, result)
            return {k: v for k, v in result.items() if not k.startswith("_")}
        except Exception as e:
            logger.error("PS re-detection failed for entity_id=%s: %s", entity_id, e)
            return {"risk_level": "未知", "reasons": [str(e)], "confidence": None, "summary": "检测失败"}


ps_detection_service = PSDetectionService()
