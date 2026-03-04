"""
OCR服务 - 提取学历证件结构化字段
使用 ZhengYan 多模态 API 识别文字，正则解析关键字段。
持久化：仅使用 MySQL ocr_results 表，无本地 JSON 缓存。
"""
import base64
import io
import json
import logging
import re
from typing import Optional, Dict, Any

import ssl

import requests
import urllib3
from requests.adapters import HTTPAdapter
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class _LegacyTLSAdapter(HTTPAdapter):
    """兼容内网服务端特殊 TLS 握手的 Adapter（SECLEVEL=1 + 禁用证书校验）"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)

    def send(self, request, **kwargs):
        # urllib3 2.x 在 send 层独立做证书校验，必须在此强制关闭
        kwargs["verify"] = False
        return super().send(request, **kwargs)


def _make_session() -> requests.Session:
    session = requests.Session()
    session.mount("https://", _LegacyTLSAdapter())
    return session

logger = logging.getLogger(__name__)

# 中文数字映射
_CN_MAP = {
    '○': '0', '〇': '0', '零': '0',
    '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
    '六': '6', '七': '7', '八': '8', '九': '9',
}

# OCR + PS 鉴别合并提示词：一次 API 调用同时完成信息提取与真伪鉴别
_OCR_PROMPT = """你是专业的证件信息提取与鉴伪专家。请对图片中的学历证件完成两项任务，以 JSON 格式一次性返回所有结果（找不到的字段填 null）：

【任务一：信息提取】
  "cert_type": "证书类型（毕业证书/学位证书/结业证书/肄业证书等）",
  "student_name": "学生姓名",
  "school": "学校全称（含大学/学院/中学等后缀）",
  "department": "院系、学部或专业名称（如'普通高中部'、'计算机科学与技术系'）",
  "principal": "校长或院长姓名",
  "grad_year": "毕业年份（4位阿拉伯数字）",
  "enrollment_year": "入学年份（4位阿拉伯数字）",
  "issue_year": "发证年份（4位阿拉伯数字）",
  "issue_date": "完整发证日期（YYYY-MM-DD格式，如1994-07-20）",
  "cert_no": "证书编号（不含学籍号）",
  "student_id_no": "学籍号码",
  "gender": "性别（男或女）",
  "birth_year": "出生年份（4位阿拉伯数字）",
  "supervisor": "监制单位名称（不含'监制'二字，如'宁夏教育委员会'）",
  "raw_text": "证件上的所有原始文字，逐行保留"

【任务二：PS/篡改鉴别】逐一检查以下维度：
① 整体一致性与分辨率差异  ② 文字/字体异常（光晕/锯齿/粗细不一）
③ 照片区切割/融合痕迹     ④ 印章颜色/文字异常
⑤ 纸张折痕与物理痕迹      ⑥ 噪声与压缩块差异
  "ps_overall_authenticity": "真实 / 疑似修改 / 高度疑似PS / 几乎确定伪造",
  "ps_tampering_probability": 20,
  "ps_confidence": 85,
  "ps_key_evidence": ["支持篡改的关键证据（若无则为空数组）"],
  "ps_no_tampering_evidence": ["支持真实的证据"],
  "ps_suspect_regions": [{"description": "可疑区域描述", "reason": "原因"}],
  "ps_recommendation": "一句话处置建议"

注意：年份转为4位阿拉伯数字；日期转为YYYY-MM-DD；tampering_probability 和 confidence 为0-100整数；只返回 JSON 对象，不要任何额外解释或 markdown 代码块。"""


# ── MySQL helpers ─────────────────────────────────────────────────

def _row_to_dict(r) -> Dict[str, Any]:
    """OcrResult ORM 行 → dict（统一字段映射）"""
    return {
        "cert_type":        r.cert_type,
        "student_name":     r.student_name,
        "school":           r.school,
        "department":       r.department,
        "principal":        r.principal,
        "grad_year":        r.grad_year,
        "enrollment_year":  r.enrollment_year,
        "issue_year":       r.issue_year,
        "issue_date":       r.issue_date,
        "cert_no":          r.cert_no,
        "student_id_no":    r.student_id_no,
        "gender":           r.gender,
        "birth_year":       r.birth_year,
        "supervisor":       r.supervisor,
        "_raw_text":        r.raw_text or "",
    }


def _db_load_all() -> Dict[str, Any]:
    """从 MySQL ocr_results 读取全部缓存"""
    try:
        from app.database import is_db_available, db_session
        from app.models import OcrResult
        if not is_db_available():
            return {}
        with db_session() as db:
            rows = db.query(OcrResult).all()
            return {str(r.entity_id): _row_to_dict(r) for r in rows}
    except Exception as e:
        logger.warning("MySQL _db_load_all failed: %s", e)
        return {}


def _db_load_one(entity_id: int) -> Optional[Dict[str, Any]]:
    """从 MySQL 读取单条 OCR 结果，不存在返回 None"""
    try:
        from app.database import is_db_available, db_session
        from app.models import OcrResult
        if not is_db_available():
            return None
        with db_session() as db:
            r = db.query(OcrResult).filter(OcrResult.entity_id == entity_id).first()
            if r is None:
                return None
            return _row_to_dict(r)
    except Exception as e:
        logger.warning("MySQL _db_load_one failed for id=%s: %s", entity_id, e)
        return None


def _db_save_one(entity_id: int, fields: Dict[str, Any]) -> None:
    """将单条 OCR 结果写入 MySQL（幂等：先删后插）"""
    try:
        from app.database import is_db_available, db_session
        from app.models import OcrResult, Certificate
        if not is_db_available():
            return

        # ── 事务1：确保 certificates 父行存在并已提交 ──────────────────
        # 必须在独立事务中提交父行，后续事务的 FK 检查才能看到它
        try:
            with db_session() as db:
                if not db.query(Certificate).filter(Certificate.id == entity_id).first():
                    db.add(Certificate(id=entity_id))
                    logger.info("Auto-created certificates row for entity_id=%s", entity_id)
        except Exception:
            pass  # 并发写入主键冲突时忽略

        # ── 事务2：写入 OCR 结果（父行已提交，FK 约束可满足）──────────
        with db_session() as db:
            db.query(OcrResult).filter(OcrResult.entity_id == entity_id).delete()
            db.add(OcrResult(
                entity_id=entity_id,
                cert_type=fields.get("cert_type"),
                student_name=fields.get("student_name"),
                school=fields.get("school"),
                department=fields.get("department"),
                principal=fields.get("principal"),
                grad_year=fields.get("grad_year"),
                enrollment_year=fields.get("enrollment_year"),
                issue_year=fields.get("issue_year"),
                issue_date=fields.get("issue_date"),
                cert_no=fields.get("cert_no"),
                student_id_no=fields.get("student_id_no"),
                gender=fields.get("gender"),
                birth_year=fields.get("birth_year"),
                supervisor=fields.get("supervisor"),
                raw_text=(fields.get("_raw_text") or "")[:10000],
            ))
    except Exception as e:
        logger.warning("MySQL _db_save_one failed for id=%s: %s", entity_id, e)


def _db_delete_one(entity_id: int) -> None:
    """从 MySQL 删除单条 OCR 记录"""
    try:
        from app.database import is_db_available, db_session
        from app.models import OcrResult
        if not is_db_available():
            return
        with db_session() as db:
            db.query(OcrResult).filter(OcrResult.entity_id == entity_id).delete()
    except Exception as e:
        logger.warning("MySQL _db_delete_one failed for id=%s: %s", entity_id, e)


def _save_ps_from_fields(entity_id: int, fields: Dict[str, Any]) -> None:
    """若 fields 中包含 PS 鉴别结果（_ps_* 前缀），自动写入 ps_detection_results 表"""
    if not fields.get("_ps_risk_level"):
        return
    try:
        from app.database import save_ps_result
        save_ps_result(entity_id, {
            "overall_authenticity":  fields.get("_ps_overall_authenticity") or "真实",
            "risk_level":            fields.get("_ps_risk_level") or "低",
            "tampering_probability": fields.get("_ps_tampering_probability"),
            "confidence":            fields.get("_ps_confidence"),
            "key_evidence":          fields.get("_ps_key_evidence") or [],
            "no_tampering_evidence": fields.get("_ps_no_tampering_evidence") or [],
            "suspect_regions":       fields.get("_ps_suspect_regions") or [],
            "recommendation":        fields.get("_ps_recommendation") or "",
            "_raw_response":         "",
        })
    except Exception as e:
        logger.warning("_save_ps_from_fields failed for id=%s: %s", entity_id, e)


# ── 中文数字转换 ──────────────────────────────────────────────────

def _cn_to_year(s: str) -> Optional[str]:
    """中文4位数字→阿拉伯数字年份，如 '二○一八' → '2018'"""
    digits = ""
    for ch in s:
        if ch in _CN_MAP:
            digits += _CN_MAP[ch]
        elif ch.isdigit():
            digits += ch
    return digits if len(digits) == 4 else None


def _cn_to_num(s: str) -> Optional[int]:
    """中文数字(1-31)→整数，支持 廿/卅 前缀，用于月份和日期解析"""
    s = s.strip()
    if not s:
        return None
    # 廿x（20-29）
    if s[0] == '廿':
        rest = s[1:]
        d = int(_CN_MAP[rest[0]]) if rest and rest[0] in _CN_MAP else 0
        return 20 + d
    # 卅x（30-31）
    if s[0] == '卅':
        rest = s[1:]
        d = int(_CN_MAP[rest[0]]) if rest and rest[0] in _CN_MAP else 0
        return 30 + d
    # 十x（10-19）
    if s[0] == '十':
        rest = s[1:]
        d = int(_CN_MAP[rest[0]]) if rest and rest[0] in _CN_MAP else 0
        return 10 + d
    # 二十x（20-29）
    if len(s) >= 2 and s[1] == '十':
        tens = int(_CN_MAP.get(s[0], '0')) * 10
        rest = s[2:]
        d = int(_CN_MAP[rest[0]]) if rest and rest[0] in _CN_MAP else 0
        return tens + d
    # 三十x（30-31）
    if len(s) >= 2 and s[0] in _CN_MAP and s[1] == '十':
        tens = int(_CN_MAP[s[0]]) * 10
        rest = s[2:]
        d = int(_CN_MAP[rest[0]]) if rest and rest[0] in _CN_MAP else 0
        return tens + d
    # 单个中文数字
    if s[0] in _CN_MAP:
        return int(_CN_MAP[s[0]])
    if s.isdigit():
        return int(s)
    return None


def _parse_cn_date(flat: str) -> Optional[str]:
    """解析中文完整日期 → YYYY-MM-DD，如 '一九九四年七月廿日' → '1994-07-20'"""
    _Y = r'[一二三四五六七八九○〇零]{4}'
    _MD = r'[一二三四五六七八九十廿卅]{1,4}'
    m = re.search(rf'({_Y})年({_MD})月({_MD})[日号]', flat)
    if not m:
        return None
    year = _cn_to_year(m.group(1))
    month = _cn_to_num(m.group(2))
    day = _cn_to_num(m.group(3))
    if year and month and day:
        return f"{year}-{month:02d}-{day:02d}"
    return None


def _extract_text_from_response(data: dict) -> str:
    """从 ZhengYan API 响应中提取文本内容，兼容多种响应格式"""
    # 格式1: choices[0].message.content（OpenAI 兼容格式）
    choices = data.get("choices") or (data.get("data") or {}).get("choices")
    if choices and isinstance(choices, list):
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content:
            return str(content).strip()

    # 格式2: data.content / data.text / data.result / data.output
    inner = data.get("data") or {}
    if isinstance(inner, dict):
        for key in ("content", "text", "result", "output"):
            val = inner.get(key)
            if val:
                return str(val).strip()

    # 格式3: 顶层 content / text / result / output
    for key in ("content", "text", "result", "output"):
        val = data.get(key)
        if val:
            return str(val).strip()

    logger.warning("ZhengYan response has no recognized text field: %s", list(data.keys()))
    return ""


# ── OCR 服务 ─────────────────────────────────────────────────────

class OCRService:

    def extract_text(self, image: Image.Image) -> str:
        """调用 ZhengYan 多模态 API 提取图片中的所有文字"""
        from app.config import settings

        if not settings.ZHENGYAN_ENDPOINT_URL:
            raise RuntimeError(
                "ZhengYan API 未配置，请设置环境变量 ZHENGYAN_ENDPOINT_URL"
            )

        # 将 PIL Image 转为 JPEG base64
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
            "text": _OCR_PROMPT,
            "image": b64,
            "user_info": {
                "user_id": settings.ZHENGYAN_USER_ID,
                "user_name": settings.ZHENGYAN_USER_NAME,
                "user_dept_name": settings.ZHENGYAN_USER_DEPT_NAME,
                "user_company": settings.ZHENGYAN_USER_COMPANY,
            },
        }

        logger.debug(
            "Sending OCR request to %s, imageBase64Length=%d",
            settings.ZHENGYAN_ENDPOINT_URL,
            len(b64),
        )

        try:
            session = _make_session()
            resp = session.post(
                settings.ZHENGYAN_ENDPOINT_URL,
                json=body,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("ZhengYan API call failed: %s", e)
            raise RuntimeError(f"Failed to call ZhengYan API: {e}") from e

        data = resp.json()
        logger.debug("ZhengYan API response keys: %s", list(data.keys()))
        return _extract_text_from_response(data)

    def parse_fields(self, text: str) -> Dict[str, Any]:
        """解析证件关键字段：优先解析模型返回的 JSON，失败时回退正则"""

        # ── 优先：JSON 结构化解析 ────────────────────────────────
        try:
            # 兼容模型把结果包在 ```json ... ``` 代码块里的情况
            json_str = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
            data = json.loads(json_str)
            if isinstance(data, dict):
                def _v(val):
                    s = str(val).strip() if val is not None else ""
                    return s if s and s.lower() not in ("null", "none", "") else None
                # PS 鉴别字段（合并在同一 JSON 响应里）
                _AUTH_TO_RISK = {
                    "真实": "低", "疑似修改": "中",
                    "高度疑似PS": "高", "几乎确定伪造": "高",
                }
                ps_auth = str(data.get("ps_overall_authenticity") or "真实").strip()
                ps_risk = _AUTH_TO_RISK.get(ps_auth, "低")

                def _jlist(val):
                    if isinstance(val, list):
                        return [str(x) for x in val if x]
                    if isinstance(val, str) and val:
                        return [val]
                    return []

                raw_regions = data.get("ps_suspect_regions") or []
                ps_regions = [
                    r if isinstance(r, dict) else {"description": str(r), "reason": ""}
                    for r in raw_regions if r
                ] if isinstance(raw_regions, list) else []

                return {
                    "cert_type":        _v(data.get("cert_type")),
                    "student_name":     _v(data.get("student_name")),
                    "school":           _v(data.get("school")),
                    "department":       _v(data.get("department")),
                    "principal":        _v(data.get("principal")),
                    "grad_year":        _v(data.get("grad_year")),
                    "enrollment_year":  _v(data.get("enrollment_year")),
                    "issue_year":       _v(data.get("issue_year")),
                    "issue_date":       _v(data.get("issue_date")),
                    "cert_no":          _v(data.get("cert_no")),
                    "student_id_no":    _v(data.get("student_id_no")),
                    "gender":           _v(data.get("gender")),
                    "birth_year":       _v(data.get("birth_year")),
                    "supervisor":       _v(data.get("supervisor")),
                    "_raw_text":        data.get("raw_text") or text,
                    # PS 鉴别结果（以 _ps_ 前缀传递，供 extract_and_cache 单独保存）
                    "_ps_overall_authenticity": ps_auth,
                    "_ps_risk_level":            ps_risk,
                    "_ps_tampering_probability": (
                        float(data["ps_tampering_probability"])
                        if data.get("ps_tampering_probability") is not None else None
                    ),
                    "_ps_confidence": (
                        float(data["ps_confidence"])
                        if data.get("ps_confidence") is not None else None
                    ),
                    "_ps_key_evidence":         _jlist(data.get("ps_key_evidence")),
                    "_ps_no_tampering_evidence": _jlist(data.get("ps_no_tampering_evidence")),
                    "_ps_suspect_regions":       ps_regions,
                    "_ps_recommendation":        str(data.get("ps_recommendation") or "").strip(),
                }
        except (json.JSONDecodeError, Exception):
            logger.debug("JSON parse failed, falling back to regex for text[:80]=%s", text[:80])

        # ── 回退：正则解析原始文本 ───────────────────────────────
        fields: Dict[str, Any] = {
            "cert_type": None, "student_name": None,
            "school": None, "department": None,
            "principal": None,
            "grad_year": None, "enrollment_year": None,
            "issue_year": None, "issue_date": None,
            "cert_no": None, "student_id_no": None,
            "gender": None, "birth_year": None,
            "supervisor": None,
        }

        # 合并行（处理跨行中文年份，如"一九\n七六年"）
        flat = text.replace('\n', '')

        # ── 毕业年份 ─────────────────────────────────────────────
        # 现代格式：毕业/结业/肄业 + 阿拉伯数字年份
        m = re.search(r'(?:毕业|结业|肄业)[^\d]{0,12}((19|20)\d{2})', text)
        if m:
            fields["grad_year"] = m.group(1)
        else:
            # 现代格式：中文年份 + 毕业（同行内）
            m = re.search(r'([一二三四五六七八九○〇零]{4})年[^\n]{0,5}?(?:毕业|结业)', text)
            if m:
                fields["grad_year"] = _cn_to_year(m.group(1))
        if not fields["grad_year"]:
            # 旧式格式："至[中文年]年"——在校截止年即毕业年（跨行合并处理）
            m = re.search(r'至([一二三四五六七八九○〇零]{4})年', flat)
            if m:
                fields["grad_year"] = _cn_to_year(m.group(1))

        # ── 入学年份 ─────────────────────────────────────────────
        # 现代格式：入学/招生/录取/入读 + 阿拉伯数字年份
        m = re.search(r'(?:入学|招生|录取|入读)[^\d]{0,12}((19|20)\d{2})', text)
        if m:
            fields["enrollment_year"] = m.group(1)
        if not fields["enrollment_year"]:
            # 旧式格式："于[中文年]年"——入校起始年（跨行合并处理）
            m = re.search(r'于([一二三四五六七八九○〇零]{4})年', flat)
            if m:
                fields["enrollment_year"] = _cn_to_year(m.group(1))

        # ── 发证年份 ─────────────────────────────────────────────
        # 现代格式：发证/颁发/签发/此证/日期 + 阿拉伯数字年份
        m = re.search(r'(?:发证|颁发|签发|此证|日期)[^\d]{0,12}((19|20)\d{2})', text)
        if m:
            fields["issue_year"] = m.group(1)
        if not fields["issue_year"]:
            # 现代格式：中文年份 + 发证关键词
            m = re.search(r'([一二三四五六七八九○〇零]{4})年[^\d]{0,10}(?:发证|颁发|签发)', text)
            if m:
                fields["issue_year"] = _cn_to_year(m.group(1))
        if not fields["issue_year"]:
            # 旧式格式：行内含"日"的中文年份（如"一九九四年七月十日"签发行）
            m = re.search(r'([一二三四五六七八九○〇零]{4})年[^\n]*日', text)
            if m:
                fields["issue_year"] = _cn_to_year(m.group(1))

        # ── 性别 ─────────────────────────────────────────────────
        if re.search(r'性别[：:\s]*男', text):
            fields["gender"] = "男"
        elif re.search(r'性别[：:\s]*女', text):
            fields["gender"] = "女"

        # ── 证书编号 ─────────────────────────────────────────────
        m = re.search(r'(?:证书编号|学历编号|编号)[：:\s]*([A-Za-z0-9\-]{6,25})', text)
        if m:
            fields["cert_no"] = m.group(1)

        # ── 学校名称 ─────────────────────────────────────────────
        m = re.search(
            r'([^\n\r\s，。、]{2,20}(?:大学|学院|学校|中学|职业技术学院|职业学院|技术学院))',
            text
        )
        if m:
            fields["school"] = m.group(1).strip()

        # ── 校长/院长 ─────────────────────────────────────────────
        m = re.search(r'(?:校长|院长)[：:\s]*([^\n\s，。]{2,6})', text)
        if m:
            fields["principal"] = m.group(1).strip()

        # ── 出生年份 ─────────────────────────────────────────────
        # 身份证号：第7-10位为出生年
        m = re.search(r'\d{17}[\dXx]', text)
        if m:
            fields["birth_year"] = m.group()[6:10]
        else:
            # 现代格式：出生 + 阿拉伯数字年份
            m = re.search(r'出生[：:\s]*((19|20)\d{2})', text)
            if m:
                fields["birth_year"] = m.group(1)
        if not fields["birth_year"]:
            # 旧式格式："生于/出生[中文年]年"（跨行合并处理）
            m = re.search(r'(?:生于|出生)([一二三四五六七八九○〇零]{4})年', flat)
            if m:
                fields["birth_year"] = _cn_to_year(m.group(1))

        # ── 证书类型 ──────────────────────────────────────────────
        m = re.search(r'(毕业证书|学位证书|结业证书|肄业证书|在读证明|学历证书)', text)
        if m:
            fields["cert_type"] = m.group(1)

        # ── 学生姓名 ──────────────────────────────────────────────
        # "学生XXX性别" 格式
        m = re.search(r'学生\s*([^\s性别男女，,。\d]{2,5})\s*性别', text)
        if not m:
            # "兹证明XXX同学/同志" 格式
            m = re.search(r'兹证明\s*([^\s，,。]{2,5})\s*(?:同学|同志)', text)
        if m:
            fields["student_name"] = m.group(1).strip()

        # ── 学籍号码 ──────────────────────────────────────────────
        m = re.search(r'学籍号(?:码)?[：:]\s*(\d{8,20})', text)
        if m:
            fields["student_id_no"] = m.group(1)

        # ── 院系/学部 ─────────────────────────────────────────────
        # "在本校XX学习" 格式（最常见）
        m = re.search(r'在本校\s*([^\s，,。]{2,20})\s*学习', text)
        if not m:
            # "XX系/学院/学部/专业" 格式
            m = re.search(r'([^\s，,。\n]{2,15}(?:学院|系|学部|专业|学科))', text)
        if m:
            fields["department"] = m.group(1).strip()

        # ── 监制单位 ──────────────────────────────────────────────
        m = re.search(r'([^\n\s，,。]{4,20})监制', text)
        if m:
            fields["supervisor"] = m.group(1).strip()

        # ── 完整发证日期 ──────────────────────────────────────────
        fields["issue_date"] = _parse_cn_date(flat)
        # 若已有 issue_year 但没有 issue_date，从 flat 里补充年份
        if not fields["issue_date"] and fields.get("issue_year"):
            fields["issue_date"] = fields["issue_year"]

        return fields

    def extract_and_cache(self, entity_id: int, image: Image.Image) -> Dict[str, Any]:
        """运行OCR（含PS鉴别）并写入 MySQL，已有记录则直接返回；OCR已缓存但无PS时补跑PS"""
        cached = _db_load_one(entity_id)
        if cached is not None:
            # OCR 有缓存，检查 PS 结果是否存在；若无则用独立服务补跑
            try:
                from app.database import get_ps_result
                from app.fraud.ps_service import ps_detection_service
                if get_ps_result(entity_id) is None:
                    logger.info("OCR cached but no PS result, running PS for entity_id=%s", entity_id)
                    ps_detection_service.detect_and_save(entity_id, image)
            except Exception as e:
                logger.warning("Supplemental PS detection failed for id=%s: %s", entity_id, e)
            return cached
        try:
            text = self.extract_text(image)
            fields = self.parse_fields(text)
            if "_raw_text" not in fields:
                fields["_raw_text"] = text
            _db_save_one(entity_id, fields)
            logger.info("OCR saved to MySQL for entity_id=%s", entity_id)
            # 若 JSON 响应中包含 PS 鉴别结果，自动保存
            _save_ps_from_fields(entity_id, fields)
            return fields
        except Exception as e:
            logger.error("OCR failed for entity_id=%s: %s", entity_id, e)
            return {}

    def get_cached(self, entity_id: int) -> Optional[Dict[str, Any]]:
        """获取 MySQL 中的 OCR 结果，不存在返回 None"""
        return _db_load_one(entity_id)

    def get_all_cached(self) -> Dict[str, Any]:
        """获取 MySQL 中全部 OCR 结果"""
        return _db_load_all()

    def clear_cache(self, entity_id: int) -> None:
        """从 MySQL 删除指定证件的 OCR 记录（供重新识别使用）"""
        _db_delete_one(entity_id)

    def run_batch(self, id_image_map: Dict[int, Image.Image]) -> Dict[str, Any]:
        """对所有在 MySQL 中无记录的 ID 批量运行 OCR"""
        existing = _db_load_all()
        for entity_id, image in id_image_map.items():
            key = str(entity_id)
            if key in existing:
                continue
            try:
                text = self.extract_text(image)
                fields = self.parse_fields(text)
                if "_raw_text" not in fields:
                    fields["_raw_text"] = text
                _db_save_one(entity_id, fields)
                _save_ps_from_fields(entity_id, fields)
                existing[key] = fields
                logger.info("OCR done for entity_id=%s", entity_id)
            except Exception as e:
                logger.error("OCR failed for entity_id=%s: %s", entity_id, e)
                existing[key] = {"_ocr_error": str(e)}
        return existing


ocr_service = OCRService()
