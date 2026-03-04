"""
SQLAlchemy ORM 数据模型
"""
import json
from datetime import datetime, timezone
from sqlalchemy import (
    Column, BigInteger, Integer, String, Boolean,
    Float, Text, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Certificate(Base):
    """证件基本信息（上传时写入）"""
    __tablename__ = "certificates"

    id = Column(BigInteger, primary_key=True, comment="entity_id，与Milvus一致")
    filename = Column(String(255), comment="原始文件名")
    has_face = Column(Boolean, default=False, comment="是否检测到人脸")
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="上传时间")


class OcrResult(Base):
    """OCR提取的结构化字段（analyze时写入，幂等）"""
    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(BigInteger, ForeignKey("certificates.id", ondelete="CASCADE"), index=True)
    # ── 基础字段 ────────────────────────────────────────────────────
    cert_type = Column(String(50), comment="证书类型（毕业证书/学位证书等）")
    student_name = Column(String(50), comment="学生姓名")
    school = Column(String(255), comment="学校名称")
    department = Column(String(100), comment="院系/学部/专业")
    principal = Column(String(100), comment="校长/院长")
    grad_year = Column(String(10), comment="毕业年份")
    enrollment_year = Column(String(10), comment="入学年份")
    issue_year = Column(String(10), comment="发证年份")
    issue_date = Column(String(30), comment="完整发证日期（YYYY-MM-DD）")
    cert_no = Column(String(100), comment="证书编号")
    student_id_no = Column(String(50), comment="学籍号码")
    gender = Column(String(10), comment="性别")
    birth_year = Column(String(10), comment="出生年份")
    supervisor = Column(String(100), comment="监制单位")
    raw_text = Column(Text, comment="OCR原始文本")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "cert_type": self.cert_type, "student_name": self.student_name,
            "school": self.school, "department": self.department,
            "principal": self.principal,
            "grad_year": self.grad_year, "enrollment_year": self.enrollment_year,
            "issue_year": self.issue_year, "issue_date": self.issue_date,
            "cert_no": self.cert_no, "student_id_no": self.student_id_no,
            "gender": self.gender, "birth_year": self.birth_year,
            "supervisor": self.supervisor,
        }


class ForensicResult(Base):
    """图像取证结果（analyze时写入，幂等）"""
    __tablename__ = "forensic_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(BigInteger, ForeignKey("certificates.id", ondelete="CASCADE"), index=True)
    ela_std = Column(Float, comment="ELA标准差")
    ela_suspicious = Column(Boolean, comment="ELA是否可疑")
    photo_edge_suspicious = Column(Boolean, comment="照片边缘是否异常")
    has_red_seal = Column(Boolean, comment="是否有红色印章")
    seal_printed = Column(Boolean, comment="印章是否为打印")
    risk_level = Column(String(10), comment="风险等级：高/中/低")
    risk_flags = Column(Text, comment="风险标记列表（JSON）")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def get_flags(self):
        try:
            return json.loads(self.risk_flags) if self.risk_flags else []
        except Exception:
            return []


class CrossValidateRun(Base):
    """交叉核验运行记录"""
    __tablename__ = "cross_validate_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_certs = Column(Integer, comment="参与核验的证件数")
    ocr_success = Column(Integer, comment="OCR成功数")
    issue_count = Column(Integer, comment="发现问题总数")
    high_severity_count = Column(Integer, comment="高风险问题数")
    message = Column(Text, comment="核验摘要")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PSDetectionResult(Base):
    """多模态 PS/篡改鉴别结果"""
    __tablename__ = "ps_detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(BigInteger, ForeignKey("certificates.id", ondelete="CASCADE"), index=True)
    # ── 核心判断 ────────────────────────────────────────────────────
    overall_authenticity = Column(String(20), comment="真实/疑似修改/高度疑似PS/几乎确定伪造")
    risk_level = Column(String(10), comment="风险等级：高/中/低（由overall_authenticity衍生）")
    tampering_probability = Column(Float, comment="篡改概率 0-100")
    confidence = Column(Float, comment="模型置信度 0-100")
    # ── 证据列表 ─────────────────────────────────────────────────────
    key_evidence = Column(Text, comment="支持篡改的关键证据（JSON数组）")
    no_tampering_evidence = Column(Text, comment="支持真实的证据（JSON数组）")
    suspect_regions = Column(Text, comment="可疑区域列表（JSON，每项含description/reason）")
    # ── 结论 ─────────────────────────────────────────────────────────
    recommendation = Column(String(500), comment="处置建议")
    raw_response = Column(Text, comment="模型原始响应（含thinking链）")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def _jl(self, col):
        try:
            return json.loads(col) if col else []
        except Exception:
            return []

    def get_key_evidence(self):         return self._jl(self.key_evidence)
    def get_no_tampering_evidence(self): return self._jl(self.no_tampering_evidence)
    def get_suspect_regions(self):      return self._jl(self.suspect_regions)


class FraudIssue(Base):
    """欺诈问题记录（单张分析或批量核验时写入）"""
    __tablename__ = "fraud_issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("cross_validate_runs.id", ondelete="SET NULL"), nullable=True,
                    comment="关联的批量核验run，单张分析时为NULL")
    entity_id = Column(BigInteger, nullable=True, comment="单张分析时的证件ID")
    rule_name = Column(String(100), comment="触发的规则名称")
    detail = Column(Text, comment="详细描述")
    severity = Column(String(10), comment="严重程度：高/中/低")
    related_ids = Column(Text, comment="关联证件ID列表（JSON）")
    issue_type = Column(String(20), default="batch", comment="single/batch")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def get_related_ids(self):
        try:
            return json.loads(self.related_ids) if self.related_ids else []
        except Exception:
            return []
