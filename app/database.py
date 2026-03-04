"""
MySQL 数据库连接与初始化
使用 SQLAlchemy + PyMySQL
"""
import json
import logging
from contextlib import contextmanager
from typing import Optional, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def init_db() -> None:
    """初始化数据库连接，自动建表"""
    global _engine, _SessionLocal

    url = settings.MYSQL_URL
    logger.info("Connecting to MySQL: %s", url.replace(settings.MYSQL_PASSWORD, "***"))

    _engine = create_engine(
        url,
        pool_pre_ping=True,       # 每次获取连接前ping一下，避免连接超时断开
        pool_recycle=3600,         # 连接1小时后回收，防止MySQL 8h超时断开
        pool_size=5,
        max_overflow=10,
        echo=False,
    )

    # 自动建表（表已存在则跳过）
    Base.metadata.create_all(bind=_engine)
    logger.info("MySQL tables initialized")

    # 向后兼容：为已存在的 ocr_results 表新增列（create_all 不会更新已有表结构）
    _migrate_ocr_results(_engine)
    # 向后兼容：为已存在的 ps_detection_results 表新增列
    _migrate_ps_results(_engine)

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def _migrate_ocr_results(engine) -> None:
    """为 ocr_results 表按需添加新列（幂等，列已存在时 MySQL 会报错但被忽略）"""
    new_cols = [
        ("cert_type",     "VARCHAR(50)  COMMENT '证书类型'"),
        ("student_name",  "VARCHAR(50)  COMMENT '学生姓名'"),
        ("department",    "VARCHAR(100) COMMENT '院系/学部/专业'"),
        ("issue_date",    "VARCHAR(30)  COMMENT '完整发证日期'"),
        ("student_id_no", "VARCHAR(50)  COMMENT '学籍号码'"),
        ("supervisor",    "VARCHAR(100) COMMENT '监制单位'"),
    ]
    with engine.connect() as conn:
        for col_name, col_def in new_cols:
            try:
                conn.execute(text(f"ALTER TABLE ocr_results ADD COLUMN {col_name} {col_def}"))
                conn.commit()
                logger.info("Migration: added column ocr_results.%s", col_name)
            except Exception as col_err:
                err_str = str(col_err)
                if "Duplicate column name" not in err_str:
                    logger.warning("Migration unexpected error for %s: %s", col_name, err_str)


def get_db() -> Generator[Session, None, None]:
    """FastAPI依赖注入用的Session生成器"""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """手动使用的上下文管理器（非依赖注入场景）"""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def is_db_available() -> bool:
    """检查数据库连接是否可用"""
    if _engine is None:
        return False
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ── 便捷写入函数 ──────────────────────────────────────────────────

def save_certificate(entity_id: int, filename: str, has_face: bool) -> None:
    """上传后记录证件基本信息（已存在则跳过）"""
    from app.models import Certificate
    if not is_db_available():
        return
    try:
        with db_session() as db:
            exists = db.get(Certificate, entity_id)
            if not exists:
                db.add(Certificate(id=entity_id, filename=filename, has_face=has_face))
    except Exception as e:
        logger.warning("save_certificate failed: %s", e)


def save_ocr_result(entity_id: int, fields: dict) -> None:
    """保存OCR结果（已存在则覆盖）"""
    from app.models import OcrResult
    if not is_db_available():
        return
    try:
        with db_session() as db:
            # 幂等：删旧插新
            db.query(OcrResult).filter(OcrResult.entity_id == entity_id).delete()
            db.add(OcrResult(
                entity_id=entity_id,
                school=fields.get("school"),
                principal=fields.get("principal"),
                grad_year=fields.get("grad_year"),
                enrollment_year=fields.get("enrollment_year"),
                issue_year=fields.get("issue_year"),
                cert_no=fields.get("cert_no"),
                gender=fields.get("gender"),
                birth_year=fields.get("birth_year"),
                raw_text=fields.get("_raw_text", "")[:10000],  # 截断防超长
            ))
    except Exception as e:
        logger.warning("save_ocr_result failed: %s", e)


def save_forensic_result(entity_id: int, forensic: dict) -> None:
    """保存图像取证结果（已存在则覆盖）"""
    from app.models import ForensicResult
    if not is_db_available():
        return
    try:
        ela = forensic.get("ela", {})
        edge = forensic.get("photo_edge", {})
        seal = forensic.get("seal", {})
        with db_session() as db:
            db.query(ForensicResult).filter(ForensicResult.entity_id == entity_id).delete()
            db.add(ForensicResult(
                entity_id=entity_id,
                ela_std=ela.get("ela_std"),
                ela_suspicious=ela.get("is_suspicious"),
                photo_edge_suspicious=edge.get("is_suspicious"),
                has_red_seal=seal.get("has_red_seal"),
                seal_printed=seal.get("is_printed_seal"),
                risk_level=forensic.get("risk_level", "低"),
                risk_flags=json.dumps(forensic.get("risk_flags", []), ensure_ascii=False),
            ))
    except Exception as e:
        logger.warning("save_forensic_result failed: %s", e)


def save_cross_validate_run(
    total_certs: int,
    ocr_success: int,
    issues: list,
    message: str,
) -> Optional[int]:
    """保存交叉核验运行记录及所有问题，返回run_id"""
    from app.models import CrossValidateRun, FraudIssue
    if not is_db_available():
        return None
    try:
        with db_session() as db:
            run = CrossValidateRun(
                total_certs=total_certs,
                ocr_success=ocr_success,
                issue_count=len(issues),
                high_severity_count=sum(1 for i in issues if i.get("severity") == "高"),
                message=message,
            )
            db.add(run)
            db.flush()  # 获取自增ID
            for issue in issues:
                db.add(FraudIssue(
                    run_id=run.id,
                    rule_name=issue.get("rule", ""),
                    detail=issue.get("detail", ""),
                    severity=issue.get("severity", "低"),
                    related_ids=json.dumps(
                        [str(i) for i in issue.get("related_ids", [])],
                        ensure_ascii=False,
                    ),
                    issue_type="batch",
                ))
            return run.id
    except Exception as e:
        logger.warning("save_cross_validate_run failed: %s", e)
        return None


def save_single_analyze_issues(entity_id: int, issues: list) -> None:
    """保存单张证件的规则校验问题"""
    from app.models import FraudIssue
    if not is_db_available() or not issues:
        return
    try:
        with db_session() as db:
            # 清除该证件的历史单张问题
            db.query(FraudIssue).filter(
                FraudIssue.entity_id == entity_id,
                FraudIssue.issue_type == "single",
            ).delete()
            for issue in issues:
                db.add(FraudIssue(
                    entity_id=entity_id,
                    rule_name=issue.get("rule", ""),
                    detail=issue.get("detail", ""),
                    severity=issue.get("severity", "低"),
                    related_ids="[]",
                    issue_type="single",
                ))
    except Exception as e:
        logger.warning("save_single_analyze_issues failed: %s", e)


def _migrate_ps_results(engine) -> None:
    """为 ps_detection_results 表按需添加新列（幂等，列已存在时忽略）"""
    new_cols = [
        ("overall_authenticity", "VARCHAR(20)  COMMENT '真实/疑似修改/高度疑似PS/几乎确定伪造'"),
        ("tampering_probability","FLOAT        COMMENT '篡改概率 0-100'"),
        ("key_evidence",         "TEXT         COMMENT '支持篡改的关键证据（JSON数组）'"),
        ("no_tampering_evidence","TEXT         COMMENT '支持真实的证据（JSON数组）'"),
        ("suspect_regions",      "TEXT         COMMENT '可疑区域列表（JSON）'"),
        ("recommendation",       "VARCHAR(500) COMMENT '处置建议'"),
    ]
    with engine.connect() as conn:
        for col_name, col_def in new_cols:
            try:
                conn.execute(text(f"ALTER TABLE ps_detection_results ADD COLUMN {col_name} {col_def}"))
                conn.commit()
                logger.info("Migration: added column ps_detection_results.%s", col_name)
            except Exception:
                pass  # 列已存在，忽略


def save_ps_result(entity_id: int, result: dict) -> None:
    """保存 PS 鉴别结果（幂等：先删后插）"""
    from app.models import PSDetectionResult, Certificate
    if not is_db_available():
        return
    try:
        # 事务1：确保 certificates 父行存在
        try:
            with db_session() as db:
                if not db.query(Certificate).filter(Certificate.id == entity_id).first():
                    db.add(Certificate(id=entity_id))
        except Exception:
            pass
        # 事务2：写入 PS 结果
        with db_session() as db:
            db.query(PSDetectionResult).filter(PSDetectionResult.entity_id == entity_id).delete()
            db.add(PSDetectionResult(
                entity_id=entity_id,
                overall_authenticity=result.get("overall_authenticity"),
                risk_level=result.get("risk_level"),
                tampering_probability=result.get("tampering_probability"),
                confidence=result.get("confidence"),
                key_evidence=json.dumps(result.get("key_evidence") or [], ensure_ascii=False),
                no_tampering_evidence=json.dumps(result.get("no_tampering_evidence") or [], ensure_ascii=False),
                suspect_regions=json.dumps(result.get("suspect_regions") or [], ensure_ascii=False),
                recommendation=result.get("recommendation") or "",
                raw_response=(result.get("_raw_response") or "")[:5000],
            ))
    except Exception as e:
        logger.warning("save_ps_result failed for id=%s: %s", entity_id, e)


def _ps_row_to_dict(r) -> dict:
    """PSDetectionResult ORM 行 → dict"""
    return {
        "overall_authenticity": r.overall_authenticity,
        "risk_level":           r.risk_level,
        "tampering_probability": r.tampering_probability,
        "confidence":           r.confidence,
        "key_evidence":         r.get_key_evidence(),
        "no_tampering_evidence": r.get_no_tampering_evidence(),
        "suspect_regions":      r.get_suspect_regions(),
        "recommendation":       r.recommendation,
    }


def get_ps_result(entity_id: int) -> Optional[dict]:
    """读取单条 PS 鉴别结果，不存在返回 None"""
    from app.models import PSDetectionResult
    if not is_db_available():
        return None
    try:
        with db_session() as db:
            r = db.query(PSDetectionResult).filter(PSDetectionResult.entity_id == entity_id).first()
            if r is None:
                return None
            return _ps_row_to_dict(r)
    except Exception as e:
        logger.warning("get_ps_result failed for id=%s: %s", entity_id, e)
        return None


def get_all_ps_results() -> dict:
    """读取全部 PS 鉴别结果，返回 {str(entity_id): result_dict}"""
    from app.models import PSDetectionResult
    if not is_db_available():
        return {}
    try:
        with db_session() as db:
            rows = db.query(PSDetectionResult).all()
            return {str(r.entity_id): _ps_row_to_dict(r) for r in rows}
    except Exception as e:
        logger.warning("get_all_ps_results failed: %s", e)
        return {}
