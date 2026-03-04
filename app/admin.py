"""
SQLAdmin 后台管理
访问地址：http://localhost:8001/admin
默认账号：admin / admin123（可通过 ADMIN_USERNAME / ADMIN_PASSWORD 环境变量覆盖）
"""
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request

from app.config import settings
from app.models import (
    Certificate, OcrResult, PSDetectionResult,
    ForensicResult, FraudIssue, CrossValidateRun,
)


# ── 认证后端 ──────────────────────────────────────────────────────

class _AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        if (
            form.get("username") == settings.ADMIN_USERNAME
            and form.get("password") == settings.ADMIN_PASSWORD
        ):
            request.session["admin_token"] = "ok"
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("admin_token") == "ok"


# ── ModelViews ────────────────────────────────────────────────────

class CertificateAdmin(ModelView, model=Certificate):
    name = "证件"
    name_plural = "证件列表"
    icon = "fa-solid fa-id-card"

    column_list = [
        Certificate.id,
        Certificate.filename,
        Certificate.has_face,
        Certificate.upload_time,
    ]
    column_searchable_list  = [Certificate.filename]
    column_sortable_list    = [Certificate.id, Certificate.upload_time]
    column_default_sort     = (Certificate.id, True)   # 倒序

    can_create = False   # 由上传接口管理
    can_edit   = True
    can_delete = True
    page_size  = 50


class OcrResultAdmin(ModelView, model=OcrResult):
    name = "OCR 结果"
    name_plural = "OCR 识别结果"
    icon = "fa-solid fa-file-lines"

    column_list = [
        OcrResult.entity_id,
        OcrResult.student_name,
        OcrResult.school,
        OcrResult.cert_type,
        OcrResult.cert_no,
        OcrResult.grad_year,
        OcrResult.issue_date,
        OcrResult.created_at,
    ]
    column_searchable_list = [
        OcrResult.student_name,
        OcrResult.school,
        OcrResult.cert_no,
        OcrResult.cert_type,
    ]
    column_sortable_list = [
        OcrResult.entity_id,
        OcrResult.grad_year,
        OcrResult.created_at,
    ]
    column_default_sort = (OcrResult.entity_id, True)

    # 隐藏 raw_text（内容太长），通过详情页查看
    column_details_exclude_list = []
    column_exclude_list = [OcrResult.raw_text]

    can_create = False
    can_edit   = True    # 允许人工修正识别错误
    can_delete = True
    page_size  = 50


class PSDetectionResultAdmin(ModelView, model=PSDetectionResult):
    name = "PS 鉴别结果"
    name_plural = "PS/篡改鉴别结果"
    icon = "fa-solid fa-magnifying-glass"

    column_list = [
        PSDetectionResult.entity_id,
        PSDetectionResult.overall_authenticity,
        PSDetectionResult.risk_level,
        PSDetectionResult.tampering_probability,
        PSDetectionResult.confidence,
        PSDetectionResult.recommendation,
        PSDetectionResult.created_at,
    ]
    column_sortable_list = [
        PSDetectionResult.entity_id,
        PSDetectionResult.tampering_probability,
        PSDetectionResult.confidence,
        PSDetectionResult.created_at,
    ]
    column_default_sort = (PSDetectionResult.tampering_probability, True)  # 高风险排前面

    # 隐藏超长字段（详情页仍可见）
    column_exclude_list = [PSDetectionResult.raw_response]

    can_create = False
    can_edit   = False
    can_delete = True
    page_size  = 50


class ForensicResultAdmin(ModelView, model=ForensicResult):
    name = "图像取证"
    name_plural = "图像取证结果"
    icon = "fa-solid fa-microscope"

    column_list = [
        ForensicResult.entity_id,
        ForensicResult.risk_level,
        ForensicResult.ela_std,
        ForensicResult.ela_suspicious,
        ForensicResult.photo_edge_suspicious,
        ForensicResult.has_red_seal,
        ForensicResult.seal_printed,
        ForensicResult.created_at,
    ]
    column_sortable_list = [
        ForensicResult.entity_id,
        ForensicResult.ela_std,
        ForensicResult.risk_level,
        ForensicResult.created_at,
    ]
    column_default_sort = (ForensicResult.entity_id, True)

    can_create = False
    can_edit   = False
    can_delete = True
    page_size  = 50


class FraudIssueAdmin(ModelView, model=FraudIssue):
    name = "欺诈问题"
    name_plural = "欺诈问题记录"
    icon = "fa-solid fa-triangle-exclamation"

    column_list = [
        FraudIssue.id,
        FraudIssue.entity_id,
        FraudIssue.severity,
        FraudIssue.rule_name,
        FraudIssue.detail,
        FraudIssue.issue_type,
        FraudIssue.created_at,
    ]
    column_searchable_list = [
        FraudIssue.rule_name,
        FraudIssue.detail,
    ]
    column_sortable_list = [
        FraudIssue.id,
        FraudIssue.severity,
        FraudIssue.created_at,
    ]
    column_default_sort = (FraudIssue.created_at, True)

    can_create = False
    can_edit   = False
    can_delete = True
    page_size  = 50


class CrossValidateRunAdmin(ModelView, model=CrossValidateRun):
    name = "批量核验"
    name_plural = "批量核验记录"
    icon = "fa-solid fa-list-check"

    column_list = [
        CrossValidateRun.id,
        CrossValidateRun.total_certs,
        CrossValidateRun.ocr_success,
        CrossValidateRun.issue_count,
        CrossValidateRun.high_severity_count,
        CrossValidateRun.message,
        CrossValidateRun.created_at,
    ]
    column_sortable_list = [
        CrossValidateRun.id,
        CrossValidateRun.created_at,
        CrossValidateRun.issue_count,
    ]
    column_default_sort = (CrossValidateRun.id, True)

    can_create = False
    can_edit   = False
    can_delete = True
    page_size  = 20


# ── 挂载函数（由 main.py lifespan 调用）──────────────────────────

def setup_admin(app, engine) -> None:
    """初始化并挂载 SQLAdmin，需在 init_db() 之后调用"""
    import logging as _logging
    _log = _logging.getLogger(__name__)
    if "change-in-production" in settings.SECRET_KEY or settings.ADMIN_PASSWORD == "admin123":
        _log.warning(
            "⚠️  Admin using default credentials/SECRET_KEY — set ADMIN_USERNAME, "
            "ADMIN_PASSWORD, SECRET_KEY env vars before deploying to production!"
        )
    authentication_backend = _AdminAuth(secret_key=settings.SECRET_KEY)
    admin = Admin(
        app,
        engine,
        title="学历稽核 · 管理后台",
        authentication_backend=authentication_backend,
    )
    admin.add_view(CertificateAdmin)
    admin.add_view(OcrResultAdmin)
    admin.add_view(PSDetectionResultAdmin)
    admin.add_view(ForensicResultAdmin)
    admin.add_view(FraudIssueAdmin)
    admin.add_view(CrossValidateRunAdmin)
