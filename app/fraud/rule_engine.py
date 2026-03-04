"""
规则引擎 - 跨证件逻辑校验
无需外部数据，纯批内文件互比对 + 单张逻辑校验

覆盖欺诈特征：
- 学制年限异常
- 发证年份早于毕业年份
- 入学年龄不符
- 照片年龄/性别与证件不符
- 同校同年校长不一致
- 不同学校出现相同校长
- 同一学校校长任期超过12年
- 证书编号重复/高度相似
"""
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

VALID_STUDY_DURATIONS = {2, 3, 4, 5}
MIN_ENROLLMENT_AGE = 15
MAX_ENROLLMENT_AGE = 30
PRINCIPAL_MAX_TENURE = 12
CERT_NO_SIMILARITY_THRESHOLD = 0.85


def _str_similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class RuleEngine:

    def check_single(
        self,
        fields: Dict[str, Any],
        face_info: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        单张证件规则校验
        face_info: {"age": int, "gender": str/'M'/'F'/0/1, "det_score": float}
        """
        issues = []

        # 规则1：学制合理性（入学到毕业年份差）
        try:
            if fields.get("grad_year") and fields.get("enrollment_year"):
                study_years = int(fields["grad_year"]) - int(fields["enrollment_year"])
                if study_years not in VALID_STUDY_DURATIONS:
                    issues.append({
                        "rule": "学制年限异常",
                        "detail": (
                            f"入学 {fields['enrollment_year']} 至毕业 {fields['grad_year']}，"
                            f"共 {study_years} 年，不在合理范围 {sorted(VALID_STUDY_DURATIONS)}"
                        ),
                        "severity": "高" if study_years < 0 else "中",
                    })
        except (ValueError, TypeError):
            pass

        # 规则2：发证年份不得早于毕业年份
        try:
            if fields.get("issue_year") and fields.get("grad_year"):
                if int(fields["issue_year"]) < int(fields["grad_year"]):
                    issues.append({
                        "rule": "发证年份早于毕业年份",
                        "detail": (
                            f"发证年份 {fields['issue_year']} < 毕业年份 {fields['grad_year']}"
                        ),
                        "severity": "高",
                    })
        except (ValueError, TypeError):
            pass

        # 规则3：入学年龄校验
        try:
            if fields.get("birth_year") and fields.get("enrollment_year"):
                age = int(fields["enrollment_year"]) - int(fields["birth_year"])
                if not (MIN_ENROLLMENT_AGE <= age <= MAX_ENROLLMENT_AGE):
                    issues.append({
                        "rule": "入学年龄异常",
                        "detail": (
                            f"出生年份 {fields['birth_year']}，入学 {fields['enrollment_year']}，"
                            f"入学时年龄 {age} 岁（合理范围 {MIN_ENROLLMENT_AGE}-{MAX_ENROLLMENT_AGE}）"
                        ),
                        "severity": "高",
                    })
        except (ValueError, TypeError):
            pass

        # 规则4：人脸年龄估计（证件照应在16-35岁）
        if face_info:
            estimated_age = face_info.get("age")
            if estimated_age and isinstance(estimated_age, (int, float)):
                if not (16 <= int(estimated_age) <= 35):
                    issues.append({
                        "rule": "照片年龄异常",
                        "detail": (
                            f"人脸年龄估计 {int(estimated_age)} 岁，"
                            "超出在校生合理范围（16-35岁）"
                        ),
                        "severity": "中",
                    })

        # 规则5：性别与人脸不符
        if face_info and fields.get("gender"):
            face_gender_raw = face_info.get("gender")
            if face_gender_raw is not None:
                if isinstance(face_gender_raw, (int, float)):
                    face_gender = "男" if face_gender_raw >= 0.5 else "女"
                else:
                    face_gender = (
                        "男" if str(face_gender_raw).upper() in ("M", "MALE", "1") else "女"
                    )
                cert_gender = fields["gender"]
                if face_gender != cert_gender:
                    issues.append({
                        "rule": "性别与照片不符",
                        "detail": (
                            f"证件标注性别【{cert_gender}】，"
                            f"人脸识别判断为【{face_gender}】"
                        ),
                        "severity": "高",
                    })

        return issues

    def check_batch(self, all_fields: Dict[str, Dict]) -> List[Dict]:
        """
        跨证件批量规则校验
        all_fields: {str(entity_id): fields_dict}
        """
        issues = []

        by_school_year: Dict[tuple, List] = defaultdict(list)
        by_principal: Dict[str, List] = defaultdict(list)
        school_principal_years: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        cert_nos: List[tuple] = []

        for eid, fields in all_fields.items():
            if fields.get("_ocr_error"):
                continue
            school = fields.get("school")
            grad_year = fields.get("grad_year")
            principal = fields.get("principal")
            cert_no = fields.get("cert_no")

            if school and grad_year:
                by_school_year[(school, grad_year)].append((eid, principal))

            if principal:
                by_principal[principal].append({
                    "id": eid, "school": school, "grad_year": grad_year,
                })
                if school and grad_year:
                    try:
                        school_principal_years[school][principal].append(int(grad_year))
                    except ValueError:
                        pass

            if cert_no:
                cert_nos.append((eid, cert_no))

        # 规则6：同校同年毕业，校长签名不一致
        for (school, grad_year), entries in by_school_year.items():
            principals = {e[1] for e in entries if e[1]}
            if len(principals) > 1:
                issues.append({
                    "rule": "同校同年校长不一致",
                    "detail": (
                        f"【{school}】{grad_year}年毕业证出现 {len(principals)} 个不同校长："
                        + "、".join(sorted(principals))
                    ),
                    "severity": "高",
                    "related_ids": [e[0] for e in entries],
                })

        # 规则7：不同学校出现相同校长
        for principal, entries in by_principal.items():
            schools = {e["school"] for e in entries if e["school"]}
            if len(schools) > 1:
                issues.append({
                    "rule": "不同学校校长相同",
                    "detail": (
                        f"校长『{principal}』出现在 {len(schools)} 所不同学校："
                        + "、".join(sorted(schools))
                    ),
                    "severity": "高",
                    "related_ids": [e["id"] for e in entries],
                })

        # 规则8：同一学校校长任期超过12年
        for school, principal_data in school_principal_years.items():
            for principal, years in principal_data.items():
                if len(years) < 2:
                    continue
                tenure = max(years) - min(years) + 1
                if tenure > PRINCIPAL_MAX_TENURE:
                    issues.append({
                        "rule": "校长任期异常",
                        "detail": (
                            f"【{school}】校长『{principal}』任期跨度 {tenure} 年"
                            f"（{min(years)}-{max(years)}），超过 {PRINCIPAL_MAX_TENURE} 年"
                        ),
                        "severity": "中",
                        "related_ids": [],
                    })

        # 规则9：证书编号完全相同（不同人）
        no_to_ids: Dict[str, List[str]] = defaultdict(list)
        for eid, no in cert_nos:
            no_to_ids[no].append(eid)
        for cert_no, dup_ids in no_to_ids.items():
            if len(dup_ids) > 1:
                issues.append({
                    "rule": "证书编号重复",
                    "detail": (
                        f"证书编号『{cert_no}』被 {len(dup_ids)} 人共用："
                        + "、".join(str(i) for i in dup_ids)
                    ),
                    "severity": "高",
                    "related_ids": dup_ids,
                })

        # 规则10：证书编号高度相似（不同人、不同编号）
        for i in range(len(cert_nos)):
            for j in range(i + 1, len(cert_nos)):
                id1, no1 = cert_nos[i]
                id2, no2 = cert_nos[j]
                if no1 == no2:
                    continue  # 已被规则9覆盖
                sim = _str_similar(no1, no2)
                if sim >= CERT_NO_SIMILARITY_THRESHOLD:
                    issues.append({
                        "rule": "证书编号高度相似",
                        "detail": (
                            f"ID {id1} 编号『{no1}』与 ID {id2} 编号『{no2}』"
                            f"相似度 {sim:.1%}"
                        ),
                        "severity": "高",
                        "related_ids": [id1, id2],
                    })

        return issues


rule_engine = RuleEngine()
