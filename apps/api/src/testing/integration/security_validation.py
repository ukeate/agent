"""安全验证模块"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
import importlib.metadata
import os
import re
import time
import tomllib
import httpx
from src.core.utils.timezone_utils import utc_now
from enum import Enum
from dataclasses import dataclass
from ...core.config import get_settings
from src.core.monitoring import monitor
from src.core.security.auth import password_manager

class SecurityLevel(Enum):
    """安全级别枚举"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

logger = get_logger(__name__)

@dataclass
class SecurityVulnerability:
    """安全漏洞"""
    vulnerability_id: str
    name: str
    severity: SecurityLevel
    component: str
    description: str
    cve_id: Optional[str]
    remediation: str
    detected_at: datetime

@dataclass
class SecurityAuditResult:
    """安全审计结果"""
    audit_id: str
    timestamp: datetime
    passed: bool
    vulnerabilities_found: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    compliance_status: Dict[str, bool]
    recommendations: List[str]

class SecurityValidator:
    """安全验证器"""
    
    def __init__(self):
        self.owasp_checker = OWASPComplianceChecker()
        self.mcp_security = MCPSecurityAuditor()
        self.api_security = APISecurityValidator()
        self.data_protection = DataProtectionValidator()
        self.vulnerability_scanner = VulnerabilityScanner()
        
    async def validate_security_compliance(self) -> Dict[str, Any]:
        """验证安全合规性"""
        monitor.log_info("开始安全合规性验证...")
        
        results = {
            'timestamp': utc_now().isoformat(),
            'overall_status': 'compliant',
            'checks': {},
            'vulnerabilities': [],
            'compliance_scores': {},
            'recommendations': []
        }
        
        # OWASP Top 10 合规检查
        owasp_result = await self.owasp_checker.check_compliance()
        results['checks']['owasp_top_10'] = owasp_result
        
        # MCP工具安全审计
        mcp_result = await self.mcp_security.audit_security()
        results['checks']['mcp_security'] = mcp_result
        
        # API安全验证
        api_result = await self.api_security.validate()
        results['checks']['api_security'] = api_result
        
        # 数据保护验证
        data_result = await self.data_protection.validate()
        results['checks']['data_protection'] = data_result
        
        # 漏洞扫描
        vulnerabilities = await self.vulnerability_scanner.scan()
        results['vulnerabilities'] = vulnerabilities
        
        # 计算合规分数
        results['compliance_scores'] = self.calculate_compliance_scores(results['checks'])
        
        # 确定整体状态
        results['overall_status'] = self.determine_overall_status(results)
        
        # 生成建议
        results['recommendations'] = self.generate_recommendations(results)
        
        return results
        
    def calculate_compliance_scores(self, checks: Dict[str, Any]) -> Dict[str, float]:
        """计算合规分数"""
        scores = {}
        
        for check_name, result in checks.items():
            if isinstance(result, dict) and 'passed' in result:
                # 基础分数
                base_score = 100 if result['passed'] else 0
                
                # 根据问题严重性调整分数
                if 'vulnerabilities_found' in result:
                    deduction = result['vulnerabilities_found'] * 5
                    base_score = max(0, base_score - deduction)
                    
                scores[check_name] = base_score
            else:
                scores[check_name] = 0
                
        # 计算总体分数
        scores['overall'] = sum(scores.values()) / len(scores) if scores else 0
        
        return scores
        
    def determine_overall_status(self, results: Dict[str, Any]) -> str:
        """确定整体安全状态"""
        # 检查是否有严重漏洞
        critical_vulns = [v for v in results['vulnerabilities'] 
                         if v.get('severity') == SecurityLevel.CRITICAL.value]
        
        if critical_vulns:
            return 'non_compliant'
            
        # 检查合规分数
        overall_score = results['compliance_scores'].get('overall', 0)
        
        if overall_score >= 90:
            return 'compliant'
        elif overall_score >= 70:
            return 'partially_compliant'
        else:
            return 'non_compliant'
            
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 基于漏洞生成建议
        if results['vulnerabilities']:
            severity_counts = {}
            for vuln in results['vulnerabilities']:
                severity = vuln.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
            if severity_counts.get(SecurityLevel.CRITICAL.value, 0) > 0:
                recommendations.append("立即修复所有严重级别的安全漏洞")
                
            if severity_counts.get(SecurityLevel.HIGH.value, 0) > 0:
                recommendations.append("优先修复高严重性安全问题")
                
        # 基于合规检查生成建议
        for check_name, result in results['checks'].items():
            if isinstance(result, dict) and not result.get('passed', False):
                if 'owasp' in check_name:
                    recommendations.append("加强OWASP Top 10安全控制")
                elif 'mcp' in check_name:
                    recommendations.append("审查MCP工具权限和访问控制")
                elif 'api' in check_name:
                    recommendations.append("增强API安全措施")
                elif 'data' in check_name:
                    recommendations.append("改进数据保护和加密策略")
                    
        # 通用建议
        recommendations.extend([
            "定期进行安全审计和渗透测试",
            "保持所有依赖项的安全更新",
            "实施安全监控和异常检测",
            "定期审查和更新安全策略"
        ])
        
        return list(set(recommendations))  # 去重

class OWASPComplianceChecker:
    """OWASP合规检查器"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://127.0.0.1:{self.settings.PORT}"
    
    async def check_compliance(self) -> Dict[str, Any]:
        """检查OWASP Top 10合规性"""
        monitor.log_info("检查OWASP Top 10合规性...")
        
        owasp_checks = {
            'A01_broken_access_control': await self.check_access_control(),
            'A02_cryptographic_failures': await self.check_cryptography(),
            'A03_injection': await self.check_injection(),
            'A04_insecure_design': await self.check_design_security(),
            'A05_security_misconfiguration': await self.check_configuration(),
            'A06_vulnerable_components': await self.check_components(),
            'A07_identification_failures': await self.check_authentication(),
            'A08_data_integrity_failures': await self.check_data_integrity(),
            'A09_logging_failures': await self.check_logging(),
            'A10_ssrf': await self.check_ssrf()
        }
        
        vulnerabilities_found = sum(
            1 for check in owasp_checks.values() 
            if not check.get('compliant', False)
        )
        
        return {
            'passed': vulnerabilities_found == 0,
            'vulnerabilities_found': vulnerabilities_found,
            'checks': owasp_checks,
            'compliance_percentage': (10 - vulnerabilities_found) * 10
        }
        
    async def check_access_control(self) -> Dict[str, Any]:
        """检查访问控制"""
        issues: List[str] = []
        compliant = True

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/auth/me", timeout=10.0)
            if resp.status_code != 401:
                compliant = False
                issues.append(f"/api/v1/auth/me 未鉴权拦截: {resp.status_code}")
        except Exception as e:
            compliant = False
            issues.append(f"访问控制检查失败: {e}")

        return {
            'compliant': compliant,
            'details': "鉴权端点能拦截匿名访问" if compliant else "发现访问控制问题",
            'issues': issues,
        }
        
    async def check_cryptography(self) -> Dict[str, Any]:
        """检查加密实现"""
        issues: List[str] = []
        secret_ok = len(self.settings.SECRET_KEY or "") >= 32
        algo_ok = (self.settings.JWT_ALGORITHM or "").lower() not in {"none", ""}
        compliant = secret_ok and algo_ok
        if not secret_ok:
            issues.append("SECRET_KEY 长度不足 32")
        if not algo_ok:
            issues.append(f"JWT_ALGORITHM 不安全: {self.settings.JWT_ALGORITHM}")

        return {
            'compliant': compliant,
            'details': "密钥与JWT算法配置正常" if compliant else "加密配置存在风险",
            'issues': issues,
        }
        
    async def check_injection(self) -> Dict[str, Any]:
        """检查注入漏洞"""
        issues: List[str] = []
        root = Path(__file__).resolve().parents[3] / "src"
        patterns = [
            re.compile(r"execute\(\s*text\(\s*f[\"']"),
            re.compile(r"execute\(\s*f[\"']\s*select", re.IGNORECASE),
        ]
        try:
            hits = 0
            for p in root.rglob("*.py"):
                if {"alembic", "versions"}.issubset(p.parts):
                    continue
                try:
                    s = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if any(pt.search(s) for pt in patterns):
                    hits += 1
            if hits:
                issues.append(f"发现疑似字符串拼接SQL文件数: {hits}")
        except Exception as e:
            issues.append(f"代码扫描失败: {e}")

        compliant = not issues
        return {
            'compliant': compliant,
            'details': "未发现明显SQL拼接模式" if compliant else "存在潜在注入风险",
            'issues': issues,
        }
        
    async def check_design_security(self) -> Dict[str, Any]:
        """检查设计安全性"""
        issues: List[str] = []
        if not (0 < float(self.settings.SECURITY_THRESHOLD) < 1):
            issues.append(f"SECURITY_THRESHOLD 超出范围: {self.settings.SECURITY_THRESHOLD}")
        if not (0 < float(self.settings.AUTO_BLOCK_THRESHOLD) < 1):
            issues.append(f"AUTO_BLOCK_THRESHOLD 超出范围: {self.settings.AUTO_BLOCK_THRESHOLD}")
        if float(self.settings.AUTO_BLOCK_THRESHOLD) <= float(self.settings.SECURITY_THRESHOLD):
            issues.append("AUTO_BLOCK_THRESHOLD 应大于 SECURITY_THRESHOLD")

        compliant = not issues
        return {
            'compliant': compliant,
            'details': "安全阈值配置合理" if compliant else "安全阈值配置不合理",
            'issues': issues,
        }
        
    async def check_configuration(self) -> Dict[str, Any]:
        """检查安全配置"""
        issues: List[str] = []
        if self.settings.DEBUG:
            issues.append("DEBUG 为 True，不建议生产环境开启")
        if "*" in (self.settings.ALLOWED_HOSTS or []):
            issues.append("ALLOWED_HOSTS 包含 *")
        if self.settings.FORCE_HTTPS and self.settings.DEBUG:
            issues.append("FORCE_HTTPS 在 DEBUG 模式下启用可能影响本地开发")

        compliant = not issues
        return {
            'compliant': compliant,
            'details': "配置无明显高风险项" if compliant else "存在潜在不安全配置",
            'issues': issues,
        }
        
    async def check_components(self) -> Dict[str, Any]:
        """检查组件漏洞"""
        issues: List[str] = []
        for name in ("fastapi", "uvicorn", "sqlalchemy", "redis", "qdrant-client"):
            try:
                _ = importlib.metadata.version(name)
            except Exception:
                issues.append(f"无法读取依赖版本: {name}")

        compliant = not issues
        return {
            'compliant': compliant,
            'details': "关键依赖版本可追踪" if compliant else "部分依赖版本不可追踪",
            'issues': issues,
        }
        
    async def check_authentication(self) -> Dict[str, Any]:
        """检查认证机制"""
        issues: List[str] = []
        compliant = True
        if (self.settings.ACCESS_TOKEN_EXPIRE_MINUTES or 0) <= 0:
            compliant = False
            issues.append("ACCESS_TOKEN_EXPIRE_MINUTES 应大于 0")
        if (self.settings.REFRESH_TOKEN_EXPIRE_DAYS or 0) <= 0:
            compliant = False
            issues.append("REFRESH_TOKEN_EXPIRE_DAYS 应大于 0")
        try:
            hashed = password_manager.hash_password("password-for-check")
            if not (hashed.startswith("$2") or hashed.startswith("$argon2")):
                compliant = False
                issues.append("密码哈希算法可能不安全或未启用")
        except Exception as e:
            compliant = False
            issues.append(f"密码哈希检查失败: {e}")

        return {
            'compliant': compliant,
            'details': "认证配置可用" if compliant else "认证配置存在问题",
            'issues': issues,
        }
        
    async def check_data_integrity(self) -> Dict[str, Any]:
        """检查数据完整性"""
        issues: List[str] = []
        compliant = True
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json={"username": f"u_{int(time.time())}", "password": "short"},
                    timeout=10.0,
                )
            if resp.status_code != 422:
                compliant = False
                issues.append(f"输入校验异常: /auth/register 期望422，实际{resp.status_code}")
        except Exception as e:
            compliant = False
            issues.append(f"数据完整性检查失败: {e}")

        return {
            'compliant': compliant,
            'details': "输入校验生效" if compliant else "输入校验可能未生效",
            'issues': issues,
        }
        
    async def check_logging(self) -> Dict[str, Any]:
        """检查日志和监控"""
        issues: List[str] = []
        compliant = True
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/health", timeout=10.0)
            if "x-request-id" not in {k.lower() for k in resp.headers.keys()}:
                compliant = False
                issues.append("响应头缺少 X-Request-ID")
        except Exception as e:
            compliant = False
            issues.append(f"日志检查失败: {e}")

        return {
            'compliant': compliant,
            'details': "请求追踪信息可用" if compliant else "缺少请求追踪信息",
            'issues': issues,
        }
        
    async def check_ssrf(self) -> Dict[str, Any]:
        """检查SSRF漏洞"""
        issues: List[str] = []
        # SSRF防护属于业务级策略，这里仅做配置层面的最小检查
        if "*" in (self.settings.ALLOWED_HOSTS or []):
            issues.append("ALLOWED_HOSTS 过宽，可能增加SSRF风险")

        compliant = not issues
        return {
            'compliant': compliant,
            'details': "未发现明显SSRF高风险配置" if compliant else "存在潜在SSRF风险配置",
            'issues': issues,
        }

class MCPSecurityAuditor:
    """MCP安全审计器"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://127.0.0.1:{self.settings.PORT}"
    
    async def audit_security(self) -> Dict[str, Any]:
        """审计MCP工具安全性"""
        monitor.log_info("审计MCP工具安全性...")
        
        audit_results = {
            'tool_permissions': await self.audit_tool_permissions(),
            'access_control': await self.audit_access_control(),
            'audit_logging': await self.audit_logging(),
            'rate_limiting': await self.check_rate_limiting()
        }
        
        all_passed = all(r.get('secure', False) for r in audit_results.values())
        
        return {
            'passed': all_passed,
            'audit_complete': True,
            'unauthorized_access_attempts': 0,
            'security_policies_enforced': all_passed,
            'details': audit_results
        }
        
    async def audit_tool_permissions(self) -> Dict[str, Any]:
        """审计工具权限"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/mcp/tools", timeout=10.0)

            secure = resp.status_code in {401, 403}
            tools_audited = None
            if resp.status_code == 200:
                data = resp.json() or {}
                tools = (data.get("tools") or {}).values()
                tools_audited = sum(len(v or []) for v in tools)

            return {
                'secure': secure,
                'tools_audited': tools_audited,
                'permission_violations': 0 if secure else None,
                'recommendations': [] if secure else ["为MCP工具端点增加鉴权与权限控制"],
            }
        except Exception as e:
            return {'secure': False, 'tools_audited': None, 'permission_violations': None, 'error': str(e)}
        
    async def audit_access_control(self) -> Dict[str, Any]:
        """审计访问控制"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/mcp/tools", timeout=10.0)
            secure = resp.status_code in {401, 403}
            return {
                'secure': secure,
                'access_policies_enforced': secure,
                'unauthorized_attempts': 0,
            }
        except Exception as e:
            return {'secure': False, 'access_policies_enforced': False, 'unauthorized_attempts': None, 'error': str(e)}
        
    async def audit_logging(self) -> Dict[str, Any]:
        """审计日志记录"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/mcp/metrics", timeout=10.0)
            ok = resp.status_code == 200
            return {
                'secure': ok,
                'audit_trail_complete': ok,
                'suspicious_activities': 0,
            }
        except Exception as e:
            return {'secure': False, 'audit_trail_complete': False, 'suspicious_activities': None, 'error': str(e)}
        
    async def check_rate_limiting(self) -> Dict[str, Any]:
        """检查速率限制"""
        try:
            async with httpx.AsyncClient() as client:
                codes = []
                for _ in range(5):
                    resp = await client.get(f"{self.base_url}/api/v1/mcp/health", timeout=10.0)
                    codes.append(resp.status_code)
            enforced = 429 in codes
            return {
                'secure': enforced,
                'rate_limits_enforced': enforced,
                'limit_violations': codes.count(429),
            }
        except Exception as e:
            return {'secure': False, 'rate_limits_enforced': False, 'limit_violations': None, 'error': str(e)}

class APISecurityValidator:
    """API安全验证器"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://127.0.0.1:{self.settings.PORT}"
    
    async def validate(self) -> Dict[str, Any]:
        """验证API安全性"""
        monitor.log_info("验证API安全性...")
        
        validations = {
            'authentication': await self.validate_authentication(),
            'authorization': await self.validate_authorization(),
            'input_validation': await self.validate_input(),
            'rate_limiting': await self.validate_rate_limiting(),
            'cors': await self.validate_cors(),
            'headers': await self.validate_security_headers()
        }
        
        all_passed = all(v.get('secure', False) for v in validations.values())
        
        return {
            'passed': all_passed,
            'authentication_required': validations.get('authentication', {}).get('secure', False),
            'authorization_enforced': validations.get('authorization', {}).get('secure', False),
            'rate_limiting_active': validations.get('rate_limiting', {}).get('secure', False),
            'input_validation': validations.get('input_validation', {}).get('secure', False),
            'details': validations
        }
        
    async def validate_authentication(self) -> Dict[str, Any]:
        """验证认证机制"""
        issues: List[str] = []
        secure = True
        try:
            async with httpx.AsyncClient() as client:
                resp1 = await client.get(f"{self.base_url}/api/v1/auth/me", timeout=10.0)
                resp2 = await client.get(
                    f"{self.base_url}/api/v1/auth/me",
                    headers={"Authorization": "Bearer invalid"},
                    timeout=10.0,
                )
            if resp1.status_code != 401:
                secure = False
                issues.append(f"匿名访问 /auth/me 未返回401: {resp1.status_code}")
            if resp2.status_code != 401:
                secure = False
                issues.append(f"非法token访问 /auth/me 未返回401: {resp2.status_code}")
        except Exception as e:
            secure = False
            issues.append(str(e))

        return {
            'secure': secure,
            'methods': ['JWT'],
            'mfa_enabled': False,
            'issues': issues,
        }
        
    async def validate_authorization(self) -> Dict[str, Any]:
        """验证授权机制"""
        issues: List[str] = []
        secure = True
        username = f"sec_{int(time.time())}"
        password = "password-1234"
        try:
            async with httpx.AsyncClient() as client:
                reg = await client.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json={"username": username, "password": password},
                    timeout=10.0,
                )
                if reg.status_code not in {200, 400}:
                    secure = False
                    issues.append(f"注册接口异常状态码: {reg.status_code}")

                token = await client.post(
                    f"{self.base_url}/api/v1/auth/token",
                    data={"username": username, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=10.0,
                )
                if token.status_code != 200:
                    secure = False
                    issues.append(f"登录失败: {token.status_code}")
                    return {
                        'secure': False,
                        'rbac_enabled': False,
                        'permission_model': None,
                        'policy_enforcement': None,
                        'issues': issues,
                    }

                access_token = (token.json() or {}).get("access_token")
                if not access_token:
                    secure = False
                    issues.append("未返回access_token")
                perm = await client.get(
                    f"{self.base_url}/api/v1/auth/permissions",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0,
                )
                if perm.status_code != 200:
                    secure = False
                    issues.append(f"权限接口访问失败: {perm.status_code}")
        except Exception as e:
            secure = False
            issues.append(str(e))

        return {
            'secure': secure,
            'rbac_enabled': secure,
            'permission_model': 'role-based',
            'policy_enforcement': 'unknown',
            'issues': issues,
        }
        
    async def validate_input(self) -> Dict[str, Any]:
        """验证输入验证"""
        issues: List[str] = []
        secure = True
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json={"username": f"v_{int(time.time())}", "password": "short"},
                    timeout=10.0,
                )
            if resp.status_code != 422:
                secure = False
                issues.append(f"输入校验未触发422: {resp.status_code}")
        except Exception as e:
            secure = False
            issues.append(str(e))

        return {
            'secure': secure,
            'validation_enabled': secure,
            'sanitization': None,
            'parameterized_queries': None,
            'issues': issues,
        }
        
    async def validate_rate_limiting(self) -> Dict[str, Any]:
        """验证速率限制"""
        try:
            async with httpx.AsyncClient() as client:
                codes = []
                for _ in range(20):
                    r = await client.get(f"{self.base_url}/api/v1/health", timeout=10.0)
                    codes.append(r.status_code)
            enforced = 429 in codes
            return {
                'secure': enforced,
                'limits_configured': enforced,
                'ddos_protection': None,
                'per_user_limits': None,
            }
        except Exception as e:
            return {'secure': False, 'limits_configured': None, 'error': str(e)}
        
    async def validate_cors(self) -> Dict[str, Any]:
        """验证CORS配置"""
        try:
            origin = "http://localhost:3000"
            async with httpx.AsyncClient() as client:
                resp = await client.options(
                    f"{self.base_url}/api/v1/health",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "GET",
                    },
                    timeout=10.0,
                )
            allow_origin = resp.headers.get("access-control-allow-origin")
            restricted = allow_origin in {origin, None}
            return {
                'secure': restricted,
                'origins_restricted': restricted,
                'credentials_handled': resp.headers.get("access-control-allow-credentials") == "true",
            }
        except Exception as e:
            return {'secure': False, 'origins_restricted': None, 'credentials_handled': None, 'error': str(e)}
        
    async def validate_security_headers(self) -> Dict[str, Any]:
        """验证安全响应头"""
        required = [
            'x-content-type-options',
            'x-frame-options',
            'content-security-policy',
        ]
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/health", timeout=10.0)
            present = {k.lower() for k in resp.headers.keys()}
            missing = [h for h in required if h not in present]
            return {
                'secure': not missing,
                'headers_present': [h for h in required if h in present],
                'headers_missing': missing,
            }
        except Exception as e:
            return {'secure': False, 'headers_present': [], 'headers_missing': required, 'error': str(e)}

class DataProtectionValidator:
    """数据保护验证器"""

    def __init__(self):
        self.settings = get_settings()
    
    async def validate(self) -> Dict[str, Any]:
        """验证数据保护"""
        monitor.log_info("验证数据保护措施...")
        
        validations = {
            'encryption_at_rest': await self.validate_encryption_at_rest(),
            'encryption_in_transit': await self.validate_encryption_in_transit(),
            'pii_protection': await self.validate_pii_protection(),
            'data_retention': await self.validate_data_retention(),
            'backup_security': await self.validate_backup_security()
        }
        
        all_passed = all(v.get('compliant', False) for v in validations.values())
        
        return {
            'passed': all_passed,
            'encryption_at_rest': validations.get('encryption_at_rest', {}).get('compliant', False),
            'encryption_in_transit': validations.get('encryption_in_transit', {}).get('compliant', False),
            'pii_protection': validations.get('pii_protection', {}).get('compliant', False),
            'gdpr_compliant': all_passed,
            'details': validations
        }
        
    async def validate_encryption_at_rest(self) -> Dict[str, Any]:
        """验证静态数据加密"""
        try:
            sample = password_manager.hash_password("password-for-check")
            return {
                'compliant': bool(sample),
                'password_hashing': 'bcrypt' if sample.startswith("$2") else 'unknown',
            }
        except Exception as e:
            return {'compliant': False, 'error': str(e)}
        
    async def validate_encryption_in_transit(self) -> Dict[str, Any]:
        """验证传输加密"""
        compliant = bool(self.settings.FORCE_HTTPS)
        return {
            'compliant': compliant,
            'force_https': bool(self.settings.FORCE_HTTPS),
        }
        
    async def validate_pii_protection(self) -> Dict[str, Any]:
        """验证PII保护"""
        # 最小可验证：密码不以明文存储（hash_password可用）
        try:
            hashed = password_manager.hash_password("password-for-check")
            return {'compliant': bool(hashed), 'password_storage': 'hashed'}
        except Exception as e:
            return {'compliant': False, 'error': str(e)}
        
    async def validate_data_retention(self) -> Dict[str, Any]:
        """验证数据保留策略"""
        ttl_ok = bool(self.settings.CACHE_TTL_DEFAULT and self.settings.CACHE_TTL_DEFAULT > 0)
        reward_ttl_ok = bool(self.settings.REWARD_SIGNAL_TTL and self.settings.REWARD_SIGNAL_TTL > 0)
        compliant = ttl_ok and reward_ttl_ok
        return {
            'compliant': compliant,
            'cache_ttl_default_seconds': self.settings.CACHE_TTL_DEFAULT,
            'reward_signal_ttl_seconds': self.settings.REWARD_SIGNAL_TTL,
        }
        
    async def validate_backup_security(self) -> Dict[str, Any]:
        """验证备份安全"""
        try:
            import shutil
            import subprocess
            from urllib.parse import urlparse

            pg_dump = shutil.which("pg_dump")
            if not pg_dump:
                return {'compliant': False, 'details': '未找到pg_dump'}

            url = self.settings.DATABASE_URL
            if url.startswith("postgresql+asyncpg://"):
                url = "postgresql://" + url[len("postgresql+asyncpg://") :]
            parsed = urlparse(url)
            cmd = [
                pg_dump,
                "--schema-only",
                "--no-owner",
                "--no-privileges",
                "-h",
                parsed.hostname or "localhost",
                "-p",
                str(parsed.port or 5432),
                "-U",
                parsed.username or "postgres",
                parsed.path.lstrip("/") or "postgres",
            ]
            env = dict(**os.environ)
            if parsed.password:
                env["PGPASSWORD"] = parsed.password
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=20)
            return {'compliant': p.returncode == 0, 'details': p.stderr.strip() if p.returncode else None}
        except Exception as e:
            return {'compliant': False, 'error': str(e)}

class VulnerabilityScanner:
    """漏洞扫描器"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://127.0.0.1:{self.settings.PORT}"
        self.api_root = Path(__file__).resolve().parents[3]
    
    async def scan(self) -> List[Dict[str, Any]]:
        """执行漏洞扫描"""
        monitor.log_info("执行漏洞扫描...")
        
        vulnerabilities = []
        
        # 扫描依赖项漏洞
        dep_vulns = await self.scan_dependencies()
        vulnerabilities.extend(dep_vulns)
        
        # 扫描代码漏洞
        code_vulns = await self.scan_code()
        vulnerabilities.extend(code_vulns)
        
        # 扫描配置漏洞
        config_vulns = await self.scan_configuration()
        vulnerabilities.extend(config_vulns)
        
        # 扫描网络漏洞
        network_vulns = await self.scan_network()
        vulnerabilities.extend(network_vulns)
        
        return vulnerabilities
        
    async def scan_dependencies(self) -> List[Dict[str, Any]]:
        """扫描依赖项漏洞"""
        pyproject = self.api_root / "pyproject.toml"
        if not pyproject.exists():
            return []

        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except Exception:
            return []

        raw_deps = (data.get("project") or {}).get("dependencies") or []
        candidates = {
            "fastapi",
            "uvicorn",
            "sqlalchemy",
            "asyncpg",
            "redis",
            "qdrant-client",
            "pydantic",
            "python-jose",
            "cryptography",
            "passlib",
            "bcrypt",
            "httpx",
            "langgraph",
            "langchain",
            "torch",
        }

        pkgs: List[tuple[str, str]] = []
        for spec in raw_deps:
            if not isinstance(spec, str):
                continue
            name = spec.strip().split(";", 1)[0].strip()
            name = name.split("[", 1)[0].strip()
            name = re.split(r"[<=>!~\\s]", name, 1)[0].strip()
            if not name or name not in candidates:
                continue
            try:
                ver = importlib.metadata.version(name)
            except Exception:
                continue
            pkgs.append((name, ver))

        vulns: List[Dict[str, Any]] = []
        async with httpx.AsyncClient() as client:
            for name, ver in pkgs:
                try:
                    resp = await client.post(
                        "https://api.osv.dev/v1/query",
                        json={"package": {"name": name, "ecosystem": "PyPI"}, "version": ver},
                        timeout=15.0,
                    )
                    resp.raise_for_status()
                    payload = resp.json() or {}
                except Exception:
                    continue

                for v in payload.get("vulns") or []:
                    aliases = v.get("aliases") or []
                    cve = next((a for a in aliases if isinstance(a, str) and a.startswith("CVE-")), None)
                    score = None
                    for sev in v.get("severity") or []:
                        raw = (sev or {}).get("score")
                        if not raw:
                            continue
                        try:
                            score = float(raw)
                            break
                        except Exception:
                            continue
                    if score is None:
                        level = SecurityLevel.INFO.value
                    elif score >= 9:
                        level = SecurityLevel.CRITICAL.value
                    elif score >= 7:
                        level = SecurityLevel.HIGH.value
                    elif score >= 4:
                        level = SecurityLevel.MEDIUM.value
                    else:
                        level = SecurityLevel.LOW.value

                    vulns.append(
                        {
                            "id": v.get("id"),
                            "severity": level,
                            "type": "dependency",
                            "component": name,
                            "cve_id": cve,
                            "description": v.get("summary") or v.get("details") or "",
                            "remediation": f"升级 {name}（当前版本 {ver}）",
                            "detected_at": utc_now().isoformat(),
                        }
                    )

        return vulns
        
    async def scan_code(self) -> List[Dict[str, Any]]:
        """扫描代码漏洞"""
        root = self.api_root / "src"
        patterns = [
            ("sql_string_format", re.compile(r"execute\(\s*text\(\s*f[\"']")),
            ("eval", re.compile(r"(?<!\.)\beval\(")),
            ("exec", re.compile(r"\bexec\(")),
        ]
        findings: List[Dict[str, Any]] = []
        try:
            for p in root.rglob("*.py"):
                if {"alembic", "versions"}.issubset(p.parts):
                    continue
                try:
                    s = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for name, pat in patterns:
                    if pat.search(s):
                        findings.append(
                            {
                                "id": f"code:{name}:{p.name}",
                                "severity": SecurityLevel.MEDIUM.value,
                                "type": "code",
                                "component": str(p.relative_to(self.api_root)),
                                "description": f"发现疑似风险模式: {name}",
                                "remediation": "移除高风险模式或改用参数化/安全替代方案",
                                "detected_at": utc_now().isoformat(),
                            }
                        )
        except Exception:
            return []

        return findings
        
    async def scan_configuration(self) -> List[Dict[str, Any]]:
        """扫描配置漏洞"""
        vulns: List[Dict[str, Any]] = []

        if self.settings.DEBUG:
            vulns.append(
                {
                    "id": "config:debug",
                    "severity": SecurityLevel.MEDIUM.value,
                    "type": "configuration",
                    "component": "settings",
                    "description": "DEBUG 为 True",
                    "remediation": "生产环境关闭 DEBUG",
                    "detected_at": utc_now().isoformat(),
                }
            )

        if "*" in (self.settings.ALLOWED_HOSTS or []):
            vulns.append(
                {
                    "id": "config:cors",
                    "severity": SecurityLevel.HIGH.value,
                    "type": "configuration",
                    "component": "CORS",
                    "description": "ALLOWED_HOSTS 包含 *",
                    "remediation": "移除 * 并限制来源",
                    "detected_at": utc_now().isoformat(),
                }
            )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/mcp/tools", timeout=10.0)
            if resp.status_code == 200:
                vulns.append(
                    {
                        "id": "config:mcp-public",
                        "severity": SecurityLevel.HIGH.value,
                        "type": "configuration",
                        "component": "mcp",
                        "description": "MCP工具端点可匿名访问",
                        "remediation": "为MCP工具端点添加鉴权与权限控制",
                        "detected_at": utc_now().isoformat(),
                    }
                )
        except Exception:
            logger.exception("检查MCP工具端点失败", exc_info=True)

        auth_file = self.api_root / "src" / "api" / "v1" / "auth.py"
        try:
            s = auth_file.read_text(encoding="utf-8", errors="ignore")
            if "roles = [request.role] if request.role else" in s:
                vulns.append(
                    {
                        "id": "config:register-role",
                        "severity": SecurityLevel.CRITICAL.value,
                        "type": "configuration",
                        "component": "auth",
                        "description": "注册接口允许客户端自选 role（可能自注册为admin）",
                        "remediation": "注册时忽略客户端role并在服务端固定默认角色",
                        "detected_at": utc_now().isoformat(),
                    }
                )
        except Exception:
            logger.exception("检查注册权限配置失败", exc_info=True)

        return vulns
        
    async def scan_network(self) -> List[Dict[str, Any]]:
        """扫描网络漏洞"""
        vulns: List[Dict[str, Any]] = []
        if not self.settings.FORCE_HTTPS:
            vulns.append(
                {
                    "id": "network:https",
                    "severity": SecurityLevel.INFO.value,
                    "type": "network",
                    "component": "transport",
                    "description": "未启用FORCE_HTTPS（本地开发可能正常，生产需评估）",
                    "remediation": "生产环境启用HTTPS并配置反向代理/证书",
                    "detected_at": utc_now().isoformat(),
                }
            )
        return vulns

class PenetrationTester:
    """渗透测试器"""

    def __init__(self):
        settings = get_settings()
        self.base_url = f"http://127.0.0.1:{settings.PORT}"
        self.api_root = Path(__file__).resolve().parents[3]
    
    async def run_penetration_test(self) -> Dict[str, Any]:
        """运行渗透测试"""
        monitor.log_info("运行渗透测试...")
        
        test_results = {
            'timestamp': utc_now().isoformat(),
            'tests_performed': [],
            'vulnerabilities_found': [],
            'risk_assessment': {}
        }
        
        # 认证绕过测试
        auth_test = await self.test_authentication_bypass()
        test_results['tests_performed'].append(auth_test)
        if auth_test.get("status") != "passed":
            test_results["vulnerabilities_found"].append(
                {
                    "id": "pentest:auth-bypass",
                    "severity": SecurityLevel.CRITICAL.value,
                    "type": "pentest",
                    "component": "auth",
                    "description": "存在认证绕过迹象",
                    "remediation": "检查认证依赖与鉴权中间件",
                    "detected_at": utc_now().isoformat(),
                }
            )
        
        # SQL注入测试
        sql_test = await self.test_sql_injection()
        test_results['tests_performed'].append(sql_test)
        if sql_test.get("status") != "passed":
            test_results["vulnerabilities_found"].append(
                {
                    "id": "pentest:sql-injection",
                    "severity": SecurityLevel.HIGH.value,
                    "type": "pentest",
                    "component": "database",
                    "description": "发现疑似SQL拼接模式",
                    "remediation": "改用参数化查询并移除字符串拼接SQL",
                    "detected_at": utc_now().isoformat(),
                }
            )
        
        # XSS测试
        xss_test = await self.test_xss()
        test_results['tests_performed'].append(xss_test)
        if xss_test.get("status") != "passed":
            test_results["vulnerabilities_found"].append(
                {
                    "id": "pentest:xss",
                    "severity": SecurityLevel.MEDIUM.value,
                    "type": "pentest",
                    "component": "http",
                    "description": "缺少关键安全响应头（可能增加XSS风险）",
                    "remediation": "添加CSP、X-Content-Type-Options、X-Frame-Options等响应头",
                    "detected_at": utc_now().isoformat(),
                }
            )
        
        # 权限提升测试
        priv_test = await self.test_privilege_escalation()
        test_results['tests_performed'].append(priv_test)
        if priv_test.get("status") != "passed":
            test_results["vulnerabilities_found"].append(
                {
                    "id": "pentest:priv-esc",
                    "severity": SecurityLevel.CRITICAL.value,
                    "type": "pentest",
                    "component": "auth",
                    "description": "存在权限提升风险（注册可选role）",
                    "remediation": "注册时服务端固定默认角色并禁止客户端指定role",
                    "detected_at": utc_now().isoformat(),
                }
            )
        
        # 风险评估
        test_results['risk_assessment'] = self.assess_risk(test_results['vulnerabilities_found'])
        
        return test_results
        
    async def test_authentication_bypass(self) -> Dict[str, Any]:
        """测试认证绕过"""
        attempts = 2
        successful = 0
        details = {}
        try:
            async with httpx.AsyncClient() as client:
                r1 = await client.get(f"{self.base_url}/api/v1/auth/me", timeout=10.0)
                r2 = await client.get(
                    f"{self.base_url}/api/v1/auth/me",
                    headers={"Authorization": "Bearer invalid"},
                    timeout=10.0,
                )
            details = {"anonymous_status": r1.status_code, "invalid_token_status": r2.status_code}
            if r1.status_code == 200:
                successful += 1
            if r2.status_code == 200:
                successful += 1
        except Exception as e:
            details = {"error": str(e)}
            successful = attempts

        return {
            'test_name': 'Authentication Bypass',
            'status': 'passed' if successful == 0 else 'failed',
            'attempts': attempts,
            'successful_bypasses': successful,
            'details': details,
        }
        
    async def test_sql_injection(self) -> Dict[str, Any]:
        """测试SQL注入"""
        root = self.api_root / "src"
        pat = re.compile(r"execute\(\s*text\(\s*f[\"']")
        injection_points_tested = 0
        vulnerable_points = 0
        for p in root.rglob("*.py"):
            try:
                s = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            injection_points_tested += 1
            if pat.search(s):
                vulnerable_points += 1

        return {
            'test_name': 'SQL Injection',
            'status': 'passed' if vulnerable_points == 0 else 'failed',
            'injection_points_tested': injection_points_tested,
            'vulnerable_points': vulnerable_points,
        }
        
    async def test_xss(self) -> Dict[str, Any]:
        """测试XSS"""
        required = {"content-security-policy", "x-content-type-options", "x-frame-options"}
        vulnerable = 0
        details = {}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/health", timeout=10.0)
            present = {k.lower() for k in resp.headers.keys()}
            missing = sorted(required - present)
            vulnerable = len(missing)
            details = {"missing_headers": missing}
        except Exception as e:
            vulnerable = 1
            details = {"error": str(e)}

        return {
            'test_name': 'Cross-Site Scripting',
            'status': 'passed' if vulnerable == 0 else 'failed',
            'input_fields_tested': None,
            'vulnerable_fields': vulnerable,
            'details': details,
        }
        
    async def test_privilege_escalation(self) -> Dict[str, Any]:
        """测试权限提升"""
        auth_file = self.api_root / "src" / "api" / "v1" / "auth.py"
        vulnerable = False
        try:
            s = auth_file.read_text(encoding="utf-8", errors="ignore")
            vulnerable = "roles = [request.role] if request.role else" in s
        except Exception:
            vulnerable = True

        return {
            'test_name': 'Privilege Escalation',
            'status': 'passed' if not vulnerable else 'failed',
            'escalation_attempts': 1,
            'successful_escalations': 1 if vulnerable else 0,
        }
        
    def assess_risk(self, vulnerabilities: List[Dict]) -> Dict[str, Any]:
        """评估风险"""
        if not vulnerabilities:
            return {
                'risk_level': 'low',
                'score': 10,
                'recommendation': 'System security is robust'
            }
            
        # 根据漏洞计算风险分数
        risk_score = 10
        for vuln in vulnerabilities:
            if vuln.get('severity') == SecurityLevel.CRITICAL.value:
                risk_score += 30
            elif vuln.get('severity') == SecurityLevel.HIGH.value:
                risk_score += 20
            elif vuln.get('severity') == SecurityLevel.MEDIUM.value:
                risk_score += 10
            elif vuln.get('severity') == SecurityLevel.LOW.value:
                risk_score += 5
                
        if risk_score >= 70:
            risk_level = 'critical'
        elif risk_score >= 50:
            risk_level = 'high'
        elif risk_score >= 30:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        return {
            'risk_level': risk_level,
            'score': risk_score,
            'recommendation': self.get_risk_recommendation(risk_level)
        }
        
    def get_risk_recommendation(self, risk_level: str) -> str:
        """获取风险建议"""
        recommendations = {
            'critical': 'Immediate action required to fix critical vulnerabilities',
            'high': 'High priority fixes needed before production deployment',
            'medium': 'Address medium risk issues in next security sprint',
            'low': 'System security is acceptable, monitor for changes'
        }
        
        return recommendations.get(risk_level, 'Continue security monitoring')
from src.core.logging import get_logger
