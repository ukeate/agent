"""系统健康监控模块"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
import asyncio
import os
import time
from enum import Enum
from dataclasses import dataclass
import httpx
import psutil
from sqlalchemy import text
from ...core.config import get_settings
from src.core.database import get_db_session
from src.core.monitoring import monitor, monitoring_service
from src.core.qdrant import qdrant_manager
from src.core.redis import get_redis

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_name: str
    status: HealthStatus
    response_time_ms: float
    last_check_time: datetime
    metrics: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]

@dataclass
class DependencyStatus:
    """依赖服务状态"""
    service_name: str
    status: str
    latency_ms: float
    last_successful_connection: datetime
    error_message: Optional[str] = None

class SystemHealthMonitor:
    """系统健康监控器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://127.0.0.1:{self.settings.PORT}"
        self.components = {
            'langgraph': LangGraphMonitor(),
            'autogen': AutoGenMonitor(),
            'pgvector': PgVectorMonitor(),
            'fastapi': FastAPIMonitor(),
            'opentelemetry': OpenTelemetryMonitor(),
            'redis': RedisMonitor(),
            'postgresql': PostgreSQLMonitor(),
            'qdrant': QdrantMonitor()
        }
        self.health_history = []
        self.alert_thresholds = self.load_alert_thresholds()
        
    def load_alert_thresholds(self) -> Dict[str, Any]:
        """加载告警阈值"""
        return {
            'response_time_ms': 500,
            'error_rate_percent': 1,
            'cpu_usage_percent': 80,
            'memory_usage_percent': 85,
            'disk_usage_percent': 90,
            'connection_pool_usage': 90
        }
        
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """运行全面健康检查"""
        monitor.log_info("开始运行系统健康检查...")
        
        health_report = {
            'timestamp': utc_now().isoformat(),
            'overall_status': HealthStatus.HEALTHY.value,
            'components': {},
            'dependencies': {},
            'metrics': {},
            'alerts': [],
            'recommendations': []
        }
        
        # 检查各组件健康状态
        component_statuses = []
        for name, monitor_instance in self.components.items():
            try:
                health = await monitor_instance.check_health()
                health_report['components'][name] = {
                    'status': health.status.value,
                    'response_time_ms': health.response_time_ms,
                    'last_check': health.last_check_time.isoformat(),
                    'metrics': health.metrics,
                    'issues': health.issues,
                    'recommendations': health.recommendations
                }
                component_statuses.append(health.status)
                
                # 检查告警条件
                alerts = self.check_alerts(name, health)
                health_report['alerts'].extend(alerts)
                
            except Exception as e:
                monitor.log_error(f"检查组件 {name} 健康状态失败: {str(e)}")
                health_report['components'][name] = {
                    'status': HealthStatus.UNKNOWN.value,
                    'error': str(e)
                }
                component_statuses.append(HealthStatus.UNKNOWN)
                
        # 确定整体健康状态
        health_report['overall_status'] = self.determine_overall_status(component_statuses)
        
        # 检查依赖服务
        health_report['dependencies'] = await self.check_dependencies()
        
        # 收集系统指标
        health_report['metrics'] = await self.collect_system_metrics()
        
        # 生成建议
        health_report['recommendations'] = self.generate_recommendations(health_report)
        
        # 保存历史记录
        self.health_history.append(health_report)
        
        return health_report
        
    def determine_overall_status(self, statuses: List[HealthStatus]) -> str:
        """确定整体健康状态"""
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY.value
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED.value
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.DEGRADED.value
        else:
            return HealthStatus.HEALTHY.value
            
    def check_alerts(self, component_name: str, health: ComponentHealth) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []
        
        # 响应时间告警
        if health.response_time_ms > self.alert_thresholds['response_time_ms']:
            alerts.append({
                'component': component_name,
                'type': 'response_time',
                'severity': 'warning',
                'message': f'{component_name} 响应时间过长: {health.response_time_ms}ms'
            })
            
        # 组件特定告警
        if health.metrics:
            if 'error_rate' in health.metrics and health.metrics['error_rate'] > self.alert_thresholds['error_rate_percent']:
                alerts.append({
                    'component': component_name,
                    'type': 'error_rate',
                    'severity': 'critical',
                    'message': f'{component_name} 错误率过高: {health.metrics["error_rate"]}%'
                })
                
        return alerts
        
    async def check_dependencies(self) -> Dict[str, Any]:
        """检查依赖服务状态"""
        dependencies = {}
        
        # PostgreSQL
        dependencies['postgresql'] = await self.check_postgresql_status()
        
        # Redis
        dependencies['redis'] = await self.check_redis_status()
        
        # Qdrant
        dependencies['qdrant'] = await self.check_qdrant_status()
        
        # 外部API
        dependencies['external_apis'] = await self.check_external_apis()
        
        return dependencies
        
    async def check_postgresql_status(self) -> Dict[str, Any]:
        """检查PostgreSQL状态"""
        try:
            t0 = time.perf_counter()
            async with get_db_session() as session:
                q0 = time.perf_counter()
                r = await session.execute(text("SELECT 1"))
                r.scalar_one()
                latency_ms = (time.perf_counter() - q0) * 1000

                active = await session.execute(
                    text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                )
                idle = await session.execute(
                    text("SELECT count(*) FROM pg_stat_activity WHERE state = 'idle'")
                )
                size = await session.execute(
                    text("SELECT pg_database_size(current_database())")
                )

            return {
                'status': 'connected',
                'latency_ms': round(latency_ms, 2),
                'connections_active': int(active.scalar_one() or 0),
                'connections_idle': int(idle.scalar_one() or 0),
                'database_size_mb': round(float(size.scalar_one() or 0) / 1024 / 1024, 2),
                'checked_in_ms': round((time.perf_counter() - t0) * 1000, 2),
            }
        except Exception as e:
            return {
                'status': 'disconnected',
                'error': str(e)
            }
            
    async def check_redis_status(self) -> Dict[str, Any]:
        """检查Redis状态"""
        try:
            client = get_redis()
            if not client:
                return {'status': 'disconnected', 'error': 'Redis未初始化'}

            t0 = time.perf_counter()
            await client.ping()
            latency_ms = (time.perf_counter() - t0) * 1000
            info = await client.info()
            keys_count = await client.dbsize()

            hits = float(info.get("keyspace_hits", 0) or 0)
            misses = float(info.get("keyspace_misses", 0) or 0)
            hit_rate = hits / (hits + misses) * 100 if hits + misses > 0 else 0.0

            return {
                'status': 'connected',
                'latency_ms': round(latency_ms, 2),
                'memory_used_mb': round(float(info.get("used_memory", 0) or 0) / 1024 / 1024, 2),
                'keys_count': int(keys_count or 0),
                'hit_rate_percent': round(hit_rate, 2),
                'connected_clients': int(info.get("connected_clients", 0) or 0),
            }
        except Exception as e:
            return {
                'status': 'disconnected',
                'error': str(e)
            }
            
    async def check_qdrant_status(self) -> Dict[str, Any]:
        """检查Qdrant状态"""
        try:
            client = qdrant_manager.get_client()
            t0 = time.perf_counter()
            collections = await asyncio.to_thread(client.get_collections)
            latency_ms = (time.perf_counter() - t0) * 1000

            total_vectors = 0
            index_status = None
            for c in getattr(collections, "collections", []) or []:
                try:
                    info = await asyncio.to_thread(client.get_collection, c.name)
                    total_vectors += int(getattr(info, "points_count", 0) or 0)
                    index_status = getattr(info, "status", None) or index_status
                except Exception:
                    continue

            return {
                'status': 'connected',
                'latency_ms': round(latency_ms, 2),
                'collections_count': len(getattr(collections, "collections", []) or []),
                'vectors_total': total_vectors,
                'index_status': str(index_status) if index_status is not None else None,
            }
        except Exception as e:
            return {
                'status': 'disconnected',
                'error': str(e)
            }
            
    async def check_external_apis(self) -> Dict[str, Any]:
        """检查外部API状态"""
        return {
            'openai': 'configured' if bool(self.settings.OPENAI_API_KEY) else 'missing_key',
            'anthropic': 'configured' if bool(self.settings.ANTHROPIC_API_KEY) else 'missing_key',
        }
        
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        net1 = psutil.net_io_counters()
        await asyncio.sleep(0.2)
        net2 = psutil.net_io_counters()
        dt = 0.2
        net_in_mbps = (net2.bytes_recv - net1.bytes_recv) * 8 / dt / 1_000_000
        net_out_mbps = (net2.bytes_sent - net1.bytes_sent) * 8 / dt / 1_000_000

        now = utc_now()
        perf = await monitoring_service.performance_monitor.get_stats()
        request_rate_qps = float(perf.get("requests_per_minute", 0) or 0) / 60
        error_rate_percent = float(perf.get("error_rate", 0) or 0) * 100

        try:
            active_connections = await asyncio.to_thread(
                lambda: len(psutil.net_connections(kind="inet"))
            )
        except Exception:
            active_connections = None

        return {
            'cpu_usage_percent': round(float(cpu), 2),
            'memory_usage_percent': round(float(mem.percent), 2),
            'disk_usage_percent': round(float(disk.percent), 2),
            'network_in_mbps': round(float(net_in_mbps), 2),
            'network_out_mbps': round(float(net_out_mbps), 2),
            'active_connections': active_connections,
            'request_rate_qps': round(float(request_rate_qps), 2),
            'error_rate_percent': round(float(error_rate_percent), 2),
        }
        
    def generate_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        # 基于告警生成建议
        if health_report['alerts']:
            for alert in health_report['alerts']:
                if alert['type'] == 'response_time':
                    recommendations.append(f"优化 {alert['component']} 的性能配置")
                elif alert['type'] == 'error_rate':
                    recommendations.append(f"检查 {alert['component']} 的错误日志并修复问题")
                    
        # 基于整体状态生成建议
        if health_report['overall_status'] == HealthStatus.DEGRADED.value:
            recommendations.append("系统性能下降，建议进行性能调优")
            
        # 基于依赖状态生成建议
        for dep_name, dep_status in health_report['dependencies'].items():
            if isinstance(dep_status, dict) and dep_status.get('status') == 'disconnected':
                recommendations.append(f"检查 {dep_name} 连接配置")
                
        return recommendations if recommendations else ["系统运行正常，无需特殊操作"]
        
    async def run_production_readiness_check(self) -> Dict[str, Any]:
        """运行生产就绪度检查"""
        monitor.log_info("运行生产就绪度检查...")
        
        checks = {
            'health_check': await self.check_all_components_healthy(),
            'performance_check': await self.check_performance_targets_met(),
            'security_check': await self.check_security_compliance(),
            'monitoring_check': await self.check_monitoring_active(),
            'backup_check': await self.check_backup_configured(),
            'documentation_check': await self.check_documentation_complete()
        }
        
        all_passed = all(check['passed'] for check in checks.values())
        
        return {
            'production_ready': all_passed,
            'checks': checks,
            'blockers': [
                name for name, check in checks.items()
                if not check['passed']
            ],
            'recommendations': self.generate_readiness_recommendations(checks)
        }
        
    async def check_all_components_healthy(self) -> Dict[str, Any]:
        """检查所有组件健康"""
        health_report = await self.run_comprehensive_health_check()
        
        return {
            'passed': health_report['overall_status'] == HealthStatus.HEALTHY.value,
            'details': f"系统健康状态: {health_report['overall_status']}",
            'components_status': health_report['components']
        }
        
    async def check_performance_targets_met(self) -> Dict[str, Any]:
        """检查性能目标"""
        perf = await monitoring_service.performance_monitor.get_stats()
        request_times = list(getattr(monitoring_service.performance_monitor, "request_times", []) or [])
        recent = [r for r in request_times if r.get("timestamp", 0) > time.time() - 60]
        durations = sorted((float(r.get("duration", 0) or 0) * 1000 for r in recent))
        if not durations:
            return {
                'passed': False,
                'details': "缺少请求样本，无法评估性能目标",
                'metrics': {},
            }

        p95_ms = durations[int(len(durations) * 0.95)]
        qps = float(perf.get("requests_per_minute", 0) or 0) / 60
        err_rate = float(perf.get("error_rate", 0) or 0) * 100

        passed = p95_ms <= self.alert_thresholds['response_time_ms'] and err_rate <= self.alert_thresholds['error_rate_percent']
        return {
            'passed': passed,
            'details': "性能目标达成" if passed else "性能目标未达成",
            'metrics': {
                'response_time_p95_ms': round(p95_ms, 2),
                'throughput_qps': round(qps, 2),
                'error_rate_percent': round(err_rate, 2),
                'sample_count': len(durations),
            },
        }
        
    async def check_security_compliance(self) -> Dict[str, Any]:
        """检查安全合规"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/v1/testing/validation/security-audit", timeout=15.0)
                resp.raise_for_status()
                data = resp.json()
            overall = data.get("overall_status") or data.get("status") or "unknown"
            vulns = data.get("vulnerabilities") or []
            passed = overall == "compliant" and not vulns
            return {
                'passed': passed,
                'details': f"安全状态: {overall}",
                'vulnerabilities': len(vulns),
            }
        except Exception as e:
            return {
                'passed': False,
                'details': f"安全检查失败: {e}",
                'vulnerabilities': None,
            }
        
    async def check_monitoring_active(self) -> Dict[str, Any]:
        """检查监控系统"""
        perf = await monitoring_service.performance_monitor.get_stats()
        passed = bool(perf.get("total_requests", 0))
        return {
            'passed': passed,
            'details': "监控指标正常采集" if passed else "暂无请求指标",
            'metrics_collected': passed,
            'alerts_configured': bool(self.alert_thresholds),
        }
        
    async def check_backup_configured(self) -> Dict[str, Any]:
        """检查备份配置"""
        try:
            import shutil
            import subprocess
            from urllib.parse import urlparse

            pg_dump = shutil.which("pg_dump")
            if not pg_dump:
                return {'passed': False, 'details': '未找到pg_dump，无法验证备份能力'}

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
            passed = p.returncode == 0
            return {
                'passed': passed,
                'details': "pg_dump可用" if passed else f"pg_dump失败: {p.stderr.strip()}",
            }
        except Exception as e:
            return {'passed': False, 'details': f'备份检查失败: {e}'}
        
    async def check_documentation_complete(self) -> Dict[str, Any]:
        """检查文档完整性"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/docs", timeout=10.0)
            return {
                'passed': resp.status_code == 200,
                'details': "API文档可访问" if resp.status_code == 200 else f"API文档不可用: {resp.status_code}",
                'api_docs': resp.status_code == 200,
            }
        except Exception as e:
            return {'passed': False, 'details': f'API文档检查失败: {e}', 'api_docs': False}
        
    def generate_readiness_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """生成就绪度建议"""
        recommendations = []
        
        for check_name, result in checks.items():
            if not result['passed']:
                if 'health' in check_name:
                    recommendations.append("修复不健康的组件后再部署")
                elif 'performance' in check_name:
                    recommendations.append("优化性能以满足生产要求")
                elif 'security' in check_name:
                    recommendations.append("修复安全漏洞后再部署")
                elif 'monitoring' in check_name:
                    recommendations.append("确保监控系统完全配置")
                elif 'backup' in check_name:
                    recommendations.append("配置自动备份策略")
                elif 'documentation' in check_name:
                    recommendations.append("完善系统文档")
                    
        return recommendations if recommendations else ["系统已准备好投入生产"]

# 组件监控器基类
class ComponentMonitor:
    """组件监控器基类"""
    
    async def check_health(self) -> ComponentHealth:
        """检查组件健康状态"""
        raise NotImplementedError

class LangGraphMonitor(ComponentMonitor):
    """LangGraph监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查LangGraph健康状态"""
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []
        try:
            from src.ai.langgraph.state_graph import create_simple_workflow
            from src.ai.langgraph.state import create_initial_state
            from src.ai.langgraph.context import create_default_context

            builder = create_simple_workflow()
            state = create_initial_state()
            state["input_records"] = [
                {"id": "health_1", "category": "health", "status": "ok", "value": 1},
                {"id": "health_2", "category": "health", "status": "ok", "value": 2},
            ]
            ctx = create_default_context()
            final_state = await builder.execute(state, context=ctx)
            stats = (final_state.get("context") or {}).get("processing_stats") or {}
            status = HealthStatus.HEALTHY
            metrics = {
                "nodes_registered": len(getattr(builder, "nodes", {}) or {}),
                "records_processed": int(stats.get("total_records", 0) or 0),
                "processing_time_ms": int(stats.get("processing_time_ms", 0) or 0),
            }
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            metrics = {}
            issues.append(str(e))
            recommendations.append("检查LangGraph依赖与工作流构建逻辑")

        return ComponentHealth(
            component_name='langgraph',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )

class AutoGenMonitor(ComponentMonitor):
    """AutoGen监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查AutoGen健康状态"""
        settings = get_settings()
        base_url = f"http://127.0.0.1:{settings.PORT}"
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []
        metrics: Dict[str, Any] = {}
        status = HealthStatus.HEALTHY
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/api/v1/events/stats", timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
            metrics = {
                "events_total": data.get("total"),
                "error": data.get("error"),
                "warning": data.get("warning"),
                "critical": data.get("critical"),
            }
        except Exception as e:
            status = HealthStatus.DEGRADED
            issues.append(str(e))
            recommendations.append("检查AutoGen事件系统是否初始化成功")

        return ComponentHealth(
            component_name='autogen',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )

class PgVectorMonitor(ComponentMonitor):
    """PgVector监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查PgVector健康状态"""
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []
        status = HealthStatus.HEALTHY
        metrics: Dict[str, Any] = {}
        try:
            async with get_db_session() as session:
                q0 = time.perf_counter()
                r = await session.execute(text("SELECT 1"))
                r.scalar_one()
                query_ms = (time.perf_counter() - q0) * 1000

                ext = await session.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
                ext_version = ext.scalar_one_or_none()
                if not ext_version:
                    status = HealthStatus.DEGRADED
                    issues.append("pgvector扩展未安装")
                    recommendations.append("安装并启用PostgreSQL的vector扩展")

            metrics = {
                "vector_extension_version": ext_version,
                "query_performance_ms": round(query_ms, 2),
            }
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            issues.append(str(e))
            recommendations.append("检查PostgreSQL连接与pgvector配置")

        return ComponentHealth(
            component_name='pgvector',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )

class FastAPIMonitor(ComponentMonitor):
    """FastAPI监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查FastAPI健康状态"""
        settings = get_settings()
        base_url = f"http://127.0.0.1:{settings.PORT}"
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []
        status = HealthStatus.HEALTHY
        metrics: Dict[str, Any] = {}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/api/v1/health", timeout=10.0)
                resp.raise_for_status()
                metrics["status_code"] = resp.status_code
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            issues.append(str(e))
            recommendations.append("检查API服务是否可访问")

        return ComponentHealth(
            component_name='fastapi',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )

class OpenTelemetryMonitor(ComponentMonitor):
    """OpenTelemetry监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查OpenTelemetry健康状态"""
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or ""
        service = os.environ.get("OTEL_SERVICE_NAME") or ""
        enabled = bool(endpoint) or bool(service)

        status = HealthStatus.HEALTHY
        if not enabled:
            recommendations.append("设置OTEL_EXPORTER_OTLP_ENDPOINT或OTEL_SERVICE_NAME以启用链路追踪")

        return ComponentHealth(
            component_name='opentelemetry',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics={"otel_exporter_otlp_endpoint": endpoint or None, "otel_service_name": service or None},
            issues=issues,
            recommendations=recommendations
        )

class RedisMonitor(ComponentMonitor):
    """Redis监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查Redis健康状态"""
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []
        client = get_redis()
        if not client:
            return ComponentHealth(
                component_name='redis',
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round((time.perf_counter() - started) * 1000, 2),
                last_check_time=utc_now(),
                metrics={},
                issues=["Redis未初始化"],
                recommendations=["确认Redis连接配置并在启动阶段初始化Redis"]
            )

        try:
            await client.ping()
            info = await client.info()
            hits = float(info.get("keyspace_hits", 0) or 0)
            misses = float(info.get("keyspace_misses", 0) or 0)
            hit_rate = hits / (hits + misses) * 100 if hits + misses > 0 else 0.0
            status = HealthStatus.HEALTHY
            metrics = {
                "memory_used_mb": round(float(info.get("used_memory", 0) or 0) / 1024 / 1024, 2),
                "connected_clients": int(info.get("connected_clients", 0) or 0),
                "hit_rate_percent": round(hit_rate, 2),
            }
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            metrics = {}
            issues.append(str(e))
            recommendations.append("检查Redis服务与网络连通性")

        return ComponentHealth(
            component_name='redis',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )

class PostgreSQLMonitor(ComponentMonitor):
    """PostgreSQL监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查PostgreSQL健康状态"""
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []
        status = HealthStatus.HEALTHY
        metrics: Dict[str, Any] = {}

        try:
            async with get_db_session() as session:
                q0 = time.perf_counter()
                r = await session.execute(text("SELECT 1"))
                r.scalar_one()
                query_ms = (time.perf_counter() - q0) * 1000
                size = await session.execute(text("SELECT pg_database_size(current_database())"))
                metrics["database_size_mb"] = round(float(size.scalar_one() or 0) / 1024 / 1024, 2)
                metrics["query_performance_ms"] = round(query_ms, 2)
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            issues.append(str(e))
            recommendations.append("检查数据库连接配置与PostgreSQL状态")

        return ComponentHealth(
            component_name='postgresql',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )

class QdrantMonitor(ComponentMonitor):
    """Qdrant监控器"""
    
    async def check_health(self) -> ComponentHealth:
        """检查Qdrant健康状态"""
        started = time.perf_counter()
        issues: List[str] = []
        recommendations: List[str] = []
        status = HealthStatus.HEALTHY
        metrics: Dict[str, Any] = {}

        try:
            client = qdrant_manager.get_client()
            collections = await asyncio.to_thread(client.get_collections)
            metrics["collections"] = len(getattr(collections, "collections", []) or [])
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            issues.append(str(e))
            recommendations.append("检查Qdrant服务是否启动并监听端口")

        return ComponentHealth(
            component_name='qdrant',
            status=status,
            response_time_ms=round((time.perf_counter() - started) * 1000, 2),
            last_check_time=utc_now(),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
