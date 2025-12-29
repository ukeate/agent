import json
import re
from pathlib import Path
from typing import Dict, List, Set
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
分析前端页面与后端API的覆盖关系
"""

def analyze_service_files(web_dir: Path) -> Dict[str, Set[str]]:
    """分析前端服务文件，找出每个服务调用的API端点"""
    services_dir = web_dir / "src" / "services"
    service_apis = {}

    if not services_dir.exists():
        return service_apis

    for service_file in services_dir.glob("*.ts"):
        if service_file.name == "apiClient.ts":
            continue

        service_name = service_file.stem
        apis = set()

        try:
            with open(service_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # 查找API调用
                # 匹配 apiClient.get/post/put/delete 等调用
                api_pattern = r'apiClient\.(get|post|put|delete|patch)\s*[<\(]\s*[^>]*[>\(]?\s*\(\s*[\'"`]([^\'"`]+)[\'"`]'
                matches = re.finditer(api_pattern, content)
                for match in matches:
                    api_path = match.group(2)
                    # 清理路径中的变量
                    api_path = re.sub(r'\$\{[^}]+\}', '{var}', api_path)
                    apis.add(api_path)

                # 也查找fetch调用
                fetch_pattern = r'fetch\s*\(\s*[\'"`]([^\'"`]+)[\'"`]'
                matches = re.finditer(fetch_pattern, content)
                for match in matches:
                    api_path = match.group(1)
                    if api_path.startswith('/api'):
                        apis.add(api_path)

        except Exception:
            logger.exception("错误分析服务文件", path=str(service_file))

        if apis:
            service_apis[service_name] = apis

    return service_apis

def analyze_page_services(pages_dir: Path) -> Dict[str, Set[str]]:
    """分析页面文件，找出每个页面使用的服务"""
    page_services = {}

    for page_file in pages_dir.glob("*.tsx"):
        page_name = page_file.stem
        services = set()

        try:
            with open(page_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # 查找import的服务
                import_pattern = r'import\s+.*?from\s+[\'"`].*?/services/([^\'"`]+)[\'"`]'
                matches = re.finditer(import_pattern, content)
                for match in matches:
                    service = match.group(1)
                    # 移除文件扩展名
                    service = service.replace('.ts', '').replace('.js', '')
                    services.add(service)

                # 查找服务调用
                service_pattern = r'(\w+Service)\.'
                matches = re.finditer(service_pattern, content)
                for match in matches:
                    service_name = match.group(1).replace('Service', '')
                    services.add(service_name)

        except Exception:
            logger.exception("错误分析页面文件", path=str(page_file))

        if services:
            page_services[page_name] = services

    return page_services

def map_api_to_module(api_path: str) -> str:
    """将API路径映射到对应的后端模块"""
    # 清理路径
    api_path = api_path.replace('/api/v1/', '')

    # 提取模块名
    parts = api_path.split('/')
    if parts:
        # 处理特殊路径映射
        module_map = {
            "multi-agent": "multi_agents",
            "batch": "batch",
            "fine-tuning": "fine_tuning",
            "model-registry": "model_registry",
            "model-service": "model_service",
            "model-evaluation": "model_evaluation",
            "model-compression": "model_compression",
            "knowledge-graph": "knowledge_graph",
            "knowledge-extraction": "knowledge_extraction",
            "knowledge-management": "knowledge_management",
            "graph-rag": "graphrag",
            "multi-step-reasoning": "multi_step_reasoning",
            "hypothesis-testing": "hypothesis_testing",
            "power-analysis": "power_analysis",
            "statistical-analysis": "statistical_analysis",
            "multiple-testing-correction": "multiple_testing_correction",
            "anomaly-detection": "anomaly_detection",
            "realtime-metrics": "realtime_metrics",
            "alert-rules": "alert_rules",
            "report-generation": "report_generation",
            "distributed-task": "distributed_task",
            "distributed-security": "distributed_security",
            "service-discovery": "service_discovery",
            "cluster-management": "cluster_management",
            "fault-tolerance": "fault_tolerance",
            "traffic-ramp": "traffic_ramp",
            "auto-scaling": "auto_scaling",
            "release-strategy": "release_strategy",
            "risk-assessment": "risk_assessment",
            "emotional-intelligence": "emotional_intelligence",
            "emotion-recognition": "emotion_recognition",
            "emotion-modeling": "emotion_modeling",
            "emotional-memory": "emotional_memory",
            "empathy-response": "empathy_response",
            "social-emotion": "social_emotion_api",
            "social-emotional": "social_emotional_understanding",
            "platform-integration": "platform_integration",
            "agent-interface": "agent_interface",
            "async-agents": "async_agents",
            "langgraph": "langgraph_features",
            "hyperparameter-optimization": "hyperparameter_optimization",
            "training-data": "training_data",
            "event-tracking": "event_tracking",
            "event-batch": "event_batch",
            "targeting-rules": "targeting_rules",
            "assignment-cache": "assignment_cache",
            "layered-experiments": "layered_experiments",
            "bandit-recommendations": "bandit_recommendations",
            "multimodal-rag": "multimodal_rag",
            "explainable-ai": "explainable_ai",
            "memory-management": "memory_management",
            "memory-analytics": "memory_analytics"
        }

        module = parts[0]
        return module_map.get(module, module.replace('-', '_'))

    return ""

def main():
    """主分析函数"""
    repo_root = Path(__file__).resolve().parents[3]
    web_dir = repo_root / "apps" / "web"
    pages_dir = web_dir / "src" / "pages"

    # 加载之前生成的API分析报告
    with open("api_business_analysis_detailed.json", "r", encoding="utf-8") as f:
        api_report = json.load(f)

    # 分析前端服务和页面
    service_apis = analyze_service_files(web_dir)
    page_services = analyze_page_services(pages_dir)

    logger.info("前端页面与后端API覆盖分析报告")
    logger.info("报告分隔线", line="=" * 80)

    # 统计所有后端API模块
    all_api_modules = {api_info["module"] for api_info in api_report["details"]}
    covered_api_modules = set()
    page_api_mapping = {}

    # 分析每个页面覆盖的API
    for page_name, services in page_services.items():
        page_apis = set()

        for service in services:
            if service in service_apis:
                for api_path in service_apis[service]:
                    api_module = map_api_to_module(api_path)
                    if api_module:
                        page_apis.add(api_module)
                        covered_api_modules.add(api_module)

        if page_apis:
            page_api_mapping[page_name] = list(page_apis)

    # 找出未覆盖的API模块
    uncovered_api_modules = all_api_modules - covered_api_modules

    # 按业务领域分类显示覆盖情况
    business_categories = {
        "智能体系统": ["multi_agents", "agents", "supervisor", "agent_interface", "async_agents"],
        "RAG系统": ["rag", "knowledge_graph", "graphrag", "knowledge_extraction", "knowledge_graph_reasoning", "knowledge_management", "multimodal_rag"],
        "实验平台": ["experiments", "hypothesis_testing", "power_analysis", "statistical_analysis", "multiple_testing_correction", "anomaly_detection"],
        "工作流系统": ["workflows", "langgraph_features"],
        "监控系统": ["realtime_metrics", "alert_rules", "report_generation", "analytics", "memory_analytics"],
        "ML平台": ["model_registry", "fine_tuning", "hyperparameter_optimization", "model_compression", "model_evaluation", "model_service", "training_data"],
        "分布式系统": ["distributed_task", "cluster_management", "service_discovery", "distributed_security", "fault_tolerance"],
        "数据处理": ["batch", "streaming", "files", "documents", "multimodal", "events", "event_tracking", "event_batch"],
        "安全系统": ["auth", "security", "acl", "risk_assessment"],
        "情感智能": ["emotional_intelligence", "emotion_recognition", "emotion_modeling", "emotional_memory", "empathy_response", "social_emotion_api", "social_emotional_understanding", "emotion_intelligence", "emotion_websocket"],
        "平台功能": ["platform_integration", "mcp", "unified", "offline", "cache", "pgvector", "memory_management"]
    }

    logger.info("按业务领域的API覆盖情况")

    for category, modules in business_categories.items():
        covered = [m for m in modules if m in covered_api_modules]
        uncovered = [m for m in modules if m in uncovered_api_modules]
        total = len(modules)
        covered_count = len(covered)
        coverage_rate = (covered_count / total * 100) if total > 0 else 0

        logger.info(
            "覆盖率",
            category=category,
            coverage_rate=round(coverage_rate, 1),
            covered=covered_count,
            total=total,
        )

        if uncovered:
            logger.warning("未覆盖的API", category=category, modules=", ".join(uncovered))

        logger.info("分类分隔", category=category)

    # 找出需要创建页面的API
    logger.info("需要创建前端页面的API模块")

    priority_apis = {
        "高优先级（核心功能）": [
            "experiments", "workflows", "rag", "agents"
        ],
        "中优先级（重要功能）": [
            "streaming", "events", "cache", "pgvector",
            "explainable_ai", "reasoning", "unified"
        ],
        "低优先级（辅助功能）": [
            "test", "health", "core_init"
        ]
    }

    for priority, apis in priority_apis.items():
        uncovered_priority = [api for api in apis if api in uncovered_api_modules]
        if uncovered_priority:
            logger.info("优先级", priority=priority)
            for api in uncovered_priority:
                # 找到API的详细信息
                api_info = next((a for a in api_report["details"] if a["module"] == api), None)
                if api_info:
                    endpoint_count = api_info.get("endpoint_count", 0)
                    logger.info("未覆盖模块", module=api, endpoint_count=endpoint_count)
            logger.info("优先级分隔", priority=priority)

    # 统计总览
    logger.info("统计总览")
    logger.info("报告分隔线", line="=" * 80)

    logger.info("总API模块数", total=len(all_api_modules))
    logger.info("已覆盖模块数", total=len(covered_api_modules))
    logger.info("未覆盖模块数", total=len(uncovered_api_modules))
    logger.info(
        "总体覆盖率",
        percent=round(len(covered_api_modules) / len(all_api_modules) * 100, 1),
    )
    logger.info("总页面数", total=len(page_api_mapping))

    # 保存详细报告
    coverage_report = {
        "summary": {
            "total_api_modules": len(all_api_modules),
            "covered_modules": len(covered_api_modules),
            "uncovered_modules": len(uncovered_api_modules),
            "coverage_rate": len(covered_api_modules) / len(all_api_modules) * 100,
            "total_pages": len(page_api_mapping)
        },
        "covered_apis": list(covered_api_modules),
        "uncovered_apis": list(uncovered_api_modules),
        "page_api_mapping": page_api_mapping,
        "service_apis": {k: list(v) for k, v in service_apis.items()},
        "recommendations": {
            "high_priority": [api for api in priority_apis["高优先级（核心功能）"] if api in uncovered_api_modules],
            "medium_priority": [api for api in priority_apis["中优先级（重要功能）"] if api in uncovered_api_modules],
            "low_priority": [api for api in priority_apis["低优先级（辅助功能）"] if api in uncovered_api_modules]
        }
    }

    with open("frontend_api_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2, ensure_ascii=False)

    logger.info("详细报告已保存", path="frontend_api_coverage_report.json")

if __name__ == "__main__":
    setup_logging()
    main()
