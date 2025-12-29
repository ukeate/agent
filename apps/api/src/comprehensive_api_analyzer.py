import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
全面的API业务功能分析器
"""

def extract_api_endpoints(file_path: str) -> List[Dict]:
    """提取API文件中的所有端点信息"""
    endpoints = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找所有路由定义
        router_pattern = r'@router\.(get|post|put|delete|patch|websocket)\s*\(\s*["\'](.*?)["\']\s*(?:.*?)\)'
        matches = re.finditer(router_pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            method = match.group(1).upper()
            path = match.group(2)

            # 获取函数定义
            func_pattern = rf'@router\.{match.group(1)}.*?\n+async\s+def\s+(\w+)\s*\('
            func_match = re.search(func_pattern, content[match.start():match.end()+500], re.DOTALL)

            func_name = func_match.group(1) if func_match else "unknown"

            # 提取函数描述（从summary或docstring）
            summary_pattern = r'summary\s*=\s*["\'](.*?)["\']'
            summary_match = re.search(summary_pattern, content[match.start():match.end()+100])
            summary = summary_match.group(1) if summary_match else ""

            # 获取response_model
            response_model_pattern = r'response_model\s*=\s*(\w+)'
            response_model_match = re.search(response_model_pattern, content[match.start():match.end()+100])
            response_model = response_model_match.group(1) if response_model_match else None

            endpoints.append({
                "method": method,
                "path": path,
                "function_name": func_name,
                "summary": summary,
                "response_model": response_model
            })

    except Exception as e:
        logger.error(f"错误处理文件 {file_path}: {e}")

    return endpoints

def categorize_api_by_business(module_name: str, endpoints: List[Dict]) -> Dict:
    """根据业务功能对API进行分类"""

    # 业务功能映射
    business_mapping = {
        # 智能体系统
        "multi_agents": {
            "category": "智能体系统",
            "description": "多智能体协作管理",
            "features": ["智能体协作", "对话管理", "工作流编排"]
        },
        "agents": {
            "category": "智能体系统",
            "description": "单智能体管理",
            "features": ["ReAct智能体", "工具调用", "任务执行"]
        },
        "supervisor": {
            "category": "智能体系统",
            "description": "监督者模式",
            "features": ["任务分配", "智能体调度", "执行监控"]
        },
        "agent_interface": {
            "category": "智能体系统",
            "description": "智能体接口管理",
            "features": ["接口定义", "协议管理", "通信规范"]
        },

        # RAG系统
        "rag": {
            "category": "RAG系统",
            "description": "检索增强生成",
            "features": ["语义检索", "文档问答", "知识库管理"]
        },
        "knowledge_graph": {
            "category": "RAG系统",
            "description": "知识图谱管理",
            "features": ["图谱构建", "实体关系", "图谱查询"]
        },
        "graphrag": {
            "category": "RAG系统",
            "description": "图谱增强RAG",
            "features": ["图谱检索", "知识推理", "混合检索"]
        },

        # 实验平台
        "experiments": {
            "category": "实验平台",
            "description": "A/B测试实验",
            "features": ["实验配置", "流量分配", "效果分析"]
        },
        "hypothesis_testing": {
            "category": "实验平台",
            "description": "统计假设检验",
            "features": ["T检验", "卡方检验", "A/B测试分析"]
        },
        "power_analysis": {
            "category": "实验平台",
            "description": "统计功效分析",
            "features": ["样本量计算", "效应量估计", "功效计算"]
        },

        # 工作流
        "workflows": {
            "category": "工作流系统",
            "description": "工作流编排",
            "features": ["流程定义", "状态管理", "执行监控"]
        },
        "langgraph_features": {
            "category": "工作流系统",
            "description": "LangGraph功能",
            "features": ["状态机", "图编排", "条件分支"]
        },

        # 监控和运维
        "realtime_metrics": {
            "category": "监控系统",
            "description": "实时指标监控",
            "features": ["性能指标", "业务指标", "实时告警"]
        },
        "alert_rules": {
            "category": "监控系统",
            "description": "告警规则管理",
            "features": ["规则配置", "阈值设置", "通知管理"]
        },

        # ML/AI功能
        "model_registry": {
            "category": "ML平台",
            "description": "模型注册中心",
            "features": ["模型管理", "版本控制", "部署管理"]
        },
        "fine_tuning": {
            "category": "ML平台",
            "description": "模型微调",
            "features": ["数据准备", "训练配置", "评估验证"]
        },
        "hyperparameter_optimization": {
            "category": "ML平台",
            "description": "超参数优化",
            "features": ["网格搜索", "贝叶斯优化", "自动调参"]
        },

        # 分布式系统
        "distributed_task": {
            "category": "分布式系统",
            "description": "分布式任务管理",
            "features": ["任务调度", "负载均衡", "故障恢复"]
        },
        "cluster_management": {
            "category": "分布式系统",
            "description": "集群管理",
            "features": ["节点管理", "资源分配", "健康检查"]
        },
        "service_discovery": {
            "category": "分布式系统",
            "description": "服务发现",
            "features": ["服务注册", "负载均衡", "健康监测"]
        },

        # 数据处理
        "batch": {
            "category": "数据处理",
            "description": "批处理系统",
            "features": ["批量任务", "作业调度", "进度监控"]
        },
        "streaming": {
            "category": "数据处理",
            "description": "流式处理",
            "features": ["实时流", "事件处理", "流式计算"]
        },
        "files": {
            "category": "数据处理",
            "description": "文件管理",
            "features": ["文件上传", "存储管理", "批量处理"]
        },

        # 安全和认证
        "auth": {
            "category": "安全系统",
            "description": "认证授权",
            "features": ["用户认证", "JWT令牌", "权限管理"]
        },
        "security": {
            "category": "安全系统",
            "description": "安全管理",
            "features": ["安全策略", "威胁检测", "审计日志"]
        },
        "distributed_security": {
            "category": "安全系统",
            "description": "分布式安全",
            "features": ["加密通信", "访问控制", "安全事件"]
        }
    }

    # 获取业务信息
    business_info = business_mapping.get(module_name, {
        "category": "其他",
        "description": module_name,
        "features": []
    })

    return {
        "module": module_name,
        "category": business_info["category"],
        "description": business_info["description"],
        "features": business_info["features"],
        "endpoints": endpoints,
        "endpoint_count": len(endpoints),
        "methods": list(set(e["method"] for e in endpoints))
    }

def analyze_api_coverage(api_infos: List[Dict], frontend_pages: List[str]) -> Dict:
    """分析API和前端页面的覆盖情况"""

    # 前端页面到API的映射
    page_api_mapping = {
        "MultiAgentChatContainer": ["multi_agents"],
        "AgentInterfacePage": ["agent_interface", "agents"],
        "WorkflowPage": ["workflows"],
        "GraphRAGPage": ["graphrag", "knowledge_graph"],
        "ExperimentDashboardPage": ["experiments"],
        "HypothesisTestingPage": ["hypothesis_testing"],
        "PowerAnalysisPage": ["power_analysis"],
        "ModelRegistryPage": ["model_registry"],
        "FineTuningJobsPage": ["fine_tuning"],
        "HyperparameterOptimizationPage": ["hyperparameter_optimization"],
        "BatchOperationsPage": ["batch"],
        "FileManagementPage": ["files"],
        "DistributedTaskMonitorPage": ["distributed_task"],
        "ServiceDiscoveryManagementPage": ["service_discovery"],
        "SecurityPage": ["security", "distributed_security"],
        "AuthManagementPage": ["auth"],
        "MonitoringDashboardPage": ["realtime_metrics", "alert_rules"]
    }

    covered_apis = set()
    uncovered_apis = set()

    for api_info in api_infos:
        module = api_info["module"]
        is_covered = False

        for page, apis in page_api_mapping.items():
            if module in apis:
                is_covered = True
                covered_apis.add(module)
                break

        if not is_covered:
            uncovered_apis.add(module)

    return {
        "total_apis": len(api_infos),
        "covered_apis": list(covered_apis),
        "uncovered_apis": list(uncovered_apis),
        "coverage_rate": len(covered_apis) / len(api_infos) * 100 if api_infos else 0
    }

def main():
    """主分析函数"""
    api_dir = Path("/Users/runout/awork/code/my_git/agent/apps/api/src/api/v1")
    api_files = list(api_dir.glob("*.py"))

    # 排除特定文件
    exclude_files = ["__init__.py", "qlearning_tensorflow_backup.py", "acl.py"]
    api_files = [f for f in api_files if f.name not in exclude_files]

    logger.info(f"\n{'='*80}")
    logger.info("API 业务功能详细分析报告")
    logger.info(f"{'='*80}\n")

    # 收集所有API信息
    all_api_infos = []
    categories = {}

    for file_path in api_files:
        endpoints = extract_api_endpoints(str(file_path))
        module_name = file_path.stem
        api_info = categorize_api_by_business(module_name, endpoints)
        all_api_infos.append(api_info)

        # 按类别分组
        category = api_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(api_info)

    # 输出分析结果
    for category, apis in sorted(categories.items()):
        logger.info(f"\n### {category}")
        logger.info("-" * 40)

        for api in sorted(apis, key=lambda x: x["module"]):
            if api["endpoint_count"] > 0:
                logger.info(f"\n📦 **{api['module']}** - {api['description']}")
                logger.info(f"   端点数: {api['endpoint_count']}")
                logger.info(f"   HTTP方法: {', '.join(api['methods'])}")

                if api['features']:
                    logger.info(f"   核心功能: {', '.join(api['features'])}")

                # 显示前5个端点
                logger.info("   主要端点:")
                for i, endpoint in enumerate(api['endpoints'][:5], 1):
                    summary = f" - {endpoint['summary']}" if endpoint['summary'] else ""
                    logger.info(f"     {i}. {endpoint['method']} {endpoint['path']}{summary}")

                if len(api['endpoints']) > 5:
                    logger.info(f"     ... 还有 {len(api['endpoints']) - 5} 个端点")

    # 统计总览
    logger.info(f"\n{'='*80}")
    logger.info("统计总览")
    logger.info(f"{'='*80}")

    total_modules = len(all_api_infos)
    total_endpoints = sum(api['endpoint_count'] for api in all_api_infos)
    modules_with_endpoints = sum(1 for api in all_api_infos if api['endpoint_count'] > 0)

    logger.info(f"\n总模块数: {total_modules}")
    logger.info(f"有端点的模块数: {modules_with_endpoints}")
    logger.info(f"总端点数: {total_endpoints}")

    # 按类别统计
    logger.info(f"\n按业务领域统计:")
    for category, apis in sorted(categories.items()):
        cat_endpoints = sum(api['endpoint_count'] for api in apis)
        if cat_endpoints > 0:
            logger.info(f"  {category}: {len(apis)}个模块, {cat_endpoints}个端点")

    # 保存详细报告
    report = {
        "summary": {
            "total_modules": total_modules,
            "modules_with_endpoints": modules_with_endpoints,
            "total_endpoints": total_endpoints,
            "categories": {
                cat: {
                    "module_count": len(apis),
                    "endpoint_count": sum(api['endpoint_count'] for api in apis)
                }
                for cat, apis in categories.items()
            }
        },
        "details": all_api_infos
    }

    with open("api_business_analysis_detailed.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\n详细报告已保存到: api_business_analysis_detailed.json")

if __name__ == "__main__":
    setup_logging()
    main()
