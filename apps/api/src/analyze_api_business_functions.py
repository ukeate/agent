import ast
import json
from pathlib import Path
from typing import Dict, List, Set
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
分析所有API的业务功能
"""

def extract_api_info(file_path: str) -> Dict:
    """从API文件中提取业务功能信息"""
    api_info = {
        "file": file_path,
        "module_name": Path(file_path).stem,
        "endpoints": [],
        "business_functions": [],
        "dependencies": set(),
        "has_websocket": False,
        "has_streaming": False
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查特殊功能
        if 'WebSocket' in content or 'websocket' in content:
            api_info["has_websocket"] = True
        if 'StreamingResponse' in content or 'stream' in content:
            api_info["has_streaming"] = True

        # 解析AST
        tree = ast.parse(content)

        for node in ast.walk(tree):
            # 提取路由定义
            if isinstance(node, ast.FunctionDef):
                # 检查是否是API端点
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr in ['get', 'post', 'put', 'delete', 'patch']:
                            endpoint = {
                                "name": node.name,
                                "method": decorator.attr.upper(),
                                "path": None,
                                "description": ast.get_docstring(node) or ""
                            }

                            # 尝试获取路径
                            if isinstance(decorator, ast.Call) and decorator.args:
                                if isinstance(decorator.args[0], ast.Constant):
                                    endpoint["path"] = decorator.args[0].value

                            api_info["endpoints"].append(endpoint)

                            # 从函数名和文档字符串推断业务功能
                            if endpoint["description"]:
                                api_info["business_functions"].append(endpoint["description"][:100])

            # 提取导入的依赖
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if 'services' in node.module or 'core' in node.module:
                        api_info["dependencies"].add(node.module)

    except Exception as e:
        api_info["error"] = str(e)

    api_info["dependencies"] = list(api_info["dependencies"])
    return api_info

def categorize_apis(api_infos: List[Dict]) -> Dict[str, List[str]]:
    """将API按业务功能分类"""
    categories = {
        "智能体系统": [],
        "RAG和知识管理": [],
        "实验和统计分析": [],
        "工作流和编排": [],
        "监控和指标": [],
        "ML和优化": [],
        "分布式系统": [],
        "情感智能": [],
        "平台和集成": [],
        "基础设施": [],
        "安全和认证": [],
        "数据处理": [],
        "其他": []
    }

    keyword_mapping = {
        "智能体系统": ["agent", "multi_agent", "supervisor", "async_agent"],
        "RAG和知识管理": ["rag", "knowledge", "graph", "extraction", "sparql"],
        "实验和统计分析": ["experiment", "statistic", "hypothesis", "power", "testing", "bandit"],
        "工作流和编排": ["workflow", "langgraph", "orchestrat"],
        "监控和指标": ["monitor", "metric", "alert", "report", "analytic"],
        "ML和优化": ["train", "model", "hyperparameter", "fine_tun", "compression", "evaluation"],
        "分布式系统": ["distributed", "cluster", "fault", "service_discovery"],
        "情感智能": ["emotion", "empathy", "social"],
        "平台和集成": ["platform", "mcp", "integration", "unified"],
        "基础设施": ["cache", "memory", "pgvector", "redis", "database"],
        "安全和认证": ["auth", "security", "acl", "risk"],
        "数据处理": ["batch", "stream", "event", "file", "document", "multimodal"]
    }

    for api_info in api_infos:
        module_name = api_info["module_name"]
        categorized = False

        for category, keywords in keyword_mapping.items():
            if any(keyword in module_name.lower() for keyword in keywords):
                categories[category].append(module_name)
                categorized = True
                break

        if not categorized:
            categories["其他"].append(module_name)

    return categories

def main():
    """主函数"""
    api_dir = Path(__file__).resolve().parent / "api" / "v1"
    api_files = list(api_dir.glob("*.py"))

    # 排除特定文件
    exclude_files = ["__init__.py", "__pycache__", "qlearning_tensorflow_backup.py"]
    api_files = [f for f in api_files if f.name not in exclude_files]

    logger.info("发现API文件", total=len(api_files))

    # 提取API信息
    api_infos = []
    for file_path in api_files:
        api_info = extract_api_info(str(file_path))
        api_infos.append(api_info)

    # 分类API
    categories = categorize_apis(api_infos)

    # 输出分析结果
    logger.info("API业务功能分析报告")
    logger.info("报告分隔线", line="=" * 80)

    for category, modules in categories.items():
        if modules:
            logger.info("分类统计", category=category, module_count=len(modules))
            for module in sorted(modules):
                # 找到对应的API信息
                api_info = next((a for a in api_infos if a["module_name"] == module), None)
                if api_info:
                    endpoint_count = len(api_info.get("endpoints", []))
                    special_features = []
                    if api_info.get("has_websocket"):
                        special_features.append("WebSocket")
                    if api_info.get("has_streaming"):
                        special_features.append("流处理")

                    feature_str = f" [{', '.join(special_features)}]" if special_features else ""
                    logger.info(
                        "模块端点统计",
                        module=module,
                        endpoint_count=endpoint_count,
                        features=feature_str.strip(),
                    )

    # 统计总览
    logger.info("统计总览")
    logger.info("报告分隔线", line="=" * 80)

    total_endpoints = sum(len(api.get("endpoints", [])) for api in api_infos)
    websocket_apis = sum(1 for api in api_infos if api.get("has_websocket"))
    streaming_apis = sum(1 for api in api_infos if api.get("has_streaming"))

    logger.info("总API模块数", total=len(api_infos))
    logger.info("总端点数", total=total_endpoints)
    logger.info("WebSocket API", total=websocket_apis)
    logger.info("流处理API", total=streaming_apis)

    # 保存详细报告
    with open("api_business_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "categories": categories,
            "detailed_info": [
                {
                    "module": api["module_name"],
                    "endpoints": api.get("endpoints", []),
                    "special_features": {
                        "websocket": api.get("has_websocket", False),
                        "streaming": api.get("has_streaming", False)
                    }
                }
                for api in api_infos
            ],
            "statistics": {
                "total_modules": len(api_infos),
                "total_endpoints": total_endpoints,
                "websocket_apis": websocket_apis,
                "streaming_apis": streaming_apis
            }
        }, f, indent=2, ensure_ascii=False)

    logger.info("详细报告已保存", path="api_business_analysis.json")

if __name__ == "__main__":
    setup_logging()
    main()
