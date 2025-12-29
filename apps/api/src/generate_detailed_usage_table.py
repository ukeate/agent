from src.core.utils.timezone_utils import utc_now
import json
from typing import Dict, List, Any
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
生成详细的API使用情况表格报告
"""

def load_report() -> Dict[str, Any]:
    """加载分析报告"""
    with open('api_usage_comprehensive_report.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_usage_table(report: Dict[str, Any]) -> str:
    """生成使用率统计表格"""
    module_stats = report['module_statistics']
    
    # 按使用率排序
    sorted_modules = sorted(
        module_stats.items(),
        key=lambda x: (x[1]['used'] / x[1]['total']) if x[1]['total'] > 0 else 0,
        reverse=True
    )
    
    table = []
    table.append("# API模块使用情况详细表格")
    table.append("")
    table.append(f"**生成时间**: {utc_now().strftime('%Y-%m-%d %H:%M:%S')}")
    table.append("")
    table.append("## 📊 总体统计摘要")
    table.append("")
    summary = report['summary']
    table.append("| 指标 | 数量 |")
    table.append("|------|------|")
    table.append(f"| API模块总数 | {summary['total_api_modules']} |")
    table.append(f"| API端点总数 | {summary['total_endpoints']} |")
    table.append(f"| 已使用端点 | {summary['used_endpoints']} |")
    table.append(f"| 未使用端点 | {summary['unused_endpoints']} |")
    table.append(f"| 整体使用率 | {summary['usage_rate']:.1f}% |")
    table.append(f"| 前端服务文件数 | {summary['total_frontend_services']} |")
    table.append(f"| 前端页面文件数 | {summary['total_frontend_pages']} |")
    table.append("")
    
    table.append("## 📈 各API模块使用率统计表")
    table.append("")
    table.append("| # | API模块 | 总端点数 | 已使用 | 未使用 | 使用率 | 状态 |")
    table.append("|---|---------|----------|--------|--------|--------|------|")
    
    for i, (module, stats) in enumerate(sorted_modules, 1):
        usage_rate = (stats['used'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # 状态图标
        if usage_rate >= 80:
            status = "🟢 优秀"
        elif usage_rate >= 50:
            status = "🟡 良好"
        elif usage_rate >= 20:
            status = "🟠 一般"
        else:
            status = "🔴 偏低"
        
        table.append(f"| {i} | `{module}` | {stats['total']} | {stats['used']} | {stats['unused']} | {usage_rate:.1f}% | {status} |")
    
    return "\n".join(table)

def generate_endpoint_details_table(report: Dict[str, Any]) -> str:
    """生成端点详细使用情况表格"""
    table = []
    table.append("")
    table.append("## 🔍 高使用率模块端点详情")
    table.append("")
    
    # 获取使用率较高的模块
    module_stats = report['module_statistics']
    high_usage_modules = []
    
    for module, stats in module_stats.items():
        usage_rate = (stats['used'] / stats['total'] * 100) if stats['total'] > 0 else 0
        if usage_rate >= 50:  # 使用率>=50%的模块
            high_usage_modules.append((module, usage_rate, stats))
    
    high_usage_modules.sort(key=lambda x: x[1], reverse=True)
    
    for module, usage_rate, stats in high_usage_modules[:10]:  # 只显示前10个
        table.append(f"### {module} (使用率: {usage_rate:.1f}%)")
        table.append("")
        
        # 查找该模块的端点详情
        endpoints_for_module = []
        for endpoint_key, endpoint_data in report['endpoint_details'].items():
            if endpoint_data['api_info']['file'] == module:
                is_used = "✅" if endpoint_data['is_used'] else "❌"
                frontend_usage = len(endpoint_data['frontend_usage'])
                endpoints_for_module.append({
                    'endpoint': endpoint_key,
                    'function': endpoint_data['api_info']['function'] or 'N/A',
                    'is_used': is_used,
                    'usage_count': frontend_usage
                })
        
        if endpoints_for_module:
            table.append("| 端点 | 函数名 | 使用状态 | 前端调用次数 |")
            table.append("|------|--------|----------|-------------|")
            
            for ep in endpoints_for_module:
                table.append(f"| `{ep['endpoint']}` | `{ep['function']}` | {ep['is_used']} | {ep['usage_count']} |")
            table.append("")
    
    return "\n".join(table)

def generate_unused_endpoints_table(report: Dict[str, Any]) -> str:
    """生成未使用端点表格"""
    table = []
    table.append("")
    table.append("## ❌ 未使用的API端点列表")
    table.append("")
    
    unused_endpoints = report['unused_endpoints']
    
    # 按模块分组
    unused_by_module = {}
    for endpoint in unused_endpoints:
        module = endpoint['module']
        if module not in unused_by_module:
            unused_by_module[module] = []
        unused_by_module[module].append(endpoint)
    
    # 按未使用端点数量排序
    sorted_modules = sorted(unused_by_module.items(), key=lambda x: len(x[1]), reverse=True)
    
    table.append(f"**总计未使用端点数**: {len(unused_endpoints)}")
    table.append("")
    table.append("| # | API模块 | 未使用端点数 | 未使用端点列表 |")
    table.append("|---|---------|-------------|----------------|")
    
    for i, (module, endpoints) in enumerate(sorted_modules[:15], 1):  # 只显示前15个
        endpoint_list = ", ".join([f"`{ep['endpoint']}`" for ep in endpoints[:5]])  # 每个模块只显示前5个
        if len(endpoints) > 5:
            endpoint_list += f" ... (还有{len(endpoints)-5}个)"
        
        table.append(f"| {i} | `{module}` | {len(endpoints)} | {endpoint_list} |")
    
    return "\n".join(table)

def generate_frontend_services_table(report: Dict[str, Any]) -> str:
    """生成前端服务使用情况表格"""
    table = []
    table.append("")
    table.append("## 🌐 前端服务API调用情况")
    table.append("")
    
    frontend_services = report['frontend_services']
    
    table.append("| # | 服务文件 | API调用数量 | 主要调用端点 |")
    table.append("|---|----------|-------------|-------------|")
    
    # 按API调用数量排序
    sorted_services = sorted(frontend_services.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (service, api_calls) in enumerate(sorted_services, 1):
        # 获取前3个API调用作为示例
        example_calls = [call['endpoint'] for call in api_calls[:3]]
        example_text = ", ".join([f"`{call}`" for call in example_calls])
        if len(api_calls) > 3:
            example_text += f" ... (共{len(api_calls)}个)"
        
        table.append(f"| {i} | `{service}` | {len(api_calls)} | {example_text} |")
    
    return "\n".join(table)

def generate_frontend_pages_summary(report: Dict[str, Any]) -> str:
    """生成前端页面服务使用摘要"""
    table = []
    table.append("")
    table.append("## 📱 前端页面服务使用情况")
    table.append("")
    
    pages_summary = report['frontend_pages_summary']
    
    table.append("| # | 页面文件 | 服务使用次数 |")
    table.append("|---|----------|-------------|")
    
    # 按服务使用次数排序
    sorted_pages = sorted(pages_summary.items(), key=lambda x: x[1], reverse=True)
    
    for i, (page, usage_count) in enumerate(sorted_pages[:20], 1):  # 只显示前20个
        table.append(f"| {i} | `{page}` | {usage_count} |")
    
    return "\n".join(table)

def generate_recommendations(report: Dict[str, Any]) -> str:
    """生成优化建议"""
    table = []
    table.append("")
    table.append("## 💡 优化建议")
    table.append("")
    
    summary = report['summary']
    usage_rate = summary['usage_rate']
    
    table.append("### 📊 总体分析")
    if usage_rate < 30:
        table.append("- ⚠️ **整体API使用率偏低** (< 30%)，存在较多冗余端点")
        table.append("- 🔧 建议审查未使用的API端点，考虑清理或重构")
    elif usage_rate < 60:
        table.append("- ✅ **整体API使用率中等** (30-60%)，大部分端点有价值")
        table.append("- 🔧 建议优化部分未使用端点，提升代码质量")
    else:
        table.append("- 🎉 **整体API使用率良好** (≥ 60%)，API设计合理")
        table.append("- 🔧 继续保持，可考虑进一步优化细节")
    
    table.append("")
    table.append("### 🎯 具体建议")
    
    # 找出使用率为0的模块
    zero_usage_modules = []
    for module, stats in report['module_statistics'].items():
        if stats['used'] == 0:
            zero_usage_modules.append(module)
    
    if zero_usage_modules:
        table.append(f"- 🚫 **完全未使用的模块** ({len(zero_usage_modules)}个): 考虑删除或重新设计")
        for module in zero_usage_modules[:5]:  # 只列出前5个
            table.append(f"  - `{module}`")
        if len(zero_usage_modules) > 5:
            table.append(f"  - ... 还有{len(zero_usage_modules)-5}个")
    
    # 找出高使用率模块
    high_usage_modules = []
    for module, stats in report['module_statistics'].items():
        usage_rate = (stats['used'] / stats['total'] * 100) if stats['total'] > 0 else 0
        if usage_rate >= 80:
            high_usage_modules.append((module, usage_rate))
    
    if high_usage_modules:
        table.append("")
        table.append(f"- ✨ **高价值模块** ({len(high_usage_modules)}个): 使用率≥80%，设计良好")
        for module, rate in high_usage_modules[:5]:
            table.append(f"  - `{module}` ({rate:.1f}%)")
    
    return "\n".join(table)

def main():
    logger.info("正在生成详细使用情况表格...")
    
    # 加载报告数据
    report = load_report()
    
    # 生成各个部分
    parts = []
    parts.append(generate_usage_table(report))
    parts.append(generate_endpoint_details_table(report))
    parts.append(generate_unused_endpoints_table(report))
    parts.append(generate_frontend_services_table(report))
    parts.append(generate_frontend_pages_summary(report))
    parts.append(generate_recommendations(report))
    
    # 合并所有部分
    full_report = "\n".join(parts)
    
    # 保存到文件
    filename = "API_使用情况详细分析表格.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    logger.info(f"✅ 详细表格报告已生成: {filename}")
    logger.info(f"📊 报告包含 {len(report['api_endpoints_by_module'])} 个API模块的详细分析")
    logger.info(f"🔍 分析了 {report['summary']['total_endpoints']} 个API端点")
    logger.info(f"📱 涵盖 {report['summary']['total_frontend_services']} 个前端服务和 {report['summary']['total_frontend_pages']} 个页面")

if __name__ == "__main__":
    setup_logging()
    main()
