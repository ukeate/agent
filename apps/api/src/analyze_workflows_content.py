import ast
import re
from pathlib import Path
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
针对workflows.py进行深度代码内容分析
分析每个API端点的业务逻辑实现，找出测试覆盖缺口
"""

def analyze_workflows_api():
    """深度分析workflows.py的代码逻辑和业务实现"""
    logger.info("深度分析 workflows.py API代码内容")
    logger.info("报告分隔线", line="=" * 60)
    
    # 读取workflows.py文件
    try:
        workflows_path = Path(__file__).resolve().parent / "api" / "v1" / "workflows.py"
        with open(workflows_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error("workflows.py 文件未找到")
        return
    
    # 解析AST
    tree = ast.parse(content)
    
    # 分析每个API端点函数
    api_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 检查是否有router装饰器
            for decorator in node.decorator_list:
                if hasattr(decorator, 'func') and hasattr(decorator.func, 'attr'):
                    if decorator.func.attr in ['get', 'post', 'put', 'delete']:
                        api_functions.append(node)
                        break
    
    logger.info("发现API端点函数", total=len(api_functions))
    logger.info("分析分隔")
    
    # 逐个分析API函数的代码逻辑
    for i, func in enumerate(api_functions, 1):
        analyze_single_api_function(func, content, i)
    
    # 分析WebSocket处理逻辑
    analyze_websocket_logic(content)
    
    # 分析ConnectionManager类
    analyze_connection_manager(content)
    
    generate_workflow_test_recommendations()

def analyze_single_api_function(func_node, content, index):
    """分析单个API函数的详细逻辑"""
    func_name = func_node.name
    
    # 提取HTTP方法和路径
    http_method = "UNKNOWN"
    path = ""
    
    for decorator in func_node.decorator_list:
        if hasattr(decorator, 'func') and hasattr(decorator.func, 'attr'):
            http_method = decorator.func.attr.upper()
            if decorator.args and hasattr(decorator.args[0], 'value'):
                path = decorator.args[0].value
            break
    
    logger.info("API端点", index=index, method=http_method, path=path, name=func_name)
    logger.info("分隔线", line="-" * 50)
    
    # 分析函数复杂度
    complexity = calculate_function_complexity(func_node)
    logger.info("复杂度评分", score=complexity)
    
    # 分析异常处理
    exceptions = analyze_exception_handling(func_node)
    logger.warning("异常处理统计", total=len(exceptions), exceptions=exceptions)
    
    # 分析业务逻辑步骤
    business_steps = analyze_business_logic_steps(func_node, content)
    logger.info("业务逻辑步骤")
    for step in business_steps:
        logger.info("业务逻辑步骤项", step=step)
    
    # 分析数据验证
    validations = analyze_data_validations(func_node)
    logger.info("数据验证统计", total=len(validations))
    for validation in validations:
        logger.info("数据验证项", validation=validation)
    
    # 分析依赖关系
    dependencies = analyze_function_dependencies(func_node)
    logger.info("依赖关系", dependencies=dependencies)
    
    # 识别测试覆盖缺口
    coverage_gaps = identify_test_coverage_gaps(func_node, business_steps, exceptions)
    if coverage_gaps:
        logger.warning("测试覆盖缺口")
        for gap in coverage_gaps:
            logger.warning("覆盖缺口项", gap=gap)
    
    logger.info("分析分隔")

def calculate_function_complexity(func_node):
    """计算函数的循环复杂度"""
    complexity = 1
    for node in ast.walk(func_node):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    return complexity

def analyze_exception_handling(func_node):
    """分析异常处理逻辑"""
    exceptions = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.ExceptHandler):
            if node.type:
                exceptions.append(ast.unparse(node.type))
        elif isinstance(node, ast.Call):
            if hasattr(node.func, 'id') and node.func.id == 'HTTPException':
                exceptions.append('HTTPException')
    return list(set(exceptions))

def analyze_business_logic_steps(func_node, content):
    """分析业务逻辑执行步骤"""
    steps = []
    
    # 提取函数源码进行字符串分析
    func_lines = content.split('\n')[func_node.lineno-1:func_node.end_lineno]
    func_content = '\n'.join(func_lines)
    
    # 分析具体的业务操作
    if 'workflow_service' in func_content:
        # 找到所有workflow_service调用
        service_calls = re.findall(r'workflow_service\.(\w+)', func_content)
        for call in service_calls:
            steps.append(f"调用工作流服务: {call}")
    
    # 分析条件判断逻辑
    if_patterns = re.findall(r'if\s+([^:]+):', func_content)
    for pattern in if_patterns[:3]:  # 限制显示数量
        steps.append(f"条件判断: {pattern.strip()}")
    
    # 分析数据处理逻辑
    if 'execute_data' in func_content:
        steps.append("处理执行数据参数")
    if 'input_data' in func_content:
        steps.append("处理输入数据")
    if 'control_data' in func_content:
        steps.append("处理控制指令")
    
    # 分析响应构建
    if 'return' in func_content:
        return_patterns = re.findall(r'return\s+({[^}]+}|\w+)', func_content)
        for pattern in return_patterns[:2]:
            steps.append(f"构建响应: {pattern[:30]}...")
    
    return steps

def analyze_data_validations(func_node):
    """分析数据验证逻辑"""
    validations = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            if 'is None' in condition or 'not' in condition or '==' in condition:
                validations.append(f"条件验证: {condition[:50]}")
        elif isinstance(node, ast.Raise):
            if node.exc:
                validations.append(f"抛出异常: {ast.unparse(node.exc)[:50]}")
    return validations

def analyze_function_dependencies(func_node):
    """分析函数依赖"""
    dependencies = []
    
    # 分析参数依赖
    for arg in func_node.args.args:
        if arg.annotation:
            annotation = ast.unparse(arg.annotation)
            if 'Path' in annotation:
                dependencies.append(f"路径参数: {arg.arg}")
            elif 'Query' in annotation:
                dependencies.append(f"查询参数: {arg.arg}")
            elif any(dep in annotation for dep in ['Request', 'Create', 'Update', 'Control']):
                dependencies.append(f"请求模型: {annotation}")
    
    return dependencies

def identify_test_coverage_gaps(func_node, business_steps, exceptions):
    """识别测试覆盖缺口"""
    gaps = []
    
    # 检查异常情况测试
    if len(exceptions) == 0:
        gaps.append("缺少异常处理测试")
    
    # 检查业务逻辑分支测试
    if len(business_steps) > 3:
        gaps.append("复杂业务逻辑需要多场景测试")
    
    # 检查边界条件
    has_parameter_validation = any('条件判断' in step for step in business_steps)
    if not has_parameter_validation:
        gaps.append("缺少参数边界值测试")
    
    # 检查异步操作
    func_content = ast.unparse(func_node)
    if 'await' in func_content:
        gaps.append("异步操作需要超时和错误测试")
    
    return gaps

def analyze_websocket_logic(content):
    """分析WebSocket逻辑"""
    logger.info("WebSocket逻辑分析")
    logger.info("分隔线", line="-" * 40)
    
    # 找到WebSocket相关代码
    websocket_lines = [line for line in content.split('\n') if 'websocket' in line.lower()]
    
    logger.info("WebSocket端点", path="/ws/{workflow_id}")
    logger.info("功能特性")
    logger.info("功能项", item="实时状态推送")
    logger.info("功能项", item="初始状态发送")
    logger.info("功能项", item="客户端消息处理")
    logger.info("功能项", item="连接管理")
    
    logger.warning("测试覆盖缺口")
    logger.warning("覆盖缺口项", item="WebSocket连接建立/断开测试")
    logger.warning("覆盖缺口项", item="消息发送/接收测试")
    logger.warning("覆盖缺口项", item="异常断线重连测试")
    logger.warning("覆盖缺口项", item="并发连接管理测试")
    logger.info("分析分隔")

def analyze_connection_manager(content):
    """分析ConnectionManager类"""
    logger.info("ConnectionManager类分析")
    logger.info("分隔线", line="-" * 40)
    
    logger.info("核心功能")
    logger.info("功能项", item="维护活跃WebSocket连接字典")
    logger.info("功能项", item="处理连接建立和断开")
    logger.info("功能项", item="单播和广播消息发送")
    logger.info("功能项", item="异常连接清理")
    
    logger.warning("测试覆盖缺口")
    logger.warning("覆盖缺口项", item="连接字典并发访问测试")
    logger.warning("覆盖缺口项", item="大量连接内存管理测试")
    logger.warning("覆盖缺口项", item="异常连接自动清理测试")
    logger.warning("覆盖缺口项", item="消息发送失败处理测试")
    logger.info("分析分隔")

def generate_workflow_test_recommendations():
    """生成工作流API测试建议"""
    logger.info("工作流API测试建议")
    logger.info("报告分隔线", line="=" * 60)
    
    test_cases = {
        "基础功能测试": [
            "创建工作流 - 正常数据",
            "获取工作流列表 - 分页和过滤",  
            "获取工作流详情 - 存在和不存在ID",
            "启动工作流 - 有无输入数据",
            "控制工作流 - 暂停/恢复/取消",
            "删除工作流 - 软删除验证"
        ],
        "异常处理测试": [
            "无效工作流ID - 404错误",
            "服务不可用 - 500错误处理", 
            "参数验证失败 - 400错误",
            "数据库连接异常 - 降级处理",
            "并发操作冲突 - 状态一致性"
        ],
        "业务逻辑测试": [
            "工作流状态转换 - 状态机验证",
            "检查点创建和恢复 - 数据完整性",
            "工作流执行流程 - 步骤顺序",
            "依赖关系处理 - 服务调用",
            "数据传递 - 输入输出映射"
        ],
        "性能测试": [
            "大量工作流列表查询 - 响应时间",
            "复杂工作流执行 - 资源使用",
            "WebSocket连接数 - 并发限制", 
            "数据库查询优化 - SQL性能",
            "内存泄漏 - 长时间运行"
        ],
        "集成测试": [
            "与workflow_service集成 - 接口契约",
            "数据库事务 - ACID特性",
            "WebSocket实时通信 - 消息时序",
            "健康检查 - 服务状态监控",
            "日志记录 - 审计跟踪"
        ]
    }
    
    for category, cases in test_cases.items():
        logger.info("测试建议分类", category=category)
        for case in cases:
            logger.info("测试建议项", case=case)
    
    logger.info(
        "测试建议统计",
        total_cases=sum(len(cases) for cases in test_cases.values()),
    )

if __name__ == "__main__":
    setup_logging()
    analyze_workflows_api()
