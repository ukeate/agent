import ast
import re
import os
from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
代码内容覆盖分析工具
深度分析每个API的业务逻辑、异常处理、边界条件等代码内容实现
不只是测试端点，而是测试代码的实际执行路径和业务逻辑
"""

@dataclass
class FunctionAnalysis:
    name: str
    http_method: str
    path: str
    parameters: List[str]
    return_type: str
    exceptions_handled: List[str]
    business_logic_steps: List[str]
    dependencies: List[str]
    validation_checks: List[str]
    database_operations: List[str]
    async_operations: List[str]
    complexity_score: int
    code_coverage_gaps: List[str]

class APICodeAnalyzer:
    def __init__(self):
        self.analyzed_functions = {}
        self.coverage_gaps = []
        
    def analyze_file(self, file_path: str) -> Dict[str, FunctionAnalysis]:
        """深度分析单个API文件的代码内容"""
        if not os.path.exists(file_path):
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST获取函数结构
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"解析文件失败 {file_path}: {e}")
            return {}
        
        functions = {}
        
        # 分析每个函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis = self._analyze_function(node, content)
                if analysis:
                    functions[analysis.name] = analysis
        
        return functions
    
    def _analyze_function(self, func_node: ast.FunctionDef, content: str) -> FunctionAnalysis:
        """深度分析单个函数的业务逻辑"""
        func_name = func_node.name
        
        # 跳过非API端点函数
        if not self._is_api_endpoint(func_node, content):
            return None
        
        # 提取HTTP方法和路径
        http_method, path = self._extract_route_info(func_node, content)
        
        # 分析函数参数
        parameters = self._extract_parameters(func_node)
        
        # 分析返回类型
        return_type = self._extract_return_type(func_node)
        
        # 分析异常处理
        exceptions = self._analyze_exception_handling(func_node, content)
        
        # 分析业务逻辑步骤
        business_steps = self._analyze_business_logic(func_node, content)
        
        # 分析依赖关系
        dependencies = self._analyze_dependencies(func_node, content)
        
        # 分析验证逻辑
        validations = self._analyze_validation_checks(func_node, content)
        
        # 分析数据库操作
        db_ops = self._analyze_database_operations(func_node, content)
        
        # 分析异步操作
        async_ops = self._analyze_async_operations(func_node, content)
        
        # 计算复杂度分数
        complexity = self._calculate_complexity(func_node, content)
        
        # 识别代码覆盖缺口
        coverage_gaps = self._identify_coverage_gaps(func_node, content)
        
        return FunctionAnalysis(
            name=func_name,
            http_method=http_method,
            path=path,
            parameters=parameters,
            return_type=return_type,
            exceptions_handled=exceptions,
            business_logic_steps=business_steps,
            dependencies=dependencies,
            validation_checks=validations,
            database_operations=db_ops,
            async_operations=async_ops,
            complexity_score=complexity,
            code_coverage_gaps=coverage_gaps
        )
    
    def _is_api_endpoint(self, func_node: ast.FunctionDef, content: str) -> bool:
        """判断是否为API端点"""
        # 检查装饰器
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                if decorator.attr in ['get', 'post', 'put', 'delete', 'patch']:
                    return True
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                    return True
        return False
    
    def _extract_route_info(self, func_node: ast.FunctionDef, content: str) -> Tuple[str, str]:
        """提取HTTP方法和路径"""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                method = decorator.func.attr.upper()
                path = ""
                if decorator.args and isinstance(decorator.args[0], ast.Constant):
                    path = decorator.args[0].value
                return method, path
        return "UNKNOWN", ""
    
    def _extract_parameters(self, func_node: ast.FunctionDef) -> List[str]:
        """提取函数参数"""
        params = []
        for arg in func_node.args.args:
            if arg.arg != 'self':
                params.append(arg.arg)
        return params
    
    def _extract_return_type(self, func_node: ast.FunctionDef) -> str:
        """提取返回类型"""
        if func_node.returns:
            return ast.unparse(func_node.returns)
        return "Any"
    
    def _analyze_exception_handling(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """分析异常处理逻辑"""
        exceptions = []
        
        for node in ast.walk(func_node):
            # try-except块
            if isinstance(node, ast.ExceptHandler):
                if node.type:
                    exceptions.append(ast.unparse(node.type))
                else:
                    exceptions.append("Exception")
            
            # HTTPException
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "HTTPException":
                    exceptions.append("HTTPException")
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "HTTPException":
                    exceptions.append("HTTPException")
        
        return list(set(exceptions))
    
    def _analyze_business_logic(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """分析业务逻辑步骤"""
        steps = []
        
        for node in ast.walk(func_node):
            # 函数调用 - 业务逻辑
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # 服务调用
                    if 'service' in str(node.func.value).lower():
                        steps.append(f"调用服务: {ast.unparse(node.func)}")
                    # 数据库操作
                    elif any(db_op in ast.unparse(node.func) for db_op in ['create', 'read', 'update', 'delete', 'query']):
                        steps.append(f"数据操作: {ast.unparse(node.func)}")
                    # 外部API调用
                    elif 'client' in str(node.func.value).lower():
                        steps.append(f"外部调用: {ast.unparse(node.func)}")
            
            # 条件判断 - 业务规则
            elif isinstance(node, ast.If):
                condition = ast.unparse(node.test)[:50]
                steps.append(f"业务判断: {condition}")
            
            # 循环处理 - 批量操作
            elif isinstance(node, ast.For):
                steps.append("批量处理操作")
        
        return steps
    
    def _analyze_dependencies(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """分析依赖关系"""
        deps = []
        
        # 分析参数中的依赖注入
        for arg in func_node.args.args:
            if arg.annotation:
                annotation = ast.unparse(arg.annotation)
                if 'Depends' in annotation:
                    deps.append(f"依赖注入: {annotation}")
        
        # 分析函数内的外部依赖
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                func_name = ast.unparse(node.func)
                if any(service in func_name for service in ['service', 'repository', 'client', 'manager']):
                    deps.append(f"服务依赖: {func_name}")
        
        return list(set(deps))
    
    def _analyze_validation_checks(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """分析数据验证逻辑"""
        validations = []
        
        for node in ast.walk(func_node):
            # 条件验证
            if isinstance(node, ast.If):
                condition = ast.unparse(node.test)
                if any(keyword in condition for keyword in ['is None', 'not', 'len(', '==']):
                    validations.append(f"条件验证: {condition[:50]}")
            
            # 异常抛出
            elif isinstance(node, ast.Raise):
                if node.exc:
                    validations.append(f"验证失败: {ast.unparse(node.exc)}")
        
        return validations
    
    def _analyze_database_operations(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """分析数据库操作"""
        db_ops = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                func_name = ast.unparse(node.func)
                if any(op in func_name for op in ['create', 'get', 'update', 'delete', 'query', 'find', 'save']):
                    db_ops.append(f"数据库操作: {func_name}")
                elif 'session' in func_name.lower() or 'db' in func_name.lower():
                    db_ops.append(f"数据库会话: {func_name}")
        
        return list(set(db_ops))
    
    def _analyze_async_operations(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """分析异步操作"""
        async_ops = []
        
        # 检查是否为异步函数
        if isinstance(func_node, ast.AsyncFunctionDef):
            async_ops.append("异步函数定义")
        
        for node in ast.walk(func_node):
            # await调用
            if isinstance(node, ast.Await):
                operation = ast.unparse(node.value)
                async_ops.append(f"异步等待: {operation[:50]}")
            
            # 异步上下文管理器
            elif isinstance(node, ast.AsyncWith):
                async_ops.append("异步上下文管理")
        
        return async_ops
    
    def _calculate_complexity(self, func_node: ast.FunctionDef, content: str) -> int:
        """计算函数复杂度"""
        complexity = 1  # 基础复杂度
        
        for node in ast.walk(func_node):
            # 分支语句增加复杂度
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            # 布尔操作符增加复杂度
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _identify_coverage_gaps(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """识别代码覆盖缺口"""
        gaps = []
        
        # 检查异常处理覆盖
        has_try_except = False
        for node in ast.walk(func_node):
            if isinstance(node, ast.Try):
                has_try_except = True
                break
        
        if not has_try_except:
            gaps.append("缺少异常处理")
        
        # 检查输入验证
        has_validation = False
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                condition = ast.unparse(node.test)
                if 'not' in condition or 'is None' in condition:
                    has_validation = True
                    break
        
        if not has_validation:
            gaps.append("缺少输入验证")
        
        # 检查返回值处理
        return_statements = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return_statements.append(node)
        
        if len(return_statements) < 2:
            gaps.append("缺少多路径返回处理")
        
        return gaps

def analyze_api_modules():
    """分析所有API模块的代码内容"""
    analyzer = APICodeAnalyzer()
    
    # 获取实际存在的API文件
    import subprocess
    result = subprocess.run(['find', '.', '-name', '*.py', '-path', '*/api/v1/*'], 
                          capture_output=True, text=True)
    api_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    
    # 过滤掉测试文件，只保留API模块
    api_files = [f for f in api_files if f and not '/test' in f and f.endswith('.py')]
    api_files = api_files[:10]  # 限制数量，避免分析过多文件
    
    logger.info("🔍 深度代码内容覆盖分析")
    logger.info("=" * 60)
    
    total_functions = 0
    total_complexity = 0
    total_gaps = []
    
    for api_file in api_files:
        if os.path.exists(api_file):
            logger.info(f"\n📋 分析 {api_file}")
            logger.info("-" * 40)
            
            functions = analyzer.analyze_file(api_file)
            
            for func_name, analysis in functions.items():
                total_functions += 1
                total_complexity += analysis.complexity_score
                total_gaps.extend(analysis.code_coverage_gaps)
                
                logger.info(f"🔧 {analysis.http_method} {analysis.path} ({func_name})")
                logger.info(f"   复杂度: {analysis.complexity_score}")
                logger.error(f"   异常处理: {len(analysis.exceptions_handled)}种")
                logger.info(f"   业务步骤: {len(analysis.business_logic_steps)}个")
                logger.info(f"   依赖关系: {len(analysis.dependencies)}个")
                logger.info(f"   数据库操作: {len(analysis.database_operations)}个")
                logger.info(f"   异步操作: {len(analysis.async_operations)}个")
                
                if analysis.code_coverage_gaps:
                    logger.warning(f"   ⚠️  覆盖缺口: {', '.join(analysis.code_coverage_gaps)}")
                
                # 显示具体的业务逻辑
                if analysis.business_logic_steps:
                    logger.info(f"   📝 业务逻辑:")
                    for step in analysis.business_logic_steps[:3]:  # 显示前3个
                        logger.info(f"      • {step}")
                
                logger.info("")
        else:
            logger.warning(f"⚠️  文件不存在: {api_file}")
    
    # 汇总统计
    logger.info("\n" + "=" * 60)
    logger.info("📊 代码内容覆盖统计")
    logger.info("=" * 60)
    logger.info(f"分析函数总数: {total_functions}")
    logger.info(f"平均复杂度: {total_complexity/max(total_functions,1):.1f}")
    logger.info(f"总覆盖缺口: {len(total_gaps)}")
    
    # 统计覆盖缺口类型
    gap_types = {}
    for gap in total_gaps:
        gap_types[gap] = gap_types.get(gap, 0) + 1
    
    logger.info("\n🔍 主要覆盖缺口:")
    for gap_type, count in sorted(gap_types.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {gap_type}: {count}个函数")
    
    return {
        'total_functions': total_functions,
        'average_complexity': total_complexity/max(total_functions,1),
        'coverage_gaps': gap_types
    }

def generate_content_based_tests():
    """基于代码内容生成测试用例"""
    logger.info("\n" + "=" * 60)  
    logger.info("🧪 基于代码内容生成测试策略")
    logger.info("=" * 60)
    
    test_strategies = {
        "异常处理测试": [
            "测试各种异常情况的处理逻辑",
            "验证异常响应的状态码和消息",
            "测试异常传播和恢复机制"
        ],
        "业务逻辑测试": [
            "测试核心业务流程的正确性",
            "验证业务规则的执行逻辑", 
            "测试不同业务场景的处理"
        ],
        "边界条件测试": [
            "测试输入参数的边界值",
            "验证数据范围和格式检查",
            "测试极端情况的处理"
        ],
        "集成测试": [
            "测试外部依赖的集成",
            "验证数据库操作的正确性",
            "测试异步操作的协调"
        ],
        "性能测试": [
            "测试复杂业务逻辑的性能",
            "验证并发请求的处理能力",
            "测试资源使用的优化"
        ]
    }
    
    for strategy, items in test_strategies.items():
        logger.info(f"\n📋 {strategy}:")
        for item in items:
            logger.info(f"  • {item}")
    
    return test_strategies

if __name__ == "__main__":
    setup_logging()
    try:
        # 分析代码内容覆盖
        coverage_stats = analyze_api_modules()
        
        # 生成基于内容的测试策略
        test_strategies = generate_content_based_tests()
        
        logger.info(f"\n🎯 分析完成！发现 {coverage_stats['total_functions']} 个API函数")
        logger.info(f"平均复杂度: {coverage_stats['average_complexity']:.1f}")
        logger.info(f"需要重点关注 {len(coverage_stats['coverage_gaps'])} 类覆盖缺口")
        
    except Exception as e:
        logger.error(f"❌ 代码内容分析出错: {e}")
        import traceback
        traceback.print_exc()
