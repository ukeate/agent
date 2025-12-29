import sys
import os
import argparse
import subprocess
from pathlib import Path
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
知识图谱测试运行脚本

提供便捷的测试运行和报告生成功能
"""

def run_tests(test_type="all", verbose=False, coverage=False, parallel=False):
    """运行测试"""
    
    # 基础pytest命令
    cmd = ["python", "-m", "pytest"]
    
    # 测试路径
    test_dir = Path(__file__).parent
    
    # 根据测试类型选择测试文件
    if test_type == "unit":
        test_files = [
            "test_graph_database.py",
            "test_schema.py",
            "test_incremental_updater.py", 
            "test_performance_optimizer.py"
        ]
        cmd.extend(["-m", "unit"])
    elif test_type == "api":
        test_files = ["test_knowledge_graph_api.py"]
        cmd.extend(["-m", "api"])
    elif test_type == "integration":
        test_files = [
            "test_graph_database.py",
            "test_schema.py",
            "test_incremental_updater.py"
        ]
        cmd.extend(["-m", "integration"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])
        test_files = None
    elif test_type == "fast":
        # 快速测试，排除慢速测试和需要数据库的测试
        cmd.extend(["-m", "not slow and not neo4j_integration"])
        test_files = None
    else:  # all
        test_files = None
    
    # 添加测试文件路径
    if test_files:
        for test_file in test_files:
            cmd.append(str(test_dir / test_file))
    else:
        cmd.append(str(test_dir))
    
    # 添加选项
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if coverage:
        cmd.extend([
            "--cov=ai.knowledge_graph",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    if parallel:
        cmd.extend(["-n", "auto"])  # 需要安装pytest-xdist
    
    # 添加输出格式
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--strict-config"
    ])
    
    logger.info(f"运行命令: {' '.join(cmd)}")
    logger.info(f"测试目录: {test_dir}")
    logger.info("=" * 60)
    
    # 执行测试
    try:
        result = subprocess.run(cmd, cwd=test_dir.parent.parent.parent)
        return result.returncode
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
        return 1
    except Exception as e:
        logger.error(f"运行测试时发生错误: {e}")
        return 1

def generate_report():
    """生成测试报告"""
    logger.info("生成测试报告...")
    
    test_dir = Path(__file__).parent
    report_dir = test_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # 生成HTML测试报告
    cmd = [
        "python", "-m", "pytest",
        str(test_dir),
        "--html=" + str(report_dir / "report.html"),
        "--self-contained-html",
        "--tb=short"
    ]
    
    subprocess.run(cmd, cwd=test_dir.parent.parent.parent)
    
    logger.info(f"测试报告已生成: {report_dir / 'report.html'}")

def check_dependencies():
    """检查测试依赖"""
    required_packages = [
        ("pytest", "pytest"),
        ("pytest-asyncio", "pytest_asyncio"),
        ("pytest-mock", "pytest_mock"),
        ("pytest-cov", "pytest_cov")
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.info("缺少以下测试依赖包:")
        for package in missing_packages:
            logger.info(f"  - {package}")
        logger.info("\n请运行以下命令安装:")
        logger.info(f"uv add {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识图谱测试运行脚本")
    
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "api", "integration", "performance", "fast"],
        default="all",
        nargs="?",
        help="测试类型 (默认: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="生成覆盖率报告"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="并行运行测试"
    )
    
    parser.add_argument(
        "-r", "--report",
        action="store_true",
        help="生成HTML测试报告"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="检查测试依赖"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps:
        if check_dependencies():
            logger.info("所有测试依赖都已安装")
            return 0
        else:
            return 1
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 生成报告
    if args.report:
        generate_report()
        return 0
    
    # 运行测试
    return run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )

if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
