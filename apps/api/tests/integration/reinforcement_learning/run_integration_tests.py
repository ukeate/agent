#!/usr/bin/env python3
"""
强化学习系统集成测试自动化脚本

执行完整的集成测试套件，包括：
- 端到端功能测试
- 性能基准测试
- 负载测试
- 回归测试
"""

import os
import sys
import asyncio
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "apps" / "api" / "src"))


class IntegrationTestRunner:
    """集成测试运行器"""
    
    def __init__(self):
        self.test_results = {
            "start_time": None,
            "end_time": None,
            "total_duration": 0,
            "test_suites": {},
            "overall_status": "UNKNOWN",
            "summary": {}
        }
        self.test_dir = Path(__file__).parent
    
    def run_test_suite(self, test_file: str, markers: List[str] = None) -> Dict[str, Any]:
        """运行测试套件"""
        print(f"\n{'=' * 60}")
        print(f"运行测试套件: {test_file}")
        print(f"{'=' * 60}")
        
        # 构建pytest命令
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / test_file),
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.test_dir}/reports/{test_file}.json"
        ]
        
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # 执行测试
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(project_root),
                timeout=1800  # 30分钟超时
            )
            
            duration = time.time() - start_time
            
            # 解析结果
            test_result = {
                "file": test_file,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "PASSED" if result.returncode == 0 else "FAILED"
            }
            
            # 尝试读取JSON报告
            json_report_path = self.test_dir / "reports" / f"{test_file}.json"
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                        test_result["json_report"] = json_report
                except Exception as e:
                    print(f"警告: 无法读取JSON报告: {e}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            return {
                "file": test_file,
                "duration": time.time() - start_time,
                "return_code": -1,
                "stdout": "",
                "stderr": "测试超时",
                "status": "TIMEOUT"
            }
        except Exception as e:
            return {
                "file": test_file,
                "duration": time.time() - start_time,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "status": "ERROR"
            }
    
    def setup_test_environment(self):
        """设置测试环境"""
        print("设置测试环境...")
        
        # 创建报告目录
        reports_dir = self.test_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # 检查依赖
        try:
            import pytest
            import psutil
            print("✓ 依赖检查通过")
        except ImportError as e:
            print(f"✗ 缺少依赖: {e}")
            sys.exit(1)
        
        # 设置环境变量
        os.environ["TESTING"] = "true"
        os.environ["LOG_LEVEL"] = "WARNING"  # 减少日志输出
        
        print("✓ 测试环境设置完成")
    
    def run_all_tests(self):
        """运行所有集成测试"""
        self.test_results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        print("开始强化学习系统集成测试")
        print(f"测试开始时间: {self.test_results['start_time']}")
        
        # 设置环境
        self.setup_test_environment()
        
        # 定义测试套件
        test_suites = [
            {
                "name": "端到端功能测试",
                "file": "test_end_to_end.py",
                "markers": [],
                "required": True
            },
            {
                "name": "性能基准测试",
                "file": "test_performance.py", 
                "markers": ["performance"],
                "required": True
            }
        ]
        
        # 执行测试套件
        all_passed = True
        for suite in test_suites:
            result = self.run_test_suite(suite["file"], suite["markers"])
            self.test_results["test_suites"][suite["name"]] = result
            
            # 打印结果摘要
            status_symbol = "✓" if result["status"] == "PASSED" else "✗"
            print(f"\n{status_symbol} {suite['name']}: {result['status']} ({result['duration']:.2f}s)")
            
            if result["status"] != "PASSED":
                if suite["required"]:
                    all_passed = False
                print(f"  错误信息: {result['stderr']}")
        
        # 计算总体结果
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["total_duration"] = time.time() - start_time
        self.test_results["overall_status"] = "PASSED" if all_passed else "FAILED"
        
        # 生成摘要
        self.generate_summary()
        
        # 保存完整报告
        self.save_report()
        
        return all_passed
    
    def generate_summary(self):
        """生成测试摘要"""
        summary = {
            "total_suites": len(self.test_results["test_suites"]),
            "passed_suites": 0,
            "failed_suites": 0,
            "total_duration": self.test_results["total_duration"],
            "status_by_suite": {}
        }
        
        for suite_name, result in self.test_results["test_suites"].items():
            summary["status_by_suite"][suite_name] = result["status"]
            if result["status"] == "PASSED":
                summary["passed_suites"] += 1
            else:
                summary["failed_suites"] += 1
        
        self.test_results["summary"] = summary
    
    def save_report(self):
        """保存测试报告"""
        report_file = self.test_dir / "reports" / "integration_test_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n完整测试报告已保存: {report_file}")
    
    def print_final_summary(self):
        """打印最终摘要"""
        print(f"\n{'=' * 80}")
        print("强化学习系统集成测试完成")
        print(f"{'=' * 80}")
        
        summary = self.test_results["summary"]
        status_symbol = "✓" if self.test_results["overall_status"] == "PASSED" else "✗"
        
        print(f"整体状态: {status_symbol} {self.test_results['overall_status']}")
        print(f"总耗时: {summary['total_duration']:.2f}秒")
        print(f"测试套件: {summary['passed_suites']}/{summary['total_suites']} 通过")
        
        print(f"\n测试套件详情:")
        for suite_name, status in summary["status_by_suite"].items():
            status_symbol = "✓" if status == "PASSED" else "✗"
            duration = self.test_results["test_suites"][suite_name]["duration"]
            print(f"  {status_symbol} {suite_name}: {status} ({duration:.2f}s)")
        
        if self.test_results["overall_status"] == "FAILED":
            print(f"\n失败详情:")
            for suite_name, result in self.test_results["test_suites"].items():
                if result["status"] != "PASSED":
                    print(f"  {suite_name}:")
                    if result["stderr"]:
                        print(f"    错误: {result['stderr'][:200]}...")
        
        print(f"\n报告文件: {self.test_dir}/reports/")


def main():
    """主函数"""
    runner = IntegrationTestRunner()
    
    try:
        success = runner.run_all_tests()
        runner.print_final_summary()
        
        # 根据测试结果设置退出码
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n测试运行器发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()