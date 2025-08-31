import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from pathlib import Path
import logging
from jinja2 import Template, Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import warnings

from .evaluation_engine import EvaluationResult, EvaluationMetrics
from .performance_monitor import PerformanceMonitor, BenchmarkMetrics
from .benchmark_manager import BenchmarkManager, BenchmarkInfo

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """报告配置"""
    title: str
    subtitle: Optional[str] = None
    author: str = "AI Model Evaluation System"
    include_charts: bool = True
    include_detailed_metrics: bool = True
    include_recommendations: bool = True
    chart_style: str = "seaborn"  # matplotlib, seaborn, plotly
    output_format: str = "html"  # html, pdf, json
    template_name: str = "default"
    logo_path: Optional[str] = None

@dataclass  
class ReportSection:
    """报告章节"""
    title: str
    content: str
    charts: List[str] = None
    order: int = 0

class EvaluationReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.charts_dir = Path("temp_charts")
        self.charts_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        # 初始化Jinja2环境
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir))
        )
        
        self._register_template_filters()
    
    def _register_template_filters(self):
        """注册模板过滤器"""
        @self.jinja_env.filter
        def format_number(value, precision=2):
            if isinstance(value, (int, float)):
                return f"{value:.{precision}f}"
            return str(value)
        
        @self.jinja_env.filter
        def format_percentage(value, precision=1):
            if isinstance(value, (int, float)):
                return f"{value * 100:.{precision}f}%"
            return str(value)
        
        @self.jinja_env.filter
        def format_time(value):
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            return str(value)
    
    def generate_evaluation_report(self,
                                 results: List[EvaluationResult],
                                 config: ReportConfig,
                                 benchmark_manager: Optional[BenchmarkManager] = None,
                                 performance_monitor: Optional[PerformanceMonitor] = None) -> str:
        """生成完整的评估报告"""
        
        logger.info(f"生成评估报告: {config.title}")
        
        # 数据预处理
        processed_data = self._process_evaluation_data(results)
        
        # 生成报告章节
        sections = []
        
        # 1. 执行摘要
        sections.append(self._generate_executive_summary(processed_data, config))
        
        # 2. 模型概览
        sections.append(self._generate_model_overview(processed_data))
        
        # 3. 基准测试结果  
        sections.append(self._generate_benchmark_results(processed_data, benchmark_manager))
        
        # 4. 性能分析
        if performance_monitor:
            sections.append(self._generate_performance_analysis(processed_data, performance_monitor))
        
        # 5. 详细指标
        if config.include_detailed_metrics:
            sections.append(self._generate_detailed_metrics(processed_data))
        
        # 6. 对比分析
        if len(processed_data["models"]) > 1:
            sections.append(self._generate_comparison_analysis(processed_data))
        
        # 7. 建议和结论
        if config.include_recommendations:
            sections.append(self._generate_recommendations(processed_data))
        
        # 生成图表
        chart_paths = []
        if config.include_charts:
            chart_paths = self._generate_charts(processed_data, config.chart_style)
        
        # 渲染报告
        report_content = self._render_report(sections, chart_paths, config)
        
        return report_content
    
    def _process_evaluation_data(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """处理评估数据"""
        data = {
            "total_evaluations": len(results),
            "models": {},
            "benchmarks": {},
            "overall_stats": {},
            "timestamps": {
                "start": min(r.timestamp for r in results) if results else utc_now(),
                "end": max(r.timestamp for r in results) if results else utc_now()
            }
        }
        
        # 按模型分组
        for result in results:
            model_name = result.model_name
            if model_name not in data["models"]:
                data["models"][model_name] = {
                    "results": [],
                    "avg_accuracy": 0.0,
                    "total_duration": 0.0,
                    "benchmarks_tested": set(),
                    "error_count": 0
                }
            
            data["models"][model_name]["results"].append(result)
            data["models"][model_name]["total_duration"] += result.duration
            data["models"][model_name]["benchmarks_tested"].add(result.benchmark_name)
            
            if result.error:
                data["models"][model_name]["error_count"] += 1
        
        # 按基准测试分组
        for result in results:
            benchmark_name = result.benchmark_name
            if benchmark_name not in data["benchmarks"]:
                data["benchmarks"][benchmark_name] = {
                    "results": [],
                    "models_tested": set(),
                    "avg_accuracy": 0.0,
                    "best_model": None,
                    "worst_model": None
                }
            
            data["benchmarks"][benchmark_name]["results"].append(result)
            data["benchmarks"][benchmark_name]["models_tested"].add(result.model_name)
        
        # 计算统计信息
        for model_name, model_data in data["models"].items():
            model_results = [r for r in model_data["results"] if not r.error]
            if model_results:
                accuracies = [r.metrics.accuracy for r in model_results]
                model_data["avg_accuracy"] = np.mean(accuracies)
                model_data["benchmarks_tested"] = list(model_data["benchmarks_tested"])
        
        for benchmark_name, benchmark_data in data["benchmarks"].items():
            benchmark_results = [r for r in benchmark_data["results"] if not r.error]
            if benchmark_results:
                accuracies = [r.metrics.accuracy for r in benchmark_results]
                benchmark_data["avg_accuracy"] = np.mean(accuracies)
                
                best_result = max(benchmark_results, key=lambda x: x.metrics.accuracy)
                worst_result = min(benchmark_results, key=lambda x: x.metrics.accuracy)
                benchmark_data["best_model"] = best_result.model_name
                benchmark_data["worst_model"] = worst_result.model_name
                benchmark_data["models_tested"] = list(benchmark_data["models_tested"])
        
        # 整体统计
        successful_results = [r for r in results if not r.error]
        if successful_results:
            data["overall_stats"] = {
                "success_rate": len(successful_results) / len(results),
                "avg_accuracy": np.mean([r.metrics.accuracy for r in successful_results]),
                "total_duration_hours": sum(r.duration for r in results) / 3600,
                "total_samples": sum(r.samples_evaluated for r in successful_results),
                "unique_models": len(data["models"]),
                "unique_benchmarks": len(data["benchmarks"])
            }
        
        return data
    
    def _generate_executive_summary(self, data: Dict[str, Any], config: ReportConfig) -> ReportSection:
        """生成执行摘要"""
        stats = data["overall_stats"]
        
        content = f"""
        <div class="executive-summary">
            <h2>执行摘要</h2>
            <div class="summary-grid">
                <div class="stat-card">
                    <h3>{stats.get('unique_models', 0)}</h3>
                    <p>评估模型数量</p>
                </div>
                <div class="stat-card">
                    <h3>{stats.get('unique_benchmarks', 0)}</h3>
                    <p>基准测试套件</p>
                </div>
                <div class="stat-card">
                    <h3>{stats.get('avg_accuracy', 0):.1%}</h3>
                    <p>平均准确率</p>
                </div>
                <div class="stat-card">
                    <h3>{stats.get('total_duration_hours', 0):.1f}h</h3>
                    <p>总评估时间</p>
                </div>
            </div>
            
            <div class="key-findings">
                <h3>关键发现</h3>
                <ul>
        """
        
        # 添加关键发现
        if data["models"]:
            best_model = max(data["models"].items(), key=lambda x: x[1].get("avg_accuracy", 0))
            content += f"<li>性能最佳模型: <strong>{best_model[0]}</strong> (平均准确率: {best_model[1].get('avg_accuracy', 0):.1%})</li>"
        
        if data["benchmarks"]:
            most_challenging = min(data["benchmarks"].items(), key=lambda x: x[1].get("avg_accuracy", 1))
            content += f"<li>最具挑战性的基准测试: <strong>{most_challenging[0]}</strong> (平均准确率: {most_challenging[1].get('avg_accuracy', 0):.1%})</li>"
        
        success_rate = stats.get('success_rate', 0)
        content += f"<li>评估成功率: <strong>{success_rate:.1%}</strong></li>"
        
        content += """
                </ul>
            </div>
        </div>
        """
        
        return ReportSection(title="执行摘要", content=content, order=1)
    
    def _generate_model_overview(self, data: Dict[str, Any]) -> ReportSection:
        """生成模型概览"""
        content = """
        <div class="model-overview">
            <h2>模型概览</h2>
            <table class="overview-table">
                <thead>
                    <tr>
                        <th>模型名称</th>
                        <th>平均准确率</th>
                        <th>测试基准数量</th>
                        <th>总耗时</th>
                        <th>错误次数</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for model_name, model_data in data["models"].items():
            avg_accuracy = model_data.get("avg_accuracy", 0)
            benchmark_count = len(model_data.get("benchmarks_tested", []))
            duration = model_data.get("total_duration", 0)
            error_count = model_data.get("error_count", 0)
            
            status = "成功" if error_count == 0 else f"部分失败 ({error_count})"
            status_class = "success" if error_count == 0 else "warning"
            
            content += f"""
                    <tr>
                        <td><strong>{model_name}</strong></td>
                        <td>{avg_accuracy:.1%}</td>
                        <td>{benchmark_count}</td>
                        <td>{duration:.1f}s</td>
                        <td>{error_count}</td>
                        <td><span class="status {status_class}">{status}</span></td>
                    </tr>
            """
        
        content += """
                </tbody>
            </table>
        </div>
        """
        
        return ReportSection(title="模型概览", content=content, order=2)
    
    def _generate_benchmark_results(self, data: Dict[str, Any], 
                                  benchmark_manager: Optional[BenchmarkManager]) -> ReportSection:
        """生成基准测试结果"""
        content = """
        <div class="benchmark-results">
            <h2>基准测试结果</h2>
        """
        
        for benchmark_name, benchmark_data in data["benchmarks"].items():
            avg_accuracy = benchmark_data.get("avg_accuracy", 0)
            best_model = benchmark_data.get("best_model", "N/A")
            worst_model = benchmark_data.get("worst_model", "N/A")
            
            # 获取基准测试描述
            description = "暂无描述"
            if benchmark_manager:
                benchmark_info = benchmark_manager.get_benchmark(benchmark_name)
                if benchmark_info:
                    description = benchmark_info.description
            
            content += f"""
            <div class="benchmark-section">
                <h3>{benchmark_name}</h3>
                <p class="benchmark-description">{description}</p>
                
                <div class="benchmark-stats">
                    <div class="stat-item">
                        <label>平均准确率:</label>
                        <span>{avg_accuracy:.1%}</span>
                    </div>
                    <div class="stat-item">
                        <label>最佳模型:</label>
                        <span>{best_model}</span>
                    </div>
                    <div class="stat-item">
                        <label>最差模型:</label>
                        <span>{worst_model}</span>
                    </div>
                    <div class="stat-item">
                        <label>参与模型数:</label>
                        <span>{len(benchmark_data.get('models_tested', []))}</span>
                    </div>
                </div>
                
                <div class="detailed-results">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>模型</th>
                                <th>准确率</th>
                                <th>推理时间</th>
                                <th>内存使用</th>
                                <th>样本数量</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            # 按准确率排序结果
            results = sorted(benchmark_data["results"], key=lambda x: x.metrics.accuracy, reverse=True)
            
            for result in results:
                if not result.error:
                    content += f"""
                            <tr>
                                <td>{result.model_name}</td>
                                <td>{result.metrics.accuracy:.1%}</td>
                                <td>{result.metrics.inference_time:.1f}ms</td>
                                <td>{result.metrics.memory_usage:.0f}MB</td>
                                <td>{result.samples_evaluated}</td>
                            </tr>
                    """
                else:
                    content += f"""
                            <tr class="error-row">
                                <td>{result.model_name}</td>
                                <td colspan="4" class="error">错误: {result.error}</td>
                            </tr>
                    """
            
            content += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
        
        content += "</div>"
        
        return ReportSection(title="基准测试结果", content=content, order=3)
    
    def _generate_performance_analysis(self, data: Dict[str, Any], 
                                     performance_monitor: PerformanceMonitor) -> ReportSection:
        """生成性能分析"""
        content = """
        <div class="performance-analysis">
            <h2>性能分析</h2>
        """
        
        # 系统资源使用情况
        system_summary = performance_monitor.get_system_metrics_summary(60)
        if "error" not in system_summary:
            content += f"""
            <div class="system-metrics">
                <h3>系统资源使用</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>CPU使用率</h4>
                        <div class="metric-value">{system_summary['cpu']['avg']:.1f}%</div>
                        <div class="metric-range">范围: {system_summary['cpu']['min']:.1f}% - {system_summary['cpu']['max']:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <h4>内存使用</h4>
                        <div class="metric-value">{system_summary['memory']['avg_used_gb']:.1f}GB</div>
                        <div class="metric-range">峰值: {system_summary['memory']['max_used_gb']:.1f}GB</div>
                    </div>
            """
            
            if system_summary.get("gpu"):
                gpu_summary = system_summary["gpu"]
                content += f"""
                    <div class="metric-card">
                        <h4>GPU使用率</h4>
                        <div class="metric-value">{gpu_summary['utilization']['avg']:.1f}%</div>
                        <div class="metric-range">峰值: {gpu_summary['utilization']['max']:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <h4>GPU内存</h4>
                        <div class="metric-value">{gpu_summary['memory']['avg_used_gb']:.1f}GB</div>
                        <div class="metric-range">峰值: {gpu_summary['memory']['max_used_gb']:.1f}GB</div>
                    </div>
                """
            
            content += """
                </div>
            </div>
            """
        
        # 模型性能对比
        content += """
        <div class="model-performance">
            <h3>模型性能对比</h3>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>模型</th>
                        <th>平均准确率</th>
                        <th>平均推理时间</th>
                        <th>吞吐量</th>
                        <th>内存效率</th>
                        <th>综合评分</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # 计算每个模型的综合性能评分
        for model_name, model_data in data["models"].items():
            results = [r for r in model_data["results"] if not r.error]
            if results:
                avg_accuracy = np.mean([r.metrics.accuracy for r in results])
                avg_inference_time = np.mean([r.metrics.inference_time for r in results if r.metrics.inference_time > 0])
                avg_throughput = np.mean([r.metrics.throughput for r in results if r.metrics.throughput > 0])
                avg_memory = np.mean([r.metrics.memory_usage for r in results if r.metrics.memory_usage > 0])
                
                # 计算综合评分
                performance_score = self._calculate_performance_score(avg_accuracy, avg_inference_time, avg_memory)
                
                content += f"""
                        <tr>
                            <td><strong>{model_name}</strong></td>
                            <td>{avg_accuracy:.1%}</td>
                            <td>{avg_inference_time:.0f}ms</td>
                            <td>{avg_throughput:.1f}/s</td>
                            <td>{avg_memory:.0f}MB</td>
                            <td><span class="score-badge">{performance_score:.0f}</span></td>
                        </tr>
                """
        
        content += """
                </tbody>
            </table>
        </div>
        """
        
        # 性能告警
        active_alerts = performance_monitor.get_active_alerts()
        if active_alerts:
            content += """
            <div class="performance-alerts">
                <h3>性能告警</h3>
                <div class="alerts-list">
            """
            
            for alert in active_alerts[:10]:  # 显示前10个告警
                severity_class = f"alert-{alert.severity}"
                content += f"""
                    <div class="alert-item {severity_class}">
                        <div class="alert-header">
                            <span class="alert-severity">{alert.severity.upper()}</span>
                            <span class="alert-time">{alert.timestamp.strftime('%H:%M:%S')}</span>
                        </div>
                        <div class="alert-title">{alert.title}</div>
                        <div class="alert-description">{alert.description}</div>
                    </div>
                """
            
            content += """
                </div>
            </div>
            """
        
        content += "</div>"
        
        return ReportSection(title="性能分析", content=content, order=4)
    
    def _calculate_performance_score(self, accuracy: float, inference_time: float, memory_usage: float) -> float:
        """计算性能综合评分"""
        # 归一化分数 (0-100)
        accuracy_score = accuracy * 100  # 直接使用准确率百分比
        
        # 推理时间分数 (越快越好, 假设1000ms以下为满分)
        time_score = max(0, 100 - (inference_time / 10))
        
        # 内存使用分数 (越少越好, 假设1GB以下为满分)
        memory_score = max(0, 100 - (memory_usage / 10))
        
        # 加权平均 (准确率50%, 速度30%, 内存20%)
        overall_score = accuracy_score * 0.5 + time_score * 0.3 + memory_score * 0.2
        return min(100, max(0, overall_score))
    
    def _generate_detailed_metrics(self, data: Dict[str, Any]) -> ReportSection:
        """生成详细指标"""
        content = """
        <div class="detailed-metrics">
            <h2>详细指标</h2>
        """
        
        for model_name, model_data in data["models"].items():
            content += f"""
            <div class="model-details">
                <h3>{model_name}</h3>
                <div class="metrics-tabs">
            """
            
            results = [r for r in model_data["results"] if not r.error]
            
            for i, result in enumerate(results):
                tab_active = "active" if i == 0 else ""
                content_active = "active" if i == 0 else ""
                
                content += f"""
                    <div class="tab-header {tab_active}" onclick="showTab('{model_name}_{i}')">
                        {result.benchmark_name}
                    </div>
                """
            
            content += "<div class='tab-contents'>"
            
            for i, result in enumerate(results):
                content_active = "active" if i == 0 else ""
                
                content += f"""
                    <div class="tab-content {content_active}" id="{model_name}_{i}">
                        <div class="metric-details">
                            <div class="metric-row">
                                <span class="metric-label">准确率:</span>
                                <span class="metric-value">{result.metrics.accuracy:.3f}</span>
                            </div>
                """
                
                if result.metrics.f1_score is not None:
                    content += f"""
                            <div class="metric-row">
                                <span class="metric-label">F1分数:</span>
                                <span class="metric-value">{result.metrics.f1_score:.3f}</span>
                            </div>
                    """
                
                if result.metrics.bleu_score is not None:
                    content += f"""
                            <div class="metric-row">
                                <span class="metric-label">BLEU分数:</span>
                                <span class="metric-value">{result.metrics.bleu_score:.3f}</span>
                            </div>
                    """
                
                if result.metrics.rouge_scores:
                    for rouge_type, rouge_score in result.metrics.rouge_scores.items():
                        content += f"""
                            <div class="metric-row">
                                <span class="metric-label">{rouge_type.upper()}:</span>
                                <span class="metric-value">{rouge_score:.3f}</span>
                            </div>
                        """
                
                if result.metrics.perplexity is not None:
                    content += f"""
                            <div class="metric-row">
                                <span class="metric-label">困惑度:</span>
                                <span class="metric-value">{result.metrics.perplexity:.2f}</span>
                            </div>
                    """
                
                content += f"""
                            <div class="metric-row">
                                <span class="metric-label">推理时间:</span>
                                <span class="metric-value">{result.metrics.inference_time:.1f}ms</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">内存使用:</span>
                                <span class="metric-value">{result.metrics.memory_usage:.0f}MB</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">吞吐量:</span>
                                <span class="metric-value">{result.metrics.throughput:.1f}样本/秒</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">评估时间:</span>
                                <span class="metric-value">{result.duration:.1f}秒</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">样本数量:</span>
                                <span class="metric-value">{result.samples_evaluated}</span>
                            </div>
                        </div>
                    </div>
                """
            
            content += """
                </div>
            </div>
            </div>
            """
        
        content += "</div>"
        
        return ReportSection(title="详细指标", content=content, order=5)
    
    def _generate_comparison_analysis(self, data: Dict[str, Any]) -> ReportSection:
        """生成对比分析"""
        content = """
        <div class="comparison-analysis">
            <h2>模型对比分析</h2>
        """
        
        # 创建对比表格
        models = list(data["models"].keys())
        benchmarks = list(data["benchmarks"].keys())
        
        content += """
        <div class="comparison-matrix">
            <h3>准确率对比矩阵</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>基准测试</th>
        """
        
        for model in models:
            content += f"<th>{model}</th>"
        
        content += """
                    </tr>
                </thead>
                <tbody>
        """
        
        for benchmark_name in benchmarks:
            content += f"<tr><td><strong>{benchmark_name}</strong></td>"
            
            benchmark_results = {}
            for result in data["benchmarks"][benchmark_name]["results"]:
                if not result.error:
                    benchmark_results[result.model_name] = result.metrics.accuracy
            
            for model in models:
                if model in benchmark_results:
                    accuracy = benchmark_results[model]
                    # 根据性能添加颜色编码
                    color_class = self._get_performance_color_class(accuracy)
                    content += f'<td class="{color_class}">{accuracy:.1%}</td>'
                else:
                    content += '<td class="no-data">N/A</td>'
            
            content += "</tr>"
        
        content += """
                </tbody>
            </table>
        </div>
        """
        
        # 添加排名分析
        content += """
        <div class="ranking-analysis">
            <h3>模型排名分析</h3>
            <div class="rankings-grid">
        """
        
        # 按平均准确率排名
        model_rankings = sorted(data["models"].items(), 
                              key=lambda x: x[1].get("avg_accuracy", 0), reverse=True)
        
        content += """
                <div class="ranking-section">
                    <h4>综合准确率排名</h4>
                    <ol class="ranking-list">
        """
        
        for i, (model_name, model_data) in enumerate(model_rankings):
            avg_accuracy = model_data.get("avg_accuracy", 0)
            medal_class = ["gold", "silver", "bronze"][i] if i < 3 else ""
            
            content += f"""
                        <li class="ranking-item {medal_class}">
                            <span class="rank">#{i+1}</span>
                            <span class="model-name">{model_name}</span>
                            <span class="accuracy">{avg_accuracy:.1%}</span>
                        </li>
            """
        
        content += """
                    </ol>
                </div>
        """
        
        # 按速度排名 (如果有推理时间数据)
        speed_data = {}
        for model_name, model_data in data["models"].items():
            results = [r for r in model_data["results"] if not r.error and r.metrics.inference_time > 0]
            if results:
                avg_time = np.mean([r.metrics.inference_time for r in results])
                speed_data[model_name] = avg_time
        
        if speed_data:
            speed_rankings = sorted(speed_data.items(), key=lambda x: x[1])  # 时间越短越好
            
            content += """
                <div class="ranking-section">
                    <h4>推理速度排名</h4>
                    <ol class="ranking-list">
            """
            
            for i, (model_name, avg_time) in enumerate(speed_rankings):
                medal_class = ["gold", "silver", "bronze"][i] if i < 3 else ""
                
                content += f"""
                            <li class="ranking-item {medal_class}">
                                <span class="rank">#{i+1}</span>
                                <span class="model-name">{model_name}</span>
                                <span class="speed">{avg_time:.0f}ms</span>
                            </li>
                """
            
            content += """
                    </ol>
                </div>
            """
        
        content += """
            </div>
        </div>
        </div>
        """
        
        return ReportSection(title="对比分析", content=content, order=6)
    
    def _get_performance_color_class(self, accuracy: float) -> str:
        """根据性能获取颜色类别"""
        if accuracy >= 0.9:
            return "performance-excellent"
        elif accuracy >= 0.8:
            return "performance-good"
        elif accuracy >= 0.7:
            return "performance-fair"
        else:
            return "performance-poor"
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> ReportSection:
        """生成建议和结论"""
        recommendations = []
        
        # 分析最佳模型
        if data["models"]:
            best_model = max(data["models"].items(), key=lambda x: x[1].get("avg_accuracy", 0))
            recommendations.append(
                f"推荐模型: **{best_model[0]}** 在综合评估中表现最佳，平均准确率达到 {best_model[1].get('avg_accuracy', 0):.1%}"
            )
        
        # 分析性能瓶颈
        slow_models = []
        for model_name, model_data in data["models"].items():
            results = [r for r in model_data["results"] if not r.error]
            if results:
                avg_time = np.mean([r.metrics.inference_time for r in results if r.metrics.inference_time > 0])
                if avg_time > 2000:  # 超过2秒
                    slow_models.append((model_name, avg_time))
        
        if slow_models:
            slow_models_str = ", ".join([f"{name}({time:.0f}ms)" for name, time in slow_models])
            recommendations.append(
                f"性能优化建议: 以下模型推理时间较长，建议进行优化: {slow_models_str}"
            )
        
        # 分析基准测试结果
        if data["benchmarks"]:
            difficult_benchmarks = []
            for benchmark_name, benchmark_data in data["benchmarks"].items():
                avg_accuracy = benchmark_data.get("avg_accuracy", 0)
                if avg_accuracy < 0.6:
                    difficult_benchmarks.append((benchmark_name, avg_accuracy))
            
            if difficult_benchmarks:
                difficult_str = ", ".join([f"{name}({acc:.1%})" for name, acc in difficult_benchmarks])
                recommendations.append(
                    f"挑战性任务: 以下基准测试对所有模型都较为困难，可能需要特殊优化: {difficult_str}"
                )
        
        # 资源使用建议
        high_memory_models = []
        for model_name, model_data in data["models"].items():
            results = [r for r in model_data["results"] if not r.error]
            if results:
                avg_memory = np.mean([r.metrics.memory_usage for r in results if r.metrics.memory_usage > 0])
                if avg_memory > 8000:  # 超过8GB
                    high_memory_models.append((model_name, avg_memory))
        
        if high_memory_models:
            memory_str = ", ".join([f"{name}({mem:.0f}MB)" for name, mem in high_memory_models])
            recommendations.append(
                f"内存优化建议: 以下模型内存使用较高，建议考虑模型压缩或量化: {memory_str}"
            )
        
        # 默认建议
        if not recommendations:
            recommendations.append("所有模型都表现良好，继续保持当前的评估和优化策略。")
        
        content = f"""
        <div class="recommendations">
            <h2>建议和结论</h2>
            
            <div class="recommendations-list">
                <h3>关键建议</h3>
                <ul>
        """
        
        for recommendation in recommendations:
            content += f"<li>{recommendation}</li>"
        
        content += """
                </ul>
            </div>
            
            <div class="conclusions">
                <h3>总结</h3>
                <p>本次评估测试了 {unique_models} 个模型在 {unique_benchmarks} 个基准测试上的性能。
                整体成功率为 {success_rate:.1%}，平均准确率达到 {avg_accuracy:.1%}。
                评估总耗时 {total_hours:.1f} 小时，共处理 {total_samples} 个样本。</p>
                
                <p>建议根据具体应用场景选择合适的模型，并持续监控模型性能以确保最佳表现。</p>
            </div>
        </div>
        """.format(
            unique_models=data["overall_stats"].get("unique_models", 0),
            unique_benchmarks=data["overall_stats"].get("unique_benchmarks", 0),
            success_rate=data["overall_stats"].get("success_rate", 0),
            avg_accuracy=data["overall_stats"].get("avg_accuracy", 0),
            total_hours=data["overall_stats"].get("total_duration_hours", 0),
            total_samples=data["overall_stats"].get("total_samples", 0)
        )
        
        return ReportSection(title="建议和结论", content=content, order=7)
    
    def _generate_charts(self, data: Dict[str, Any], chart_style: str) -> List[str]:
        """生成图表"""
        chart_paths = []
        
        try:
            # 1. 模型准确率对比图
            if len(data["models"]) > 1:
                chart_path = self._create_model_accuracy_chart(data, chart_style)
                if chart_path:
                    chart_paths.append(chart_path)
            
            # 2. 基准测试结果热力图
            if len(data["models"]) > 1 and len(data["benchmarks"]) > 1:
                chart_path = self._create_heatmap_chart(data, chart_style)
                if chart_path:
                    chart_paths.append(chart_path)
            
            # 3. 性能vs准确率散点图
            chart_path = self._create_performance_scatter_chart(data, chart_style)
            if chart_path:
                chart_paths.append(chart_path)
            
        except Exception as e:
            logger.error(f"生成图表失败: {e}")
        
        return chart_paths
    
    def _create_model_accuracy_chart(self, data: Dict[str, Any], style: str) -> Optional[str]:
        """创建模型准确率对比图"""
        models = []
        accuracies = []
        
        for model_name, model_data in data["models"].items():
            if model_data.get("avg_accuracy", 0) > 0:
                models.append(model_name)
                accuracies.append(model_data["avg_accuracy"])
        
        if not models:
            return None
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        
        plt.title('模型准确率对比', fontsize=16, fontweight='bold')
        plt.xlabel('模型名称', fontsize=12)
        plt.ylabel('平均准确率', fontsize=12)
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = self.charts_dir / "model_accuracy_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_heatmap_chart(self, data: Dict[str, Any], style: str) -> Optional[str]:
        """创建基准测试结果热力图"""
        models = list(data["models"].keys())
        benchmarks = list(data["benchmarks"].keys())
        
        # 创建结果矩阵
        matrix = np.zeros((len(benchmarks), len(models)))
        
        for i, benchmark_name in enumerate(benchmarks):
            benchmark_results = {}
            for result in data["benchmarks"][benchmark_name]["results"]:
                if not result.error:
                    benchmark_results[result.model_name] = result.metrics.accuracy
            
            for j, model_name in enumerate(models):
                matrix[i, j] = benchmark_results.get(model_name, 0)
        
        plt.figure(figsize=(max(8, len(models) * 1.5), max(6, len(benchmarks) * 0.8)))
        sns.heatmap(matrix, 
                   xticklabels=models,
                   yticklabels=benchmarks,
                   annot=True,
                   fmt='.1%',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1,
                   cbar_kws={'label': '准确率'})
        
        plt.title('基准测试结果热力图', fontsize=16, fontweight='bold')
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('基准测试', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        chart_path = self.charts_dir / "benchmark_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_performance_scatter_chart(self, data: Dict[str, Any], style: str) -> Optional[str]:
        """创建性能vs准确率散点图"""
        models = []
        accuracies = []
        inference_times = []
        memory_usage = []
        
        for model_name, model_data in data["models"].items():
            results = [r for r in model_data["results"] if not r.error]
            if results:
                avg_accuracy = model_data.get("avg_accuracy", 0)
                avg_time = np.mean([r.metrics.inference_time for r in results if r.metrics.inference_time > 0])
                avg_memory = np.mean([r.metrics.memory_usage for r in results if r.metrics.memory_usage > 0])
                
                models.append(model_name)
                accuracies.append(avg_accuracy)
                inference_times.append(avg_time)
                memory_usage.append(avg_memory)
        
        if len(models) < 2:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率 vs 推理时间
        scatter1 = ax1.scatter(inference_times, accuracies, 
                              s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        ax1.set_xlabel('平均推理时间 (ms)', fontsize=12)
        ax1.set_ylabel('平均准确率', fontsize=12)
        ax1.set_title('准确率 vs 推理时间', fontsize=14, fontweight='bold')
        
        for i, model in enumerate(models):
            ax1.annotate(model, (inference_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 准确率 vs 内存使用
        scatter2 = ax2.scatter(memory_usage, accuracies, 
                              s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        ax2.set_xlabel('平均内存使用 (MB)', fontsize=12)
        ax2.set_ylabel('平均准确率', fontsize=12)
        ax2.set_title('准确率 vs 内存使用', fontsize=14, fontweight='bold')
        
        for i, model in enumerate(models):
            ax2.annotate(model, (memory_usage[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        chart_path = self.charts_dir / "performance_scatter.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _render_report(self, sections: List[ReportSection], 
                      chart_paths: List[str], config: ReportConfig) -> str:
        """渲染最终报告"""
        
        # 按顺序排序章节
        sections.sort(key=lambda x: x.order)
        
        # 准备模板数据
        template_data = {
            "title": config.title,
            "subtitle": config.subtitle,
            "author": config.author,
            "generated_time": utc_now(),
            "sections": sections,
            "charts": chart_paths,
            "include_charts": config.include_charts,
            "logo_path": config.logo_path
        }
        
        # 加载并渲染模板
        try:
            template = self.jinja_env.get_template(f"{config.template_name}.html")
            html_content = template.render(**template_data)
            return html_content
        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
            # 返回简单的HTML格式
            return self._render_simple_html(sections, config)
    
    def _render_simple_html(self, sections: List[ReportSection], config: ReportConfig) -> str:
        """渲染简单HTML格式"""
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{config.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .stat-card {{ display: inline-block; margin: 10px; padding: 20px; background: #f4f4f4; border-radius: 8px; }}
                .status.success {{ color: green; }}
                .status.warning {{ color: orange; }}
                .performance-excellent {{ background-color: #4CAF50; color: white; }}
                .performance-good {{ background-color: #8BC34A; }}
                .performance-fair {{ background-color: #FF9800; }}
                .performance-poor {{ background-color: #F44336; color: white; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .error-row {{ background-color: #ffebee; }}
                .ranking-item.gold {{ background-color: #FFD700; }}
                .ranking-item.silver {{ background-color: #C0C0C0; }}
                .ranking-item.bronze {{ background-color: #CD7F32; }}
            </style>
        </head>
        <body>
            <h1>{config.title}</h1>
            <p><em>生成时间: {utc_now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            <p><em>作者: {config.author}</em></p>
        """
        
        for section in sections:
            html += section.content
        
        html += """
        <script>
            function showTab(tabId) {
                // 隐藏所有tab内容
                var tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(function(content) {
                    content.classList.remove('active');
                });
                
                // 显示选中的tab内容
                document.getElementById(tabId).classList.add('active');
                
                // 更新tab头样式
                var tabHeaders = document.querySelectorAll('.tab-header');
                tabHeaders.forEach(function(header) {
                    header.classList.remove('active');
                });
                event.target.classList.add('active');
            }
        </script>
        </body>
        </html>
        """
        
        return html
    
    def save_report(self, content: str, output_path: str, format_type: str = "html"):
        """保存报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == "html":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # 其他格式的处理可以在这里扩展
            raise ValueError(f"不支持的格式: {format_type}")
        
        logger.info(f"报告已保存到: {output_path}")
        return str(output_file)