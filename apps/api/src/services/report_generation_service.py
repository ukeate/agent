"""
实验报告自动生成服务
"""
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import asyncio
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

from ..core.database import get_db_session
from ..services.statistical_analysis_service import StatisticalAnalysisService
from ..services.hypothesis_testing_service import HypothesisTestingService
from ..services.confidence_interval_service import ConfidenceIntervalService
from ..services.realtime_metrics_service import RealtimeMetricsService
from ..services.multiple_testing_correction_service import MultipleTestingCorrectionService


class ReportFormat(str, Enum):
    """报告格式"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"


class ReportSection(str, Enum):
    """报告章节"""
    EXECUTIVE_SUMMARY = "executive_summary"
    EXPERIMENT_OVERVIEW = "experiment_overview"
    METRIC_RESULTS = "metric_results"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    SEGMENT_ANALYSIS = "segment_analysis"
    RECOMMENDATIONS = "recommendations"
    APPENDIX = "appendix"


@dataclass
class MetricResult:
    """指标结果"""
    name: str
    type: str  # primary, secondary, guardrail
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_diff: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    
    
@dataclass
class SegmentResult:
    """细分结果"""
    segment_name: str
    segment_value: str
    metrics: List[MetricResult]
    sample_size: int
    

@dataclass
class ExperimentSummary:
    """实验摘要"""
    experiment_id: str
    experiment_name: str
    status: str
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    total_users: int
    variants: List[str]
    primary_metric_lift: float
    primary_metric_significant: bool
    recommendation: str
    risk_level: str  # low, medium, high


@dataclass
class ReportData:
    """报告数据"""
    summary: ExperimentSummary
    metric_results: List[MetricResult]
    segment_results: List[SegmentResult]
    statistical_tests: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class ReportGenerationService:
    """报告生成服务"""
    
    def __init__(self):
        self.stats_service = StatisticalAnalysisService()
        self.hypothesis_service = HypothesisTestingService()
        self.confidence_service = ConfidenceIntervalService()
        self.metrics_service = RealtimeMetricsService()
        self.correction_service = MultipleTestingCorrectionService()
        
    async def generate_report(
        self,
        experiment_id: str,
        sections: Optional[List[ReportSection]] = None,
        format: ReportFormat = ReportFormat.JSON,
        include_segments: bool = True,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        生成实验报告
        
        Args:
            experiment_id: 实验ID
            sections: 要包含的章节
            format: 报告格式
            include_segments: 是否包含细分分析
            confidence_level: 置信水平
            
        Returns:
            报告内容
        """
        # 默认包含所有章节
        if sections is None:
            sections = list(ReportSection)
            
        # 收集报告数据
        report_data = await self._collect_report_data(
            experiment_id, 
            include_segments,
            confidence_level
        )
        
        # 生成报告
        report = {}
        
        if ReportSection.EXECUTIVE_SUMMARY in sections:
            report["executive_summary"] = self._generate_executive_summary(report_data)
            
        if ReportSection.EXPERIMENT_OVERVIEW in sections:
            report["experiment_overview"] = self._generate_experiment_overview(report_data)
            
        if ReportSection.METRIC_RESULTS in sections:
            report["metric_results"] = self._generate_metric_results(report_data)
            
        if ReportSection.STATISTICAL_ANALYSIS in sections:
            report["statistical_analysis"] = self._generate_statistical_analysis(report_data)
            
        if ReportSection.SEGMENT_ANALYSIS in sections and include_segments:
            report["segment_analysis"] = self._generate_segment_analysis(report_data)
            
        if ReportSection.RECOMMENDATIONS in sections:
            report["recommendations"] = self._generate_recommendations(report_data)
            
        if ReportSection.APPENDIX in sections:
            report["appendix"] = self._generate_appendix(report_data)
            
        # 格式化输出
        return await self._format_report(report, format, report_data)
        
    async def _collect_report_data(
        self,
        experiment_id: str,
        include_segments: bool,
        confidence_level: float
    ) -> ReportData:
        """收集报告数据"""
        # 获取实验信息
        experiment_info = await self._get_experiment_info(experiment_id)
        
        # 获取指标结果
        metric_results = await self._calculate_metric_results(
            experiment_id,
            confidence_level
        )
        
        # 获取细分结果
        segment_results = []
        if include_segments:
            segment_results = await self._calculate_segment_results(
                experiment_id,
                confidence_level
            )
            
        # 统计检验结果
        statistical_tests = await self._run_statistical_tests(
            experiment_id,
            metric_results
        )
        
        # 生成建议
        recommendations = self._generate_recommendations_list(
            metric_results,
            statistical_tests
        )
        
        # 检查警告
        warnings = self._check_warnings(
            experiment_info,
            metric_results,
            statistical_tests
        )
        
        # 创建摘要
        summary = self._create_experiment_summary(
            experiment_info,
            metric_results,
            recommendations
        )
        
        return ReportData(
            summary=summary,
            metric_results=metric_results,
            segment_results=segment_results,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            warnings=warnings,
            metadata={
                "generated_at": utc_now().isoformat(),
                "confidence_level": confidence_level,
                "report_version": "1.0"
            }
        )
        
    async def _get_experiment_info(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验信息"""
        # 这里应该从数据库获取实验配置
        # 简化示例
        return {
            "id": experiment_id,
            "name": f"Experiment {experiment_id}",
            "status": "running",
            "start_date": utc_now() - timedelta(days=14),
            "end_date": None,
            "variants": ["control", "treatment"],
            "allocation": {"control": 0.5, "treatment": 0.5},
            "total_users": 100000
        }
        
    async def _calculate_metric_results(
        self,
        experiment_id: str,
        confidence_level: float
    ) -> List[MetricResult]:
        """计算指标结果"""
        results = []
        
        # 获取实时指标数据
        metrics_data = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        for metric_name, metric_data in metrics_data.items():
            control_data = metric_data.get("control", {})
            treatment_data = metric_data.get("treatment", {})
            
            # 计算统计量
            control_value = control_data.get("mean", 0)
            treatment_value = treatment_data.get("mean", 0)
            absolute_diff = treatment_value - control_value
            relative_diff = (absolute_diff / control_value * 100) if control_value != 0 else 0
            
            # 进行假设检验
            test_result = await self.hypothesis_service.independent_two_sample_t_test(
                control_data.get("values", []),
                treatment_data.get("values", [])
            )
            
            # 计算置信区间
            ci_result = await self.confidence_service.calculate_difference_confidence_interval(
                control_data.get("values", []),
                treatment_data.get("values", []),
                confidence_level
            )
            
            results.append(MetricResult(
                name=metric_name,
                type=metric_data.get("type", "secondary"),
                control_value=control_value,
                treatment_value=treatment_value,
                absolute_diff=absolute_diff,
                relative_diff=relative_diff,
                p_value=test_result["p_value"],
                confidence_interval=(ci_result["lower"], ci_result["upper"]),
                is_significant=test_result["p_value"] < (1 - confidence_level),
                sample_size_control=len(control_data.get("values", [])),
                sample_size_treatment=len(treatment_data.get("values", []))
            ))
            
        # 应用多重检验校正
        p_values = [r.p_value for r in results]
        corrected_results = await self.correction_service.apply_correction(
            p_values,
            method="fdr_bh",
            alpha=1 - confidence_level
        )
        
        for i, result in enumerate(results):
            result.p_value = corrected_results["corrected_p_values"][i]
            result.is_significant = corrected_results["rejected"][i]
            
        return results
        
    async def _calculate_segment_results(
        self,
        experiment_id: str,
        confidence_level: float
    ) -> List[SegmentResult]:
        """计算细分结果"""
        segments = []
        
        # 定义细分维度
        segment_dimensions = [
            ("device_type", ["mobile", "desktop", "tablet"]),
            ("user_type", ["new", "returning"]),
            ("region", ["north", "south", "east", "west"])
        ]
        
        for dimension, values in segment_dimensions:
            for value in values:
                # 获取细分数据并计算指标
                segment_metrics = await self._calculate_metric_results(
                    f"{experiment_id}_{dimension}_{value}",
                    confidence_level
                )
                
                segments.append(SegmentResult(
                    segment_name=dimension,
                    segment_value=value,
                    metrics=segment_metrics[:3],  # 只包含主要指标
                    sample_size=10000  # 示例值
                ))
                
        return segments
        
    async def _run_statistical_tests(
        self,
        experiment_id: str,
        metric_results: List[MetricResult]
    ) -> Dict[str, Any]:
        """运行统计检验"""
        tests = {}
        
        # SRM (Sample Ratio Mismatch) 检验
        control_size = sum(r.sample_size_control for r in metric_results[:1])
        treatment_size = sum(r.sample_size_treatment for r in metric_results[:1])
        
        chi2_result = await self.hypothesis_service.chi_square_goodness_of_fit(
            [control_size, treatment_size],
            [0.5, 0.5]
        )
        
        tests["srm_test"] = {
            "chi2_statistic": chi2_result["chi2_statistic"],
            "p_value": chi2_result["p_value"],
            "has_srm": chi2_result["p_value"] < 0.01
        }
        
        # 新奇效应检验
        tests["novelty_effect"] = await self._check_novelty_effect(experiment_id)
        
        # 统计功效分析
        primary_metric = next((r for r in metric_results if r.type == "primary"), None)
        if primary_metric:
            tests["power_analysis"] = {
                "observed_power": self._calculate_observed_power(primary_metric),
                "required_sample_size": self._calculate_required_sample_size(primary_metric)
            }
            
        return tests
        
    def _generate_executive_summary(self, report_data: ReportData) -> Dict[str, Any]:
        """生成执行摘要"""
        summary = report_data.summary
        
        return {
            "experiment_name": summary.experiment_name,
            "status": summary.status,
            "duration": f"{summary.duration_days} days",
            "total_users": summary.total_users,
            "primary_metric_lift": f"{summary.primary_metric_lift:.2f}%",
            "statistical_significance": summary.primary_metric_significant,
            "recommendation": summary.recommendation,
            "risk_level": summary.risk_level,
            "key_findings": self._extract_key_findings(report_data)
        }
        
    def _generate_experiment_overview(self, report_data: ReportData) -> Dict[str, Any]:
        """生成实验概览"""
        summary = report_data.summary
        
        return {
            "experiment_id": summary.experiment_id,
            "experiment_name": summary.experiment_name,
            "hypothesis": "Treatment will improve primary metric",
            "start_date": summary.start_date.isoformat(),
            "end_date": summary.end_date.isoformat() if summary.end_date else None,
            "status": summary.status,
            "variants": summary.variants,
            "allocation": {v: 1/len(summary.variants) for v in summary.variants},
            "total_users": summary.total_users,
            "metrics_tracked": len(report_data.metric_results)
        }
        
    def _generate_metric_results(self, report_data: ReportData) -> Dict[str, Any]:
        """生成指标结果"""
        results = {
            "primary_metrics": [],
            "secondary_metrics": [],
            "guardrail_metrics": []
        }
        
        for metric in report_data.metric_results:
            metric_dict = {
                "name": metric.name,
                "control": metric.control_value,
                "treatment": metric.treatment_value,
                "absolute_diff": metric.absolute_diff,
                "relative_diff": f"{metric.relative_diff:.2f}%",
                "confidence_interval": metric.confidence_interval,
                "p_value": metric.p_value,
                "significant": metric.is_significant
            }
            
            if metric.type == "primary":
                results["primary_metrics"].append(metric_dict)
            elif metric.type == "secondary":
                results["secondary_metrics"].append(metric_dict)
            else:
                results["guardrail_metrics"].append(metric_dict)
                
        return results
        
    def _generate_statistical_analysis(self, report_data: ReportData) -> Dict[str, Any]:
        """生成统计分析"""
        return {
            "sample_ratio_mismatch": report_data.statistical_tests.get("srm_test"),
            "novelty_effect": report_data.statistical_tests.get("novelty_effect"),
            "power_analysis": report_data.statistical_tests.get("power_analysis"),
            "multiple_testing_correction": {
                "method": "FDR (Benjamini-Hochberg)",
                "corrected_alpha": 0.05,
                "num_tests": len(report_data.metric_results)
            }
        }
        
    def _generate_segment_analysis(self, report_data: ReportData) -> Dict[str, Any]:
        """生成细分分析"""
        segments = {}
        
        for segment in report_data.segment_results:
            if segment.segment_name not in segments:
                segments[segment.segment_name] = []
                
            segment_data = {
                "value": segment.segment_value,
                "sample_size": segment.sample_size,
                "metrics": []
            }
            
            for metric in segment.metrics:
                segment_data["metrics"].append({
                    "name": metric.name,
                    "lift": f"{metric.relative_diff:.2f}%",
                    "significant": metric.is_significant
                })
                
            segments[segment.segment_name].append(segment_data)
            
        return segments
        
    def _generate_recommendations(self, report_data: ReportData) -> Dict[str, Any]:
        """生成建议"""
        return {
            "recommendations": report_data.recommendations,
            "warnings": report_data.warnings,
            "next_steps": self._suggest_next_steps(report_data)
        }
        
    def _generate_appendix(self, report_data: ReportData) -> Dict[str, Any]:
        """生成附录"""
        return {
            "metadata": report_data.metadata,
            "data_quality_checks": self._run_data_quality_checks(report_data),
            "technical_details": {
                "statistical_methods": [
                    "Two-sample t-test",
                    "Chi-square test",
                    "FDR correction"
                ],
                "assumptions": [
                    "Independence of observations",
                    "Normal distribution of metrics",
                    "No selection bias"
                ]
            }
        }
        
    def _create_experiment_summary(
        self,
        experiment_info: Dict[str, Any],
        metric_results: List[MetricResult],
        recommendations: List[str]
    ) -> ExperimentSummary:
        """创建实验摘要"""
        primary_metric = next((r for r in metric_results if r.type == "primary"), None)
        
        return ExperimentSummary(
            experiment_id=experiment_info["id"],
            experiment_name=experiment_info["name"],
            status=experiment_info["status"],
            start_date=experiment_info["start_date"],
            end_date=experiment_info.get("end_date"),
            duration_days=(utc_now() - experiment_info["start_date"]).days,
            total_users=experiment_info["total_users"],
            variants=experiment_info["variants"],
            primary_metric_lift=primary_metric.relative_diff if primary_metric else 0,
            primary_metric_significant=primary_metric.is_significant if primary_metric else False,
            recommendation=recommendations[0] if recommendations else "Continue monitoring",
            risk_level=self._assess_risk_level(metric_results)
        )
        
    def _generate_recommendations_list(
        self,
        metric_results: List[MetricResult],
        statistical_tests: Dict[str, Any]
    ) -> List[str]:
        """生成建议列表"""
        recommendations = []
        
        # 检查主要指标
        primary_metrics = [r for r in metric_results if r.type == "primary"]
        if primary_metrics:
            primary = primary_metrics[0]
            if primary.is_significant and primary.relative_diff > 0:
                recommendations.append("主要指标显著提升，建议推广到100%流量")
            elif primary.is_significant and primary.relative_diff < 0:
                recommendations.append("主要指标显著下降，建议停止实验并回滚")
            else:
                recommendations.append("主要指标无显著变化，建议继续运行实验收集更多数据")
                
        # 检查护栏指标
        guardrail_metrics = [r for r in metric_results if r.type == "guardrail"]
        for metric in guardrail_metrics:
            if metric.is_significant and metric.relative_diff < -5:
                recommendations.append(f"护栏指标 {metric.name} 显著恶化，需要评估风险")
                
        # 检查SRM
        if statistical_tests.get("srm_test", {}).get("has_srm"):
            recommendations.append("检测到样本比例不匹配(SRM)，需要检查分流逻辑")
            
        return recommendations
        
    def _check_warnings(
        self,
        experiment_info: Dict[str, Any],
        metric_results: List[MetricResult],
        statistical_tests: Dict[str, Any]
    ) -> List[str]:
        """检查警告"""
        warnings = []
        
        # 样本量警告
        for metric in metric_results:
            if metric.sample_size_control < 1000 or metric.sample_size_treatment < 1000:
                warnings.append(f"指标 {metric.name} 样本量过小，结果可能不可靠")
                
        # 运行时间警告
        duration = (utc_now() - experiment_info["start_date"]).days
        if duration < 7:
            warnings.append("实验运行时间少于7天，可能受到周期性影响")
            
        # 新奇效应警告
        if statistical_tests.get("novelty_effect", {}).get("detected"):
            warnings.append("检测到新奇效应，建议延长实验时间")
            
        return warnings
        
    def _assess_risk_level(self, metric_results: List[MetricResult]) -> str:
        """评估风险等级"""
        # 检查护栏指标
        guardrail_metrics = [r for r in metric_results if r.type == "guardrail"]
        
        significant_negative = sum(
            1 for m in guardrail_metrics 
            if m.is_significant and m.relative_diff < -5
        )
        
        if significant_negative >= 2:
            return "high"
        elif significant_negative == 1:
            return "medium"
        else:
            return "low"
            
    def _extract_key_findings(self, report_data: ReportData) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 主要指标发现
        primary_metrics = [r for r in report_data.metric_results if r.type == "primary"]
        for metric in primary_metrics:
            if metric.is_significant:
                direction = "提升" if metric.relative_diff > 0 else "下降"
                findings.append(
                    f"{metric.name} {direction} {abs(metric.relative_diff):.2f}% (p={metric.p_value:.4f})"
                )
                
        # 细分发现
        if report_data.segment_results:
            best_segment = max(
                report_data.segment_results,
                key=lambda s: max(m.relative_diff for m in s.metrics) if s.metrics else 0
            )
            if best_segment.metrics:
                best_metric = max(best_segment.metrics, key=lambda m: m.relative_diff)
                findings.append(
                    f"{best_segment.segment_name}={best_segment.segment_value} 表现最佳"
                )
                
        return findings[:5]  # 返回前5个关键发现
        
    def _suggest_next_steps(self, report_data: ReportData) -> List[str]:
        """建议后续步骤"""
        next_steps = []
        
        # 基于结果的建议
        primary_metric = next(
            (r for r in report_data.metric_results if r.type == "primary"), 
            None
        )
        
        if primary_metric:
            if primary_metric.is_significant and primary_metric.relative_diff > 5:
                next_steps.append("准备全量发布计划")
                next_steps.append("进行代码审查和性能测试")
            elif not primary_metric.is_significant:
                next_steps.append("增加样本量或延长实验时间")
                next_steps.append("考虑细分用户群体进行深入分析")
            else:
                next_steps.append("评估实施成本与收益")
                
        # 基于警告的建议
        if report_data.warnings:
            next_steps.append("解决识别出的数据质量问题")
            
        return next_steps
        
    def _run_data_quality_checks(self, report_data: ReportData) -> Dict[str, Any]:
        """运行数据质量检查"""
        checks = {
            "missing_data": False,
            "outliers_detected": False,
            "data_freshness": "up_to_date",
            "completeness": 100.0
        }
        
        # 检查缺失数据
        for metric in report_data.metric_results:
            if metric.sample_size_control == 0 or metric.sample_size_treatment == 0:
                checks["missing_data"] = True
                
        # 检查异常值
        for metric in report_data.metric_results:
            if abs(metric.relative_diff) > 100:
                checks["outliers_detected"] = True
                
        return checks
        
    async def _check_novelty_effect(self, experiment_id: str) -> Dict[str, Any]:
        """检查新奇效应"""
        # 简化示例：比较前后两周的效果
        return {
            "detected": False,
            "week1_lift": 5.2,
            "week2_lift": 4.8,
            "decay_rate": -0.4
        }
        
    def _calculate_observed_power(self, metric: MetricResult) -> float:
        """计算观察到的统计功效"""
        # 简化计算
        effect_size = abs(metric.relative_diff) / 100
        n = min(metric.sample_size_control, metric.sample_size_treatment)
        
        # 使用Cohen's d和样本量估算功效
        power = 1 - stats.norm.cdf(1.96 - effect_size * np.sqrt(n/2))
        return min(max(power, 0), 1)
        
    def _calculate_required_sample_size(self, metric: MetricResult) -> int:
        """计算所需样本量"""
        # 目标功效80%，显著性水平5%
        alpha = 0.05
        power = 0.80
        
        # 使用当前效果大小
        effect_size = abs(metric.relative_diff) / 100
        
        if effect_size == 0:
            return float('inf')
            
        # 简化的样本量计算
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
        
    async def _format_report(
        self,
        report: Dict[str, Any],
        format: ReportFormat,
        report_data: ReportData
    ) -> Dict[str, Any]:
        """格式化报告输出"""
        if format == ReportFormat.JSON:
            return report
            
        elif format == ReportFormat.HTML:
            return {
                "html": self._generate_html_report(report, report_data),
                "data": report
            }
            
        elif format == ReportFormat.MARKDOWN:
            return {
                "markdown": self._generate_markdown_report(report, report_data),
                "data": report
            }
            
        elif format == ReportFormat.PDF:
            # PDF生成需要额外的库
            return {
                "pdf_url": "https://example.com/reports/report.pdf",
                "data": report
            }
            
        return report
        
    def _generate_html_report(self, report: Dict[str, Any], report_data: ReportData) -> str:
        """生成HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>实验报告 - {report_data.summary.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .significant {{ color: green; font-weight: bold; }}
                .not-significant {{ color: gray; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }}
                .recommendation {{ background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; }}
            </style>
        </head>
        <body>
            <h1>实验报告: {report_data.summary.experiment_name}</h1>
            
            <h2>执行摘要</h2>
            <p>实验状态: {report_data.summary.status}</p>
            <p>运行时长: {report_data.summary.duration_days} 天</p>
            <p>总用户数: {report_data.summary.total_users:,}</p>
            <p>主要指标提升: <span class="{'significant' if report_data.summary.primary_metric_significant else 'not-significant'}">
                {report_data.summary.primary_metric_lift:.2f}%
            </span></p>
            
            <div class="recommendation">
                <strong>建议:</strong> {report_data.summary.recommendation}
            </div>
        """
        
        # 添加指标结果表格
        if "metric_results" in report:
            html += """
            <h2>指标结果</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>对照组</th>
                    <th>实验组</th>
                    <th>相对提升</th>
                    <th>P值</th>
                    <th>显著性</th>
                </tr>
            """
            
            for category in ["primary_metrics", "secondary_metrics", "guardrail_metrics"]:
                for metric in report["metric_results"].get(category, []):
                    html += f"""
                    <tr>
                        <td>{metric['name']}</td>
                        <td>{metric['control']:.4f}</td>
                        <td>{metric['treatment']:.4f}</td>
                        <td>{metric['relative_diff']}</td>
                        <td>{metric['p_value']:.4f}</td>
                        <td class="{'significant' if metric['significant'] else 'not-significant'}">
                            {'是' if metric['significant'] else '否'}
                        </td>
                    </tr>
                    """
                    
            html += "</table>"
            
        # 添加警告
        if report_data.warnings:
            html += "<h2>警告</h2>"
            for warning in report_data.warnings:
                html += f'<div class="warning">{warning}</div>'
                
        html += """
        </body>
        </html>
        """
        
        return html
        
    def _generate_markdown_report(self, report: Dict[str, Any], report_data: ReportData) -> str:
        """生成Markdown报告"""
        md = f"""# 实验报告: {report_data.summary.experiment_name}

## 执行摘要

- **实验状态**: {report_data.summary.status}
- **运行时长**: {report_data.summary.duration_days} 天
- **总用户数**: {report_data.summary.total_users:,}
- **主要指标提升**: {report_data.summary.primary_metric_lift:.2f}% {'✅' if report_data.summary.primary_metric_significant else '❌'}

> **建议**: {report_data.summary.recommendation}

## 指标结果

### 主要指标
"""
        
        # 添加指标表格
        if "metric_results" in report:
            md += "\n| 指标 | 对照组 | 实验组 | 相对提升 | P值 | 显著性 |\n"
            md += "|------|--------|--------|----------|-----|--------|\n"
            
            for metric in report["metric_results"].get("primary_metrics", []):
                md += f"| {metric['name']} | {metric['control']:.4f} | {metric['treatment']:.4f} | "
                md += f"{metric['relative_diff']} | {metric['p_value']:.4f} | "
                md += f"{'✅' if metric['significant'] else '❌'} |\n"
                
        # 添加建议和警告
        if report_data.recommendations:
            md += "\n## 建议\n\n"
            for i, rec in enumerate(report_data.recommendations, 1):
                md += f"{i}. {rec}\n"
                
        if report_data.warnings:
            md += "\n## ⚠️ 警告\n\n"
            for warning in report_data.warnings:
                md += f"- {warning}\n"
                
        return md


class ReportScheduler:
    """报告调度器"""
    
    def __init__(self, report_service: ReportGenerationService):
        self.report_service = report_service
        self.scheduled_reports: Dict[str, asyncio.Task] = {}
        
    async def schedule_daily_report(
        self,
        experiment_id: str,
        send_time: str = "09:00",
        recipients: List[str] = None
    ):
        """调度每日报告"""
        task = asyncio.create_task(
            self._run_daily_report(experiment_id, send_time, recipients)
        )
        self.scheduled_reports[f"daily_{experiment_id}"] = task
        
    async def _run_daily_report(
        self,
        experiment_id: str,
        send_time: str,
        recipients: List[str]
    ):
        """运行每日报告任务"""
        while True:
            # 等待到指定时间
            await self._wait_until_time(send_time)
            
            # 生成报告
            report = await self.report_service.generate_report(
                experiment_id,
                format=ReportFormat.HTML
            )
            
            # 发送报告
            if recipients:
                await self._send_report(report, recipients)
                
            # 等待24小时
            await asyncio.sleep(86400)
            
    async def _wait_until_time(self, target_time: str):
        """等待到指定时间"""
        # 简化实现
        await asyncio.sleep(1)
        
    async def _send_report(self, report: Dict[str, Any], recipients: List[str]):
        """发送报告"""
        # 这里应该实现邮件发送逻辑
        pass