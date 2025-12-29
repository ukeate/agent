"""
自动报告生成器

生成行为分析的综合报告，包括趋势分析、异常检测和模式识别结果。
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass
import json
from ..models import BehaviorPattern, AnomalyDetection
from ..storage.event_store import EventStore
from .pattern_recognition import PatternRecognitionEngine
from .anomaly_detection import AnomalyDetectionEngine
from ..models.forecasting import TrendAnalyzer

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: Dict[str, Any]
    charts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]

@dataclass
class AnalyticsReport:
    """分析报告"""
    report_id: str
    title: str
    generated_at: datetime
    period: Dict[str, datetime]
    executive_summary: str
    sections: List[ReportSection]
    appendix: Dict[str, Any]
    metadata: Dict[str, Any]

class ReportGenerator:
    """报告生成器"""
    
    def __init__(
        self,
        event_store: EventStore,
        pattern_engine: PatternRecognitionEngine,
        anomaly_engine: AnomalyDetectionEngine,
        trend_analyzer: TrendAnalyzer
    ):
        self.event_store = event_store
        self.pattern_engine = pattern_engine
        self.anomaly_engine = anomaly_engine
        self.trend_analyzer = trend_analyzer
    
    async def generate_comprehensive_report(
        self,
        time_range_days: int = 30,
        user_ids: Optional[List[str]] = None,
        include_forecasts: bool = True
    ) -> AnalyticsReport:
        """生成综合分析报告"""
        report_id = f"report_{int(utc_now().timestamp())}"
        
        try:
            # 计算时间范围
            end_time = utc_now()
            start_time = end_time - timedelta(days=time_range_days)
            period = {'start_time': start_time, 'end_time': end_time}
            
            # 并发获取各种分析结果
            tasks = [
                self._generate_overview_section(period, user_ids),
                self._generate_pattern_analysis_section(period, user_ids),
                self._generate_anomaly_detection_section(period, user_ids),
                self._generate_trend_analysis_section(period, user_ids, include_forecasts),
                self._generate_user_behavior_section(period, user_ids)
            ]
            
            sections = await asyncio.gather(*tasks)
            
            # 生成执行摘要
            executive_summary = await self._generate_executive_summary(sections, period)
            
            # 生成附录
            appendix = await self._generate_appendix(period, user_ids)
            
            report = AnalyticsReport(
                report_id=report_id,
                title=f"行为分析报告 - {start_time.strftime('%Y-%m-%d')} 至 {end_time.strftime('%Y-%m-%d')}",
                generated_at=utc_now(),
                period=period,
                executive_summary=executive_summary,
                sections=sections,
                appendix=appendix,
                metadata={
                    'time_range_days': time_range_days,
                    'user_count': len(user_ids) if user_ids else 'all',
                    'include_forecasts': include_forecasts
                }
            )
            
            return report
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            raise
    
    async def _generate_overview_section(
        self,
        period: Dict[str, datetime],
        user_ids: Optional[List[str]]
    ) -> ReportSection:
        """生成概览章节"""
        try:
            # 获取基础统计数据
            stats = await self.event_store.get_event_statistics(
                start_time=period['start_time'],
                end_time=period['end_time']
            )
            
            insights = []
            recommendations = []
            charts = []
            
            # 分析总体活跃度
            total_events = stats.get('total_events', 0)
            unique_users = stats.get('unique_users', 0)
            unique_sessions = stats.get('unique_sessions', 0)
            
            if total_events > 0:
                avg_events_per_user = total_events / max(unique_users, 1)
                avg_events_per_session = total_events / max(unique_sessions, 1)
                
                insights.append(f"分析期间共记录{total_events:,}个用户行为事件")
                insights.append(f"涉及{unique_users:,}个独立用户，{unique_sessions:,}个会话")
                insights.append(f"平均每用户{avg_events_per_user:.1f}个事件，每会话{avg_events_per_session:.1f}个事件")
                
                # 活跃度评估
                if avg_events_per_user > 50:
                    insights.append("用户活跃度较高，参与度良好")
                elif avg_events_per_user > 20:
                    insights.append("用户活跃度中等，有提升空间")
                else:
                    insights.append("用户活跃度偏低，需要关注用户参与度")
                    recommendations.append("建议优化用户体验，提升用户参与度")
            
            # 事件类型分布图表
            if stats.get('event_type_distribution'):
                charts.append({
                    'type': 'pie',
                    'title': '事件类型分布',
                    'data': stats['event_type_distribution']
                })
            
            # 每日活跃趋势图表
            if stats.get('daily_stats'):
                charts.append({
                    'type': 'line',
                    'title': '每日事件量趋势',
                    'data': stats['daily_stats']
                })
            
            return ReportSection(
                title="概览",
                content=stats,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"生成概览章节失败: {e}")
            return ReportSection("概览", {}, [], [f"生成概览数据时出错: {str(e)}"], [])
    
    async def _generate_pattern_analysis_section(
        self,
        period: Dict[str, datetime],
        user_ids: Optional[List[str]]
    ) -> ReportSection:
        """生成模式分析章节"""
        try:
            # 执行模式识别
            pattern_results = await self.pattern_engine.analyze_behavior_patterns(
                user_ids=user_ids,
                time_range_days=(period['end_time'] - period['start_time']).days
            )
            
            insights = []
            recommendations = []
            charts = []
            content = {}
            
            # 分析序列模式
            if 'sequence' in pattern_results:
                sequence_patterns = pattern_results['sequence'].patterns
                content['sequence_patterns'] = [p.model_dump(mode="json") for p in sequence_patterns]
                
                if sequence_patterns:
                    top_pattern = max(sequence_patterns, key=lambda p: p.support)
                    insights.append(f"发现{len(sequence_patterns)}个行为序列模式")
                    insights.append(f"最常见的行为序列支持度为{top_pattern.support:.1%}")
                    
                    # 序列模式支持度图表
                    charts.append({
                        'type': 'bar',
                        'title': '行为序列模式支持度',
                        'data': {
                            p.pattern_name: p.support * 100 
                            for p in sorted(sequence_patterns, key=lambda x: x.support, reverse=True)[:10]
                        }
                    })
                else:
                    insights.append("未发现明显的行为序列模式")
                    recommendations.append("建议增加数据收集时间或优化用户引导流程")
            
            # 分析聚类模式
            if 'clustering' in pattern_results:
                clustering_patterns = pattern_results['clustering'].patterns
                content['clustering_patterns'] = [p.model_dump(mode="json") for p in clustering_patterns]
                
                if clustering_patterns:
                    largest_cluster = max(clustering_patterns, key=lambda p: p.users_count)
                    insights.append(f"用户行为被分为{len(clustering_patterns)}个群体")
                    insights.append(f"最大用户群体包含{largest_cluster.users_count}个用户")
                    
                    # 用户群体分布图表
                    charts.append({
                        'type': 'pie',
                        'title': '用户行为群体分布',
                        'data': {
                            p.pattern_name: p.users_count 
                            for p in clustering_patterns
                        }
                    })
                    
                    recommendations.append("针对不同用户群体制定个性化的交互策略")
                else:
                    insights.append("用户行为模式相对统一")
            
            return ReportSection(
                title="行为模式分析",
                content=content,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"生成模式分析章节失败: {e}")
            return ReportSection("行为模式分析", {}, [], [f"模式分析出错: {str(e)}"], [])
    
    async def _generate_anomaly_detection_section(
        self,
        period: Dict[str, datetime],
        user_ids: Optional[List[str]]
    ) -> ReportSection:
        """生成异常检测章节"""
        try:
            # 执行异常检测
            detection_result = await self.anomaly_engine.detect_batch_anomalies(
                user_ids=user_ids,
                time_range_days=(period['end_time'] - period['start_time']).days
            )
            
            insights = []
            recommendations = []
            charts = []
            content = {
                'total_samples': detection_result.total_samples,
                'anomalies_count': len(detection_result.anomalies),
                'anomaly_rate': detection_result.anomaly_rate * 100,
                'processing_time': detection_result.processing_time_seconds
            }
            
            anomalies = detection_result.anomalies
            
            if anomalies:
                # 按严重程度统计
                severity_counts = {}
                for anomaly in anomalies:
                    severity_counts[anomaly.severity.value] = severity_counts.get(anomaly.severity.value, 0) + 1
                
                content['severity_distribution'] = severity_counts
                
                # 按类型统计
                type_counts = {}
                for anomaly in anomalies:
                    type_counts[anomaly.anomaly_type.value] = type_counts.get(anomaly.anomaly_type.value, 0) + 1
                
                content['type_distribution'] = type_counts
                
                insights.append(f"检测到{len(anomalies)}个异常事件，异常率为{detection_result.anomaly_rate:.1%}")
                
                critical_anomalies = [a for a in anomalies if a.severity.value == 'critical']
                if critical_anomalies:
                    insights.append(f"其中{len(critical_anomalies)}个为严重异常，需要立即关注")
                    recommendations.append("立即调查严重异常事件的根本原因")
                
                # 异常严重程度分布图表
                charts.append({
                    'type': 'pie',
                    'title': '异常严重程度分布',
                    'data': severity_counts
                })
                
                # 异常类型分布图表
                charts.append({
                    'type': 'bar',
                    'title': '异常类型分布',
                    'data': type_counts
                })
                
                # 异常时间分布图表
                hourly_anomalies = {}
                for anomaly in anomalies:
                    hour = anomaly.detected_at.hour
                    hourly_anomalies[f"{hour:02d}:00"] = hourly_anomalies.get(f"{hour:02d}:00", 0) + 1
                
                charts.append({
                    'type': 'bar',
                    'title': '异常事件时间分布',
                    'data': hourly_anomalies
                })
                
                recommendations.append("建立异常事件的实时监控和告警机制")
                recommendations.append("分析异常事件的时间模式，优化系统在高风险时段的稳定性")
            else:
                insights.append("分析期间未检测到异常事件")
                insights.append("系统运行状态良好")
            
            return ReportSection(
                title="异常检测分析",
                content=content,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"生成异常检测章节失败: {e}")
            return ReportSection("异常检测分析", {}, [], [f"异常检测出错: {str(e)}"], [])
    
    async def _generate_trend_analysis_section(
        self,
        period: Dict[str, datetime],
        user_ids: Optional[List[str]],
        include_forecasts: bool
    ) -> ReportSection:
        """生成趋势分析章节"""
        try:
            insights = []
            recommendations = []
            charts = []
            content = {}
            
            # 分析多个关键指标的趋势
            metrics_to_analyze = ['event_count', 'avg_duration', 'total_duration']
            
            for metric_name in metrics_to_analyze:
                try:
                    # 趋势分析
                    trend_analysis = await self.trend_analyzer.analyze_metric_trends(
                        metric_name=metric_name,
                        time_range_days=(period['end_time'] - period['start_time']).days,
                        user_id=user_ids[0] if user_ids and len(user_ids) == 1 else None
                    )
                    
                    if 'error' not in trend_analysis:
                        content[f'{metric_name}_analysis'] = trend_analysis
                        
                        # 趋势图表
                        if 'decomposition' in trend_analysis:
                            decomp = trend_analysis['decomposition']
                            charts.append({
                                'type': 'line',
                                'title': f'{metric_name} 趋势分解',
                                'data': {
                                    'dates': decomp['dates'],
                                    'original': trend_analysis['decomposition'].get('original_values', []),
                                    'trend': decomp['trend'],
                                    'seasonal': decomp['seasonal']
                                }
                            })
                        
                        # 分析趋势方向
                        trend_stats = trend_analysis.get('trend_statistics', {})
                        trend_direction = trend_stats.get('trend_direction', 'unknown')
                        
                        if trend_direction == 'increasing':
                            insights.append(f"{metric_name} 呈上升趋势")
                        elif trend_direction == 'decreasing':
                            insights.append(f"{metric_name} 呈下降趋势")
                        else:
                            insights.append(f"{metric_name} 趋势相对平稳")
                        
                        # 季节性分析
                        seasonal_strength = trend_stats.get('seasonal_strength', 0)
                        if seasonal_strength > 0.3:
                            insights.append(f"{metric_name} 存在明显的季节性模式")
                        
                        # 预测
                        if include_forecasts:
                            try:
                                forecast = await self.trend_analyzer.generate_forecast(
                                    metric_name=metric_name,
                                    forecast_periods=7,
                                    time_range_days=(period['end_time'] - period['start_time']).days,
                                    user_id=user_ids[0] if user_ids and len(user_ids) == 1 else None
                                )
                                
                                content[f'{metric_name}_forecast'] = forecast.model_dump(mode="json")
                                
                                # 预测图表
                                charts.append({
                                    'type': 'line',
                                    'title': f'{metric_name} 7天预测',
                                    'data': {
                                        'dates': [d.isoformat() for d in forecast.forecast_dates],
                                        'predicted': forecast.predicted_values,
                                        'confidence_lower': forecast.confidence_lower,
                                        'confidence_upper': forecast.confidence_upper
                                    }
                                })
                                
                            except Exception as e:
                                logger.warning(f"生成{metric_name}预测失败: {e}")
                
                except Exception as e:
                    logger.warning(f"分析{metric_name}趋势失败: {e}")
            
            if insights:
                recommendations.append("持续监控关键指标趋势，及时发现业务变化")
                recommendations.append("基于季节性模式优化资源配置和用户体验")
            
            return ReportSection(
                title="趋势分析",
                content=content,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"生成趋势分析章节失败: {e}")
            return ReportSection("趋势分析", {}, [], [f"趋势分析出错: {str(e)}"], [])
    
    async def _generate_user_behavior_section(
        self,
        period: Dict[str, datetime],
        user_ids: Optional[List[str]]
    ) -> ReportSection:
        """生成用户行为章节"""
        try:
            insights = []
            recommendations = []
            charts = []
            content = {}
            
            # 获取用户会话统计
            if user_ids:
                all_sessions = []
                for user_id in user_ids[:10]:  # 限制分析用户数量
                    sessions = await self.event_store.get_user_sessions(user_id, limit=50)
                    all_sessions.extend(sessions)
                
                if all_sessions:
                    # 会话时长分析
                    session_durations = []
                    for session in all_sessions:
                        if session.get('duration_seconds'):
                            session_durations.append(session['duration_seconds'] / 60)  # 转为分钟
                    
                    if session_durations:
                        avg_duration = sum(session_durations) / len(session_durations)
                        max_duration = max(session_durations)
                        
                        content['session_analysis'] = {
                            'total_sessions': len(all_sessions),
                            'avg_duration_minutes': avg_duration,
                            'max_duration_minutes': max_duration
                        }
                        
                        insights.append(f"平均会话时长为{avg_duration:.1f}分钟")
                        
                        if avg_duration > 10:
                            insights.append("用户会话时长较长，用户参与度较高")
                        elif avg_duration > 5:
                            insights.append("用户会话时长适中")
                        else:
                            insights.append("用户会话时长较短，可能存在用户体验问题")
                            recommendations.append("分析用户快速离开的原因，优化用户留存")
                        
                        # 会话时长分布图表
                        duration_ranges = {
                            '0-2分钟': len([d for d in session_durations if d <= 2]),
                            '2-5分钟': len([d for d in session_durations if 2 < d <= 5]),
                            '5-10分钟': len([d for d in session_durations if 5 < d <= 10]),
                            '10分钟以上': len([d for d in session_durations if d > 10])
                        }
                        
                        charts.append({
                            'type': 'pie',
                            'title': '会话时长分布',
                            'data': duration_ranges
                        })
            else:
                insights.append("未指定具体用户，无法进行详细的用户行为分析")
                recommendations.append("建议对重点用户进行个性化行为分析")
            
            return ReportSection(
                title="用户行为深度分析",
                content=content,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"生成用户行为章节失败: {e}")
            return ReportSection("用户行为深度分析", {}, [], [f"用户行为分析出错: {str(e)}"], [])
    
    async def _generate_executive_summary(
        self,
        sections: List[ReportSection],
        period: Dict[str, datetime]
    ) -> str:
        """生成执行摘要"""
        try:
            key_insights = []
            key_recommendations = []
            
            # 收集各章节的关键洞察
            for section in sections:
                if section.insights:
                    key_insights.extend(section.insights[:2])  # 每章节最多2个洞察
                if section.recommendations:
                    key_recommendations.extend(section.recommendations[:1])  # 每章节最多1个建议
            
            # 构建执行摘要
            summary_parts = []
            
            period_str = f"{period['start_time'].strftime('%Y年%m月%d日')}至{period['end_time'].strftime('%Y年%m月%d日')}"
            summary_parts.append(f"本报告分析了{period_str}期间的用户行为数据。")
            
            if key_insights:
                summary_parts.append("关键发现包括：")
                for i, insight in enumerate(key_insights[:5], 1):
                    summary_parts.append(f"{i}. {insight}")
            
            if key_recommendations:
                summary_parts.append("\n建议采取的行动包括：")
                for i, recommendation in enumerate(key_recommendations[:3], 1):
                    summary_parts.append(f"{i}. {recommendation}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"生成执行摘要失败: {e}")
            return "报告生成过程中出现错误，请参阅详细章节内容。"
    
    async def _generate_appendix(
        self,
        period: Dict[str, datetime],
        user_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """生成附录"""
        return {
            'generation_metadata': {
                'generated_at': utc_now().isoformat(),
                'period_analyzed': {
                    'start': period['start_time'].isoformat(),
                    'end': period['end_time'].isoformat(),
                    'days': (period['end_time'] - period['start_time']).days
                },
                'scope': {
                    'user_count': len(user_ids) if user_ids else 'all_users',
                    'user_ids': user_ids[:10] if user_ids else None  # 只显示前10个
                }
            },
            'methodology': {
                'pattern_recognition': 'PrefixSpan序列模式挖掘 + K-means聚类',
                'anomaly_detection': 'Isolation Forest + 统计异常检测',
                'trend_analysis': 'Prophet时间序列预测 + 季节性分解',
                'data_quality': '实时数据质量监控和验证'
            },
            'definitions': {
                'anomaly_rate': '异常事件数量占总事件数量的比例',
                'pattern_support': '模式在所有序列中出现的频率',
                'seasonal_strength': '季节性成分相对于整体变异的强度',
                'trend_direction': '基于线性回归确定的总体趋势方向'
            }
        }

class ReportExporter:
    """报告导出器"""
    
    @staticmethod
    async def export_to_json(report: AnalyticsReport) -> str:
        """导出为JSON格式"""
        try:
            report_dict = {
                'report_id': report.report_id,
                'title': report.title,
                'generated_at': report.generated_at.isoformat(),
                'period': {
                    'start_time': report.period['start_time'].isoformat(),
                    'end_time': report.period['end_time'].isoformat()
                },
                'executive_summary': report.executive_summary,
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'charts': section.charts,
                        'insights': section.insights,
                        'recommendations': section.recommendations
                    }
                    for section in report.sections
                ],
                'appendix': report.appendix,
                'metadata': report.metadata
            }
            
            return json.dumps(report_dict, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"导出JSON格式失败: {e}")
            raise
    
    @staticmethod
    async def export_to_html(report: AnalyticsReport) -> str:
        """导出为HTML格式"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ border-bottom: 3px solid #007acc; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ color: #007acc; border-left: 4px solid #007acc; padding-left: 15px; }}
        .insights, .recommendations {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .insights h4 {{ color: #28a745; }}
        .recommendations h4 {{ color: #ffc107; }}
        .chart-placeholder {{ background: #e9ecef; padding: 20px; text-align: center; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report.title}</h1>
        <p><strong>生成时间:</strong> {report.generated_at.strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        <p><strong>分析期间:</strong> {report.period['start_time'].strftime('%Y年%m月%d日')} 至 {report.period['end_time'].strftime('%Y年%m月%d日')}</p>
    </div>
    
    <div class="section">
        <h2>执行摘要</h2>
        <p>{report.executive_summary.replace(chr(10), '<br>')}</p>
    </div>
"""
            
            # 添加各个章节
            for section in report.sections:
                html_content += f"""
    <div class="section">
        <h2>{section.title}</h2>
        
        {f'<div class="insights"><h4>关键洞察</h4><ul>' + ''.join([f'<li>{insight}</li>' for insight in section.insights]) + '</ul></div>' if section.insights else ''}
        
        {f'<div class="recommendations"><h4>建议</h4><ul>' + ''.join([f'<li>{rec}</li>' for rec in section.recommendations]) + '</ul></div>' if section.recommendations else ''}
        
        {f'<div class="charts">' + ''.join([f'<div class="chart-placeholder">图表: {chart["title"]}</div>' for chart in section.charts]) + '</div>' if section.charts else ''}
    </div>
"""
            
            html_content += """
</body>
</html>
"""
            
            return html_content
            
        except Exception as e:
            logger.error(f"导出HTML格式失败: {e}")
            raise
