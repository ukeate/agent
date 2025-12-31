"""解释格式化器

本模块实现解释的多格式输出，支持HTML、Markdown、JSON等格式的结构化展示。
"""

import json
import html
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Any, Dict, List, Optional, Union
from jinja2 import Template
from src.models.schemas.explanation import (

    DecisionExplanation,
    ExplanationComponent,
    ExplanationLevel,
    ExplanationType,
    ConfidenceMetrics,
    CounterfactualScenario
)

from src.core.logging import get_logger
logger = get_logger(__name__)

class ExplanationFormatter:
    """解释格式化器"""
    
    def __init__(self):
        """初始化格式化器"""
        self.html_templates = self._load_html_templates()
        self.markdown_templates = self._load_markdown_templates()
        
        # 格式化配置
        self.format_config = {
            "html": {
                "include_styles": True,
                "responsive": True,
                "interactive": True
            },
            "markdown": {
                "include_toc": True,
                "use_tables": True,
                "emoji_support": True
            },
            "json": {
                "pretty_print": True,
                "include_metadata": True,
                "compact_arrays": False
            },
            "text": {
                "line_width": 80,
                "indent_size": 2,
                "bullet_style": "•"
            }
        }
    
    def format_explanation(
        self,
        explanation: DecisionExplanation,
        output_format: str = "html",
        template_name: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """格式化解释为指定格式"""
        
        format_config = {**self.format_config.get(output_format, {})}
        if custom_config:
            format_config.update(custom_config)
        
        if output_format.lower() == "html":
            return self._format_to_html(explanation, template_name, format_config)
        elif output_format.lower() == "markdown":
            return self._format_to_markdown(explanation, template_name, format_config)
        elif output_format.lower() == "json":
            return self._format_to_json(explanation, format_config)
        elif output_format.lower() == "text":
            return self._format_to_text(explanation, format_config)
        elif output_format.lower() == "xml":
            return self._format_to_xml(explanation, format_config)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
    
    def _format_to_html(
        self,
        explanation: DecisionExplanation,
        template_name: Optional[str],
        config: Dict[str, Any]
    ) -> str:
        """格式化为HTML"""
        
        template_name = template_name or "default"
        template = self.html_templates.get(template_name, self.html_templates["default"])
        
        # 准备模板数据
        template_data = {
            "explanation": explanation,
            "config": config,
            "formatted_date": self._format_datetime(explanation.created_at),
            "confidence_percentage": f"{explanation.confidence_metrics.overall_confidence:.1%}",
            "components_by_importance": sorted(
                explanation.components,
                key=lambda x: x.weight * x.impact_score,
                reverse=True
            ),
            "visualization_charts": self._prepare_chart_data(explanation.visualization_data),
            "counterfactual_summary": self._summarize_counterfactuals(explanation.counterfactuals)
        }
        
        # 渲染模板
        html_output = template.render(**template_data)
        
        # 添加样式
        if config.get("include_styles", True):
            html_output = self._add_html_styles(html_output, config)
        
        return html_output
    
    def _format_to_markdown(
        self,
        explanation: DecisionExplanation,
        template_name: Optional[str],
        config: Dict[str, Any]
    ) -> str:
        """格式化为Markdown"""
        
        template_name = template_name or "default"
        template = self.markdown_templates.get(template_name, self.markdown_templates["default"])
        
        # 准备模板数据
        template_data = {
            "explanation": explanation,
            "config": config,
            "decision_emoji": self._get_decision_emoji(explanation.decision_outcome),
            "confidence_emoji": self._get_confidence_emoji(explanation.confidence_metrics.overall_confidence),
            "components_table": self._create_components_table(explanation.components),
            "counterfactuals_list": self._create_counterfactuals_list(explanation.counterfactuals),
            "confidence_bars": self._create_confidence_bars(explanation.confidence_metrics)
        }
        
        return template.render(**template_data)
    
    def _format_to_json(
        self,
        explanation: DecisionExplanation,
        config: Dict[str, Any]
    ) -> str:
        """格式化为JSON"""
        
        # 转换为字典
        explanation_dict = explanation.model_dump()
        
        # 添加格式化的辅助数据
        if config.get("include_metadata", True):
            explanation_dict["_formatted"] = {
                "confidence_percentage": f"{explanation.confidence_metrics.overall_confidence:.1%}",
                "formatted_date": self._format_datetime(explanation.created_at),
                "component_count": len(explanation.components),
                "counterfactual_count": len(explanation.counterfactuals),
                "explanation_length": len(explanation.summary_explanation or ""),
                "format_timestamp": utc_now().isoformat()
            }
        
        # 格式化输出
        indent = 2 if config.get("pretty_print", True) else None
        return json.dumps(explanation_dict, indent=indent, ensure_ascii=False, default=str)
    
    def _format_to_text(
        self,
        explanation: DecisionExplanation,
        config: Dict[str, Any]
    ) -> str:
        """格式化为纯文本"""
        
        line_width = config.get("line_width", 80)
        indent = " " * config.get("indent_size", 2)
        bullet = config.get("bullet_style", "•")
        
        lines = []
        
        # 标题
        lines.append("=" * line_width)
        lines.append(f"决策解释报告 - {explanation.decision_id}")
        lines.append("=" * line_width)
        lines.append("")
        
        # 基本信息
        lines.append("📋 基本信息")
        lines.append("-" * 20)
        lines.append(f"{indent}决策结果: {explanation.decision_outcome}")
        lines.append(f"{indent}置信度: {explanation.confidence_metrics.overall_confidence:.1%}")
        lines.append(f"{indent}生成时间: {self._format_datetime(explanation.created_at)}")
        lines.append("")
        
        # 概要解释
        if explanation.summary_explanation:
            lines.append("📝 概要解释")
            lines.append("-" * 20)
            summary_lines = self._wrap_text(explanation.summary_explanation, line_width - len(indent))
            for line in summary_lines:
                lines.append(f"{indent}{line}")
            lines.append("")
        
        # 关键因素
        if explanation.components:
            lines.append("🔍 关键因素")
            lines.append("-" * 20)
            for i, component in enumerate(sorted(
                explanation.components,
                key=lambda x: x.weight * x.impact_score,
                reverse=True
            )[:5], 1):
                importance = component.weight * component.impact_score
                lines.append(f"{indent}{i}. {component.factor_name}")
                lines.append(f"{indent}{indent}值: {component.factor_value}")
                lines.append(f"{indent}{indent}重要性: {importance:.2f}")
                lines.append(f"{indent}{indent}说明: {component.evidence_content}")
                lines.append("")
        
        # 置信度分析
        lines.append("📊 置信度分析")
        lines.append("-" * 20)
        confidence = explanation.confidence_metrics
        lines.append(f"{indent}整体置信度: {confidence.overall_confidence:.1%}")
        lines.append(f"{indent}不确定性: {confidence.uncertainty_score:.1%}")
        if confidence.confidence_interval_lower and confidence.confidence_interval_upper:
            lines.append(f"{indent}置信区间: {confidence.confidence_interval_lower:.1%} - {confidence.confidence_interval_upper:.1%}")
        lines.append("")
        
        # 反事实分析
        if explanation.counterfactuals:
            lines.append("🔮 反事实分析")
            lines.append("-" * 20)
            for i, scenario in enumerate(explanation.counterfactuals, 1):
                lines.append(f"{indent}{i}. {scenario.scenario_name}")
                lines.append(f"{indent}{indent}预测结果: {scenario.predicted_outcome}")
                lines.append(f"{indent}{indent}影响差异: {scenario.impact_difference:+.1%}")
                lines.append(f"{indent}{indent}说明: {scenario.explanation}")
                lines.append("")
        
        # 详细解释
        if explanation.detailed_explanation:
            lines.append("📖 详细解释")
            lines.append("-" * 20)
            detailed_lines = self._wrap_text(explanation.detailed_explanation, line_width - len(indent))
            for line in detailed_lines:
                lines.append(f"{indent}{line}")
            lines.append("")
        
        # 技术细节
        if explanation.technical_explanation:
            lines.append("⚙️ 技术细节")
            lines.append("-" * 20)
            technical_lines = self._wrap_text(explanation.technical_explanation, line_width - len(indent))
            for line in technical_lines:
                lines.append(f"{indent}{line}")
            lines.append("")
        
        lines.append("=" * line_width)
        
        return "\n".join(lines)
    
    def _format_to_xml(
        self,
        explanation: DecisionExplanation,
        config: Dict[str, Any]
    ) -> str:
        """格式化为XML"""
        
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<decision_explanation>')
        
        # 基本信息
        lines.append(f'  <id>{html.escape(str(explanation.id))}</id>')
        lines.append(f'  <decision_id>{html.escape(explanation.decision_id)}</decision_id>')
        lines.append(f'  <type>{explanation.explanation_type.value}</type>')
        lines.append(f'  <level>{explanation.explanation_level.value}</level>')
        lines.append(f'  <outcome>{html.escape(explanation.decision_outcome)}</outcome>')
        lines.append(f'  <created_at>{explanation.created_at.isoformat()}</created_at>')
        
        # 置信度指标
        lines.append('  <confidence_metrics>')
        confidence = explanation.confidence_metrics
        lines.append(f'    <overall_confidence>{confidence.overall_confidence}</overall_confidence>')
        lines.append(f'    <uncertainty_score>{confidence.uncertainty_score}</uncertainty_score>')
        if confidence.confidence_interval_lower:
            lines.append(f'    <confidence_interval_lower>{confidence.confidence_interval_lower}</confidence_interval_lower>')
        if confidence.confidence_interval_upper:
            lines.append(f'    <confidence_interval_upper>{confidence.confidence_interval_upper}</confidence_interval_upper>')
        lines.append('  </confidence_metrics>')
        
        # 解释文本
        if explanation.summary_explanation:
            lines.append(f'  <summary_explanation><![CDATA[{explanation.summary_explanation}]]></summary_explanation>')
        if explanation.detailed_explanation:
            lines.append(f'  <detailed_explanation><![CDATA[{explanation.detailed_explanation}]]></detailed_explanation>')
        if explanation.technical_explanation:
            lines.append(f'  <technical_explanation><![CDATA[{explanation.technical_explanation}]]></technical_explanation>')
        
        # 解释组件
        if explanation.components:
            lines.append('  <components>')
            for component in explanation.components:
                lines.append('    <component>')
                lines.append(f'      <factor_name>{html.escape(component.factor_name)}</factor_name>')
                lines.append(f'      <factor_value>{html.escape(str(component.factor_value))}</factor_value>')
                lines.append(f'      <weight>{component.weight}</weight>')
                lines.append(f'      <impact_score>{component.impact_score}</impact_score>')
                lines.append(f'      <evidence_content><![CDATA[{component.evidence_content}]]></evidence_content>')
                lines.append('    </component>')
            lines.append('  </components>')
        
        # 反事实场景
        if explanation.counterfactuals:
            lines.append('  <counterfactual_scenarios>')
            for scenario in explanation.counterfactuals:
                lines.append('    <scenario>')
                lines.append(f'      <scenario_name><![CDATA[{scenario.scenario_name}]]></scenario_name>')
                lines.append(f'      <predicted_outcome><![CDATA[{scenario.predicted_outcome}]]></predicted_outcome>')
                lines.append(f'      <impact_difference>{scenario.impact_difference}</impact_difference>')
                lines.append(f'      <explanation><![CDATA[{scenario.explanation}]]></explanation>')
                lines.append('    </scenario>')
            lines.append('  </counterfactual_scenarios>')
        
        lines.append('</decision_explanation>')
        
        return '\n'.join(lines)
    
    def _load_html_templates(self) -> Dict[str, Template]:
        """加载HTML模板"""
        
        default_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>决策解释报告 - {{ explanation.decision_id }}</title>
    {% if config.include_styles %}
    <style>
        body { font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #e0e0e0; }
        .confidence-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; }
        .confidence-high { background-color: #4CAF50; }
        .confidence-medium { background-color: #FF9800; }
        .confidence-low { background-color: #f44336; }
        .section { margin: 30px 0; }
        .section-title { font-size: 1.3em; font-weight: bold; color: #333; margin-bottom: 15px; padding-left: 10px; border-left: 4px solid #2196F3; }
        .factor-card { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #2196F3; }
        .factor-importance { float: right; background: #e3f2fd; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; }
        .progress-bar { width: 100%; height: 8px; background: #e0e0e0; border-radius: 4px; margin: 5px 0; }
        .progress-fill { height: 100%; background: linear-gradient(to right, #4CAF50, #2196F3); border-radius: 4px; }
        .counterfactual { background: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #FF9800; }
        .metadata { font-size: 0.9em; color: #666; text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; }
    </style>
    {% endif %}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>决策解释报告</h1>
            <p><strong>决策ID:</strong> {{ explanation.decision_id }}</p>
            <p><strong>决策结果:</strong> {{ explanation.decision_outcome }}</p>
            <span class="confidence-badge {{ 'confidence-high' if explanation.confidence_metrics.overall_confidence > 0.7 else 'confidence-medium' if explanation.confidence_metrics.overall_confidence > 0.4 else 'confidence-low' }}">
                置信度: {{ confidence_percentage }}
            </span>
        </div>

        {% if explanation.summary_explanation %}
        <div class="section">
            <h2 class="section-title">📝 概要解释</h2>
            <p>{{ explanation.summary_explanation }}</p>
        </div>
        {% endif %}

        {% if components_by_importance %}
        <div class="section">
            <h2 class="section-title">🔍 关键影响因素</h2>
            {% for component in components_by_importance[:5] %}
            <div class="factor-card">
                <div class="factor-importance">重要性: {{ "%.2f"|format(component.weight * component.impact_score) }}</div>
                <h3>{{ component.factor_name }}</h3>
                <p><strong>值:</strong> {{ component.factor_value }}</p>
                <p>{{ component.contribution_explanation }}</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ (component.weight * component.impact_score * 100)|int }}%"></div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="section">
            <h2 class="section-title">📊 置信度分析</h2>
            <p><strong>整体置信度:</strong> {{ confidence_percentage }}</p>
            <p><strong>不确定性:</strong> {{ "%.1f"|format(explanation.confidence_metrics.uncertainty_score * 100) }}%</p>
            {% if explanation.confidence_metrics.confidence_interval_lower and explanation.confidence_metrics.confidence_interval_upper %}
            <p><strong>置信区间:</strong> {{ "%.1f"|format(explanation.confidence_metrics.confidence_interval_lower * 100) }}% - {{ "%.1f"|format(explanation.confidence_metrics.confidence_interval_upper * 100) }}%</p>
            {% endif %}
        </div>

        {% if explanation.counterfactuals %}
        <div class="section">
            <h2 class="section-title">🔮 反事实分析</h2>
            {% for scenario in explanation.counterfactuals %}
            <div class="counterfactual">
                <h3>{{ scenario.scenario_name }}</h3>
                <p><strong>预测结果:</strong> {{ scenario.predicted_outcome }}</p>
                <p><strong>影响差异:</strong> {{ "{:+.1f}".format(scenario.impact_difference * 100) }}%</p>
                <p>{{ scenario.explanation }}</p>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if explanation.detailed_explanation %}
        <div class="section">
            <h2 class="section-title">📖 详细解释</h2>
            <p>{{ explanation.detailed_explanation }}</p>
        </div>
        {% endif %}

        {% if explanation.technical_explanation %}
        <div class="section">
            <h2 class="section-title">⚙️ 技术细节</h2>
            <p>{{ explanation.technical_explanation }}</p>
        </div>
        {% endif %}

        <div class="metadata">
            <p>报告生成时间: {{ formatted_date }}</p>
            <p>解释类型: {{ explanation.explanation_type.value }} | 详细程度: {{ explanation.explanation_level.value }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return {
            "default": Template(default_html),
            "minimal": Template(self._get_minimal_html_template()),
            "dashboard": Template(self._get_dashboard_html_template())
        }
    
    def _load_markdown_templates(self) -> Dict[str, Template]:
        """加载Markdown模板"""
        
        default_markdown = """# {{ decision_emoji }} 决策解释报告

**决策ID:** `{{ explanation.decision_id }}`  
**决策结果:** {{ explanation.decision_outcome }}  
**置信度:** {{ confidence_emoji }} {{ "%.1f"|format(explanation.confidence_metrics.overall_confidence * 100) }}%  
**生成时间:** {{ explanation.created_at.strftime("%Y-%m-%d %H:%M:%S") }}

---

## 📝 概要解释

{{ explanation.summary_explanation or "无概要解释" }}

## 🔍 关键影响因素

{% if explanation.components %}
{{ components_table }}
{% else %}
*暂无关键因素数据*
{% endif %}

## 📊 置信度分析

{{ confidence_bars }}

- **整体置信度:** {{ "%.1f"|format(explanation.confidence_metrics.overall_confidence * 100) }}%
- **不确定性评分:** {{ "%.1f"|format(explanation.confidence_metrics.uncertainty_score * 100) }}%
{% if explanation.confidence_metrics.confidence_interval_lower and explanation.confidence_metrics.confidence_interval_upper %}
- **置信区间:** {{ "%.1f"|format(explanation.confidence_metrics.confidence_interval_lower * 100) }}% - {{ "%.1f"|format(explanation.confidence_metrics.confidence_interval_upper * 100) }}%
{% endif %}

{% if explanation.counterfactuals %}
## 🔮 反事实分析

{{ counterfactuals_list }}
{% endif %}

{% if explanation.detailed_explanation %}
## 📖 详细解释

{{ explanation.detailed_explanation }}
{% endif %}

{% if explanation.technical_explanation %}
## ⚙️ 技术细节

{{ explanation.technical_explanation }}
{% endif %}

---

*此报告由AI决策解释系统自动生成 | 解释类型: {{ explanation.explanation_type.value }} | 详细程度: {{ explanation.explanation_level.value }}*
        """
        
        return {
            "default": Template(default_markdown),
            "github": Template(self._get_github_markdown_template()),
            "technical": Template(self._get_technical_markdown_template())
        }
    
    def _get_minimal_html_template(self) -> str:
        """获取极简HTML模板"""
        return """
<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
    <h2>{{ explanation.decision_outcome }}</h2>
    <p><strong>置信度:</strong> {{ confidence_percentage }}</p>
    <p>{{ explanation.summary_explanation }}</p>
</div>
        """
    
    def _get_dashboard_html_template(self) -> str:
        """获取仪表板HTML模板"""
        return """
<div class="dashboard-widget">
    <div class="metric-card">
        <h3>{{ explanation.decision_outcome }}</h3>
        <div class="confidence-meter">
            <div class="meter-fill" style="width: {{ (explanation.confidence_metrics.overall_confidence * 100)|int }}%"></div>
        </div>
        <span class="confidence-text">{{ confidence_percentage }}</span>
    </div>
    <div class="summary">{{ explanation.summary_explanation[:100] }}...</div>
</div>
        """
    
    def _get_github_markdown_template(self) -> str:
        """获取GitHub风格Markdown模板"""
        return """# :robot: AI决策解释报告

> **决策ID:** `{{ explanation.decision_id }}`  
> **结果:** {{ explanation.decision_outcome }}  
> **置信度:** {{ confidence_emoji }} {{ "%.1f"|format(explanation.confidence_metrics.overall_confidence * 100) }}%

## :memo: 概要

{{ explanation.summary_explanation }}

## :bar_chart: 关键因素

{{ components_table }}

## :warning: 注意事项

{% for scenario in explanation.counterfactuals[:2] %}
- **{{ scenario.scenario_description }}:** {{ scenario.predicted_outcome }}
{% endfor %}
        """
    
    def _get_technical_markdown_template(self) -> str:
        """获取技术文档Markdown模板"""
        return """# Technical Decision Analysis Report

## Executive Summary
- **Decision ID:** `{{ explanation.decision_id }}`
- **Outcome:** {{ explanation.decision_outcome }}
- **Confidence:** {{ "%.3f"|format(explanation.confidence_metrics.overall_confidence) }}
- **Uncertainty:** {{ "%.3f"|format(explanation.confidence_metrics.uncertainty_score) }}

## Model Analysis
{{ explanation.technical_explanation or "No technical details available" }}

## Factor Analysis
{{ components_table }}

## Counterfactual Analysis
{{ counterfactuals_list }}
        """
    
    def _format_datetime(self, dt: datetime) -> str:
        """格式化日期时间"""
        return dt.strftime("%Y年%m月%d日 %H:%M:%S")
    
    def _get_decision_emoji(self, outcome: str) -> str:
        """获取决策表情符号"""
        outcome_lower = outcome.lower()
        if "approved" in outcome_lower or "通过" in outcome_lower or "同意" in outcome_lower:
            return "✅"
        elif "rejected" in outcome_lower or "拒绝" in outcome_lower or "否决" in outcome_lower:
            return "❌"
        elif "pending" in outcome_lower or "待定" in outcome_lower:
            return "⏳"
        else:
            return "📋"
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        """获取置信度表情符号"""
        if confidence > 0.8:
            return "🟢"
        elif confidence > 0.6:
            return "🟡"
        elif confidence > 0.4:
            return "🟠"
        else:
            return "🔴"
    
    def _create_components_table(self, components: List[ExplanationComponent]) -> str:
        """创建因素表格"""
        if not components:
            return "*暂无关键因素*"
        
        lines = ["| 因素名称 | 值 | 权重 | 影响 | 重要性 |", "| --- | --- | --- | --- | --- |"]
        
        for component in sorted(components, key=lambda x: x.weight * x.impact_score, reverse=True)[:5]:
            importance = component.weight * component.impact_score
            lines.append(
                f"| {component.factor_name} | {component.factor_value} | "
                f"{component.weight:.2f} | {component.impact_score:.2f} | {importance:.2f} |"
            )
        
        return "\n".join(lines)
    
    def _create_counterfactuals_list(self, counterfactuals: List[CounterfactualScenario]) -> str:
        """创建反事实列表"""
        if not counterfactuals:
            return "*暂无反事实分析*"
        
        lines = []
        for i, scenario in enumerate(counterfactuals, 1):
            lines.append(f"{i}. **{scenario.scenario_name}**")
            lines.append(f"   - 预测结果: {scenario.predicted_outcome}")
            lines.append(f"   - 影响差异: {scenario.impact_difference:+.1%}")
            lines.append(f"   - 说明: {scenario.explanation}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_confidence_bars(self, metrics: ConfidenceMetrics) -> str:
        """创建置信度进度条"""
        confidence_bar = "█" * int(metrics.overall_confidence * 10) + "░" * (10 - int(metrics.overall_confidence * 10))
        uncertainty_bar = "█" * int(metrics.uncertainty_score * 10) + "░" * (10 - int(metrics.uncertainty_score * 10))
        
        return f"""
```
置信度: {confidence_bar} {metrics.overall_confidence:.1%}
不确定性: {uncertainty_bar} {metrics.uncertainty_score:.1%}
```
        """
    
    def _prepare_chart_data(self, visualization_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """准备图表数据"""
        if not visualization_data:
            return {}
        
        return {
            "has_charts": True,
            "chart_count": len(visualization_data),
            "available_charts": list(visualization_data.keys())
        }
    
    def _summarize_counterfactuals(self, counterfactuals: List[CounterfactualScenario]) -> str:
        """总结反事实场景"""
        if not counterfactuals:
            return "无反事实分析"
        
        return f"共{len(counterfactuals)}个反事实场景"
    
    def _add_html_styles(self, html_content: str, config: Dict[str, Any]) -> str:
        """添加HTML样式"""
        if config.get("responsive", True):
            responsive_css = """
            <style>
            @media (max-width: 768px) {
                .container { padding: 15px; }
                .factor-importance { float: none; margin-top: 10px; }
            }
            </style>
            """
            html_content = html_content.replace("</head>", responsive_css + "</head>")
        
        return html_content
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """文本换行"""
        if not text:
            return []
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= width:
                current_line = (current_line + " " + word).strip()
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines

class ExplanationExporter:
    """解释导出器"""
    
    def __init__(self, formatter: Optional[ExplanationFormatter] = None):
        """初始化导出器"""
        self.formatter = formatter or ExplanationFormatter()
    
    def export_to_file(
        self,
        explanation: DecisionExplanation,
        file_path: str,
        output_format: Optional[str] = None,
        template_name: Optional[str] = None
    ) -> bool:
        """导出解释到文件"""
        
        try:
            # 自动检测格式
            if output_format is None:
                if file_path.endswith('.html'):
                    output_format = 'html'
                elif file_path.endswith('.md'):
                    output_format = 'markdown'
                elif file_path.endswith('.json'):
                    output_format = 'json'
                elif file_path.endswith('.xml'):
                    output_format = 'xml'
                else:
                    output_format = 'text'
            
            # 格式化内容
            formatted_content = self.formatter.format_explanation(
                explanation, output_format, template_name
            )
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            return True
            
        except Exception as e:
            logger.error("导出失败", error=str(e), exc_info=True)
            return False
    
    def batch_export(
        self,
        explanations: List[DecisionExplanation],
        output_dir: str,
        formats: List[str] = None
    ) -> Dict[str, bool]:
        """批量导出解释"""
        
        if formats is None:
            formats = ['html', 'json']
        
        results = {}
        
        for explanation in explanations:
            for format_type in formats:
                file_name = f"{explanation.decision_id}.{format_type}"
                file_path = f"{output_dir}/{file_name}"
                
                success = self.export_to_file(explanation, file_path, format_type)
                results[file_path] = success
        
        return results
