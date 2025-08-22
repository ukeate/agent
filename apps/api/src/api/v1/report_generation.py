"""
实验报告生成API端点
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from ...services.report_generation_service import (
    ReportGenerationService,
    ReportFormat,
    ReportSection,
    ReportScheduler
)


router = APIRouter(prefix="/reports", tags=["Report Generation"])

# 服务实例
report_service = ReportGenerationService()
report_scheduler = ReportScheduler(report_service)


class GenerateReportRequest(BaseModel):
    """生成报告请求"""
    experiment_id: str = Field(..., description="实验ID")
    sections: Optional[List[ReportSection]] = Field(
        None, 
        description="要包含的报告章节"
    )
    format: ReportFormat = Field(
        ReportFormat.JSON, 
        description="报告格式"
    )
    include_segments: bool = Field(
        True, 
        description="是否包含细分分析"
    )
    confidence_level: float = Field(
        0.95, 
        ge=0.5, 
        le=0.999,
        description="置信水平"
    )


class ScheduleReportRequest(BaseModel):
    """调度报告请求"""
    experiment_id: str = Field(..., description="实验ID")
    frequency: str = Field(..., description="报告频率: daily, weekly, on_demand")
    send_time: str = Field("09:00", description="发送时间 (HH:MM)")
    recipients: List[str] = Field([], description="接收人邮箱列表")
    format: ReportFormat = Field(ReportFormat.HTML, description="报告格式")


class ReportTemplateRequest(BaseModel):
    """报告模板请求"""
    name: str = Field(..., description="模板名称")
    description: str = Field(..., description="模板描述")
    sections: List[ReportSection] = Field(..., description="包含的章节")
    default_format: ReportFormat = Field(ReportFormat.JSON, description="默认格式")
    custom_settings: Dict[str, Any] = Field({}, description="自定义设置")


class BatchReportRequest(BaseModel):
    """批量报告请求"""
    experiment_ids: List[str] = Field(..., description="实验ID列表")
    format: ReportFormat = Field(ReportFormat.JSON, description="报告格式")
    merge_results: bool = Field(False, description="是否合并结果")


@router.post("/generate")
async def generate_report(request: GenerateReportRequest) -> Dict[str, Any]:
    """
    生成实验报告
    
    支持多种格式和自定义章节
    """
    try:
        report = await report_service.generate_report(
            experiment_id=request.experiment_id,
            sections=request.sections,
            format=request.format,
            include_segments=request.include_segments,
            confidence_level=request.confidence_level
        )
        
        return {
            "success": True,
            "report": report,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-summary")
async def generate_summary(experiment_id: str) -> Dict[str, Any]:
    """
    生成快速摘要报告
    
    只包含执行摘要和关键指标
    """
    try:
        report = await report_service.generate_report(
            experiment_id=experiment_id,
            sections=[ReportSection.EXECUTIVE_SUMMARY, ReportSection.METRIC_RESULTS],
            format=ReportFormat.JSON,
            include_segments=False
        )
        
        return {
            "success": True,
            "summary": report,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule")
async def schedule_report(
    request: ScheduleReportRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    调度定期报告
    
    支持每日、每周等定期报告
    """
    try:
        if request.frequency == "daily":
            background_tasks.add_task(
                report_scheduler.schedule_daily_report,
                request.experiment_id,
                request.send_time,
                request.recipients
            )
            
        return {
            "success": True,
            "message": f"已调度{request.frequency}报告",
            "experiment_id": request.experiment_id,
            "recipients": request.recipients
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{experiment_id}")
async def export_report(
    experiment_id: str,
    format: ReportFormat = Query(ReportFormat.PDF, description="导出格式")
) -> Dict[str, Any]:
    """
    导出报告为指定格式
    
    支持PDF、HTML等格式导出
    """
    try:
        report = await report_service.generate_report(
            experiment_id=experiment_id,
            format=format
        )
        
        # 根据格式返回不同的响应
        if format == ReportFormat.PDF:
            return {
                "success": True,
                "download_url": f"/reports/download/{experiment_id}.pdf",
                "expires_at": datetime.utcnow().isoformat()
            }
        elif format == ReportFormat.HTML:
            return {
                "success": True,
                "html_content": report.get("html", ""),
                "preview_url": f"/reports/preview/{experiment_id}"
            }
        else:
            return {
                "success": True,
                "data": report
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate")
async def batch_generate_reports(request: BatchReportRequest) -> Dict[str, Any]:
    """
    批量生成报告
    
    一次性生成多个实验的报告
    """
    try:
        reports = {}
        
        for exp_id in request.experiment_ids:
            reports[exp_id] = await report_service.generate_report(
                experiment_id=exp_id,
                format=request.format
            )
            
        # 如果需要合并结果
        if request.merge_results:
            merged_report = _merge_reports(reports)
            return {
                "success": True,
                "merged_report": merged_report,
                "individual_reports": reports
            }
        else:
            return {
                "success": True,
                "reports": reports
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates() -> Dict[str, Any]:
    """
    列出可用的报告模板
    """
    templates = [
        {
            "id": "standard",
            "name": "标准报告",
            "description": "包含所有标准章节的完整报告",
            "sections": list(ReportSection)
        },
        {
            "id": "executive",
            "name": "执行摘要",
            "description": "仅包含执行摘要和关键结果",
            "sections": [ReportSection.EXECUTIVE_SUMMARY, ReportSection.RECOMMENDATIONS]
        },
        {
            "id": "technical",
            "name": "技术报告",
            "description": "包含详细统计分析的技术报告",
            "sections": [
                ReportSection.EXPERIMENT_OVERVIEW,
                ReportSection.METRIC_RESULTS,
                ReportSection.STATISTICAL_ANALYSIS,
                ReportSection.APPENDIX
            ]
        }
    ]
    
    return {
        "success": True,
        "templates": templates
    }


@router.post("/templates")
async def create_template(request: ReportTemplateRequest) -> Dict[str, Any]:
    """
    创建自定义报告模板
    """
    # 这里应该保存到数据库
    template = {
        "id": request.name.lower().replace(" ", "_"),
        "name": request.name,
        "description": request.description,
        "sections": request.sections,
        "default_format": request.default_format,
        "custom_settings": request.custom_settings,
        "created_at": datetime.utcnow().isoformat()
    }
    
    return {
        "success": True,
        "template": template
    }


@router.get("/preview/{experiment_id}")
async def preview_report(experiment_id: str) -> Dict[str, Any]:
    """
    预览报告
    
    生成报告预览，不保存
    """
    try:
        report = await report_service.generate_report(
            experiment_id=experiment_id,
            sections=[
                ReportSection.EXECUTIVE_SUMMARY,
                ReportSection.METRIC_RESULTS
            ],
            format=ReportFormat.HTML
        )
        
        return {
            "success": True,
            "preview": report.get("html", ""),
            "experiment_id": experiment_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_experiments(
    experiment_ids: List[str] = Query(..., description="要比较的实验ID列表")
) -> Dict[str, Any]:
    """
    比较多个实验的结果
    
    生成对比报告
    """
    try:
        comparison_data = {}
        
        for exp_id in experiment_ids:
            report = await report_service.generate_report(
                experiment_id=exp_id,
                sections=[ReportSection.METRIC_RESULTS],
                format=ReportFormat.JSON
            )
            comparison_data[exp_id] = report.get("metric_results", {})
            
        # 生成比较表
        comparison_table = _generate_comparison_table(comparison_data)
        
        return {
            "success": True,
            "comparison": comparison_table,
            "experiment_ids": experiment_ids
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _merge_reports(reports: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个报告"""
    merged = {
        "experiments": [],
        "overall_metrics": {},
        "combined_recommendations": []
    }
    
    for exp_id, report in reports.items():
        if "executive_summary" in report:
            merged["experiments"].append({
                "experiment_id": exp_id,
                "summary": report["executive_summary"]
            })
            
        if "recommendations" in report:
            merged["combined_recommendations"].extend(
                report["recommendations"].get("recommendations", [])
            )
            
    return merged


def _generate_comparison_table(comparison_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成比较表"""
    table = []
    
    # 提取所有指标名称
    all_metrics = set()
    for exp_data in comparison_data.values():
        for category in ["primary_metrics", "secondary_metrics"]:
            for metric in exp_data.get(category, []):
                all_metrics.add(metric["name"])
                
    # 为每个指标生成比较行
    for metric_name in all_metrics:
        row = {"metric": metric_name}
        
        for exp_id, exp_data in comparison_data.items():
            # 查找该指标在实验中的值
            metric_value = None
            for category in ["primary_metrics", "secondary_metrics"]:
                for metric in exp_data.get(category, []):
                    if metric["name"] == metric_name:
                        metric_value = {
                            "lift": metric.get("relative_diff", "N/A"),
                            "significant": metric.get("significant", False)
                        }
                        break
                        
            row[exp_id] = metric_value or {"lift": "N/A", "significant": False}
            
        table.append(row)
        
    return table


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "service": "report_generation",
        "status": "healthy"
    }