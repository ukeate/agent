"""
统计分析API端点 - 提供基础统计计算功能
"""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field, field_validator, model_validator
from src.api.base_model import ApiBaseModel
from src.services.statistical_analysis_service import (
    DescriptiveStats,
    DistributionType,
    GroupStats,
    MetricType,
    get_stats_calculator,
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/statistical-analysis", tags=["统计分析"])

# 请求模型
class BasicStatsRequest(ApiBaseModel):
    """基础统计计算请求"""
    values: List[Union[int, float]] = Field(..., min_items=1, description="数值列表")
    calculate_advanced: bool = Field(True, description="是否计算高级统计指标（偏度、峰度）")
    
    @field_validator('values')
    def validate_values(cls, v):
        if not v:
            raise ValueError("Values list cannot be empty")
        for val in v:
            if not isinstance(val, (int, float)):
                raise ValueError("All values must be numbers")
        return v

class ConversionStatsRequest(ApiBaseModel):
    """转化率统计计算请求"""
    conversions: int = Field(..., ge=0, description="转化用户数")
    total_users: int = Field(..., gt=0, description="总用户数")
    
    @model_validator(mode="after")
    def validate_conversions(self):
        if self.conversions > self.total_users:
            raise ValueError("Conversions cannot exceed total_users")
        return self

class GroupData(ApiBaseModel):
    """分组数据"""
    name: str = Field(..., description="分组名称")
    values: Optional[List[Union[int, float]]] = Field(None, description="数值列表")
    conversions: Optional[int] = Field(None, ge=0, description="转化用户数")
    total_users: Optional[int] = Field(None, gt=0, description="总用户数")
    
    @model_validator(mode="after")
    def validate_conversions_with_total(self):
        if self.conversions is not None:
            if self.total_users is None:
                raise ValueError("total_users is required when conversions is provided")
            if self.conversions > self.total_users:
                raise ValueError("Conversions cannot exceed total_users")
        return self

class MultipleGroupsStatsRequest(ApiBaseModel):
    """多分组统计计算请求"""
    groups: Dict[str, GroupData] = Field(..., min_items=2, description="分组数据字典")
    metric_type: MetricType = Field(..., description="指标类型")
    
    @model_validator(mode="after")
    def validate_groups_data(self):
        for group_id, group_data in self.groups.items():
            if self.metric_type == MetricType.CONVERSION:
                if group_data.conversions is None or group_data.total_users is None:
                    raise ValueError(f"Conversion metric requires 'conversions' and 'total_users' for group {group_id}")
            else:
                if not group_data.values:
                    raise ValueError(f"Non-conversion metrics require 'values' list for group {group_id}")
        return self

class PercentileRequest(ApiBaseModel):
    """分位数计算请求"""
    values: List[Union[int, float]] = Field(..., min_items=1, description="数值列表")
    percentiles: List[float] = Field(..., min_items=1, description="分位数列表（0-100）")
    
    @field_validator('percentiles')
    def validate_percentiles(cls, v):
        for p in v:
            if not (0 <= p <= 100):
                raise ValueError(f"Percentile must be between 0 and 100, got {p}")
        return v

# 响应模型
class BasicStatsResponse(ApiBaseModel):
    """基础统计响应"""
    stats: Dict[str, Any] = Field(..., description="描述性统计结果")
    message: str = Field(default="Statistics calculated successfully")

class ConversionStatsResponse(ApiBaseModel):
    """转化率统计响应"""
    conversion_rate: float = Field(..., description="转化率")
    stats: Dict[str, Any] = Field(..., description="转化率统计结果")
    message: str = Field(default="Conversion statistics calculated successfully")

class MultipleGroupsStatsResponse(ApiBaseModel):
    """多分组统计响应"""
    groups_stats: Dict[str, Dict[str, Any]] = Field(..., description="各分组统计结果")
    summary: Dict[str, Any] = Field(..., description="汇总信息")
    message: str = Field(default="Multiple groups statistics calculated successfully")

class PercentileResponse(ApiBaseModel):
    """分位数响应"""
    percentiles: Dict[str, float] = Field(..., description="分位数结果")
    message: str = Field(default="Percentiles calculated successfully")

# API端点
@router.post("/basic-stats", response_model=BasicStatsResponse)
async def calculate_basic_statistics(request: BasicStatsRequest):
    """计算基础描述性统计"""
    try:
        calculator = get_stats_calculator()
        
        stats = calculator.basic_calculator.calculate_descriptive_stats(
            values=request.values,
            calculate_advanced=request.calculate_advanced
        )
        
        return BasicStatsResponse(
            stats=stats.to_dict(),
            message=f"Successfully calculated statistics for {len(request.values)} values"
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate basic statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Statistics calculation failed: {str(e)}")

@router.post("/conversion-stats", response_model=ConversionStatsResponse)
async def calculate_conversion_statistics(request: ConversionStatsRequest):
    """计算转化率统计"""
    try:
        calculator = get_stats_calculator()
        
        stats = calculator.basic_calculator.calculate_conversion_rate_stats(
            conversions=request.conversions,
            total_users=request.total_users
        )
        
        conversion_rate = request.conversions / request.total_users
        
        return ConversionStatsResponse(
            conversion_rate=conversion_rate,
            stats=stats.to_dict(),
            message=f"Conversion rate: {conversion_rate:.4f} ({request.conversions}/{request.total_users})"
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate conversion statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversion statistics calculation failed: {str(e)}")

@router.post("/percentiles", response_model=PercentileResponse)
async def calculate_percentiles(request: PercentileRequest):
    """计算分位数"""
    try:
        calculator = get_stats_calculator()
        
        percentile_values = calculator.basic_calculator.calculate_percentiles(
            values=request.values,
            percentiles=request.percentiles
        )
        
        # 构建结果字典
        percentiles_dict = {
            f"p{p}": value for p, value in zip(request.percentiles, percentile_values)
        }
        
        return PercentileResponse(
            percentiles=percentiles_dict,
            message=f"Successfully calculated {len(request.percentiles)} percentiles"
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate percentiles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Percentiles calculation failed: {str(e)}")

@router.post("/multiple-groups-stats", response_model=MultipleGroupsStatsResponse)
async def calculate_multiple_groups_statistics(request: MultipleGroupsStatsRequest):
    """计算多分组统计"""
    try:
        calculator = get_stats_calculator()
        
        # 转换请求数据格式
        groups_data = {}
        for group_id, group_data in request.groups.items():
            groups_data[group_id] = {
                "name": group_data.name,
                "values": group_data.values,
                "conversions": group_data.conversions,
                "total_users": group_data.total_users
            }
        
        # 计算分组统计
        groups_stats = calculator.calculate_multiple_groups_stats(
            groups_data=groups_data,
            metric_type=request.metric_type
        )
        
        # 构建响应数据
        groups_stats_dict = {
            group_id: group_stat.to_dict() 
            for group_id, group_stat in groups_stats.items()
        }
        
        # 生成汇总信息
        summary = {
            "total_groups": len(groups_stats_dict),
            "successful_groups": len([g for g in groups_stats_dict.values() if g]),
            "metric_type": request.metric_type.value,
            "groups_overview": {
                group_id: {
                    "name": stats["group_name"],
                    "sample_size": stats["stats"]["count"],
                    "mean": stats["stats"]["mean"]
                }
                for group_id, stats in groups_stats_dict.items()
            }
        }
        
        return MultipleGroupsStatsResponse(
            groups_stats=groups_stats_dict,
            summary=summary,
            message=f"Successfully calculated statistics for {len(groups_stats_dict)} groups"
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate multiple groups statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multiple groups statistics calculation failed: {str(e)}")

@router.get("/mean")
async def calculate_mean(values: List[float] = Query(..., description="数值列表")):
    """快速计算均值"""
    try:
        if not values:
            raise HTTPException(status_code=400, detail="Values list cannot be empty")
        
        calculator = get_stats_calculator()
        mean = calculator.basic_calculator.calculate_mean(values)
        
        return {
            "mean": mean,
            "count": len(values),
            "message": f"Mean calculated for {len(values)} values"
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate mean: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mean calculation failed: {str(e)}")

@router.get("/variance")
async def calculate_variance(
    values: List[float] = Query(..., description="数值列表"),
    sample: bool = Query(True, description="是否为样本方差")
):
    """快速计算方差"""
    try:
        if not values:
            raise HTTPException(status_code=400, detail="Values list cannot be empty")
        
        calculator = get_stats_calculator()
        variance = calculator.basic_calculator.calculate_variance(values, sample)
        std_dev = calculator.basic_calculator.calculate_std_deviation(values, sample)
        
        return {
            "variance": variance,
            "std_deviation": std_dev,
            "count": len(values),
            "sample": sample,
            "message": f"Variance calculated for {len(values)} values"
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate variance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Variance calculation failed: {str(e)}")

@router.get("/summary")
async def get_quick_summary(values: List[float] = Query(..., description="数值列表")):
    """获取数据快速摘要"""
    try:
        if not values:
            raise HTTPException(status_code=400, detail="Values list cannot be empty")
        
        calculator = get_stats_calculator()
        
        # 计算基础统计
        mean = calculator.basic_calculator.calculate_mean(values)
        std_dev = calculator.basic_calculator.calculate_std_deviation(values)
        percentiles = calculator.basic_calculator.calculate_percentiles(values, [25, 50, 75])
        
        return {
            "count": len(values),
            "mean": mean,
            "std_dev": std_dev,
            "min": min(values),
            "max": max(values),
            "median": percentiles[1],
            "q25": percentiles[0],
            "q75": percentiles[2],
            "range": max(values) - min(values),
            "message": "Quick summary generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate quick summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quick summary failed: {str(e)}")

@router.get("/health")
async def health_check():
    """统计分析服务健康检查"""
    try:
        # 简单的功能测试
        calculator = get_stats_calculator()
        test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        test_mean = calculator.basic_calculator.calculate_mean(test_values)
        
        return {
            "status": "healthy",
            "service": "statistical-analysis",
            "test_calculation": {
                "input": test_values,
                "mean": test_mean,
                "expected": 3.0,
                "passed": abs(test_mean - 3.0) < 1e-10
            },
            "message": "Statistical analysis service is running properly"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "statistical-analysis",
            "error": str(e),
            "message": "Statistical analysis service has issues"
        }
