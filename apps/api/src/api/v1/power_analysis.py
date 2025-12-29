"""
统计功效和样本量计算API端点
"""

from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import Field, field_validator, ValidationInfo
from src.services.power_analysis_service import (
    get_power_analysis_service,
    PowerAnalysisType,
    TestType,
    AlternativeHypothesis
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/power-analysis", tags=["统计功效分析"])

# 请求模型
class PowerCalculationRequest(ApiBaseModel):
    """统计功效计算请求"""
    test_type: TestType = Field(..., description="检验类型")
    effect_size: float = Field(..., gt=0, description="效应量")
    sample_size: Union[int, List[int]] = Field(..., description="样本量")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    alternative: AlternativeHypothesis = Field(AlternativeHypothesis.TWO_SIDED, description="备择假设类型")
    
    @field_validator('sample_size')
    def validate_sample_size(cls, v):
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("For two-sample tests, provide exactly 2 sample sizes")
            if any(n <= 0 for n in v):
                raise ValueError("All sample sizes must be positive")
        elif v <= 0:
            raise ValueError("Sample size must be positive")
        return v

class SampleSizeCalculationRequest(ApiBaseModel):
    """样本量计算请求"""
    test_type: TestType = Field(..., description="检验类型")
    effect_size: float = Field(..., gt=0, description="效应量")
    power: float = Field(0.8, ge=0.5, le=0.99, description="期望统计功效")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    alternative: AlternativeHypothesis = Field(AlternativeHypothesis.TWO_SIDED, description="备择假设类型")
    ratio: float = Field(1.0, gt=0, description="样本量比例（n2/n1）")

class EffectSizeCalculationRequest(ApiBaseModel):
    """效应量计算请求"""
    test_type: TestType = Field(..., description="检验类型")
    sample_size: Union[int, List[int]] = Field(..., description="样本量")
    power: float = Field(0.8, ge=0.5, le=0.99, description="期望统计功效")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    alternative: AlternativeHypothesis = Field(AlternativeHypothesis.TWO_SIDED, description="备择假设类型")

class ProportionPowerRequest(ApiBaseModel):
    """比例检验功效分析请求"""
    test_type: TestType = Field(..., description="检验类型")
    p1: float = Field(..., ge=0, le=1, description="第一组比例")
    p2: float = Field(..., ge=0, le=1, description="第二组比例或假设值")
    sample_size: Union[int, List[int]] = Field(..., description="样本量")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    alternative: AlternativeHypothesis = Field(AlternativeHypothesis.TWO_SIDED, description="备择假设类型")

class ProportionSampleSizeRequest(ApiBaseModel):
    """比例检验样本量计算请求"""
    test_type: TestType = Field(..., description="检验类型")
    p1: float = Field(..., ge=0, le=1, description="第一组比例")
    p2: float = Field(..., ge=0, le=1, description="第二组比例")
    power: float = Field(0.8, ge=0.5, le=0.99, description="期望统计功效")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    alternative: AlternativeHypothesis = Field(AlternativeHypothesis.TWO_SIDED, description="备择假设类型")
    ratio: float = Field(1.0, gt=0, description="样本量比例")

class ABTestSampleSizeRequest(ApiBaseModel):
    """A/B测试样本量计算请求"""
    baseline_conversion_rate: float = Field(..., ge=0, le=1, description="基准转化率")
    minimum_detectable_effect: float = Field(..., gt=0, le=1, description="最小可检测效应（相对提升）")
    power: float = Field(0.8, ge=0.5, le=0.99, description="期望统计功效")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    alternative: AlternativeHypothesis = Field(AlternativeHypothesis.TWO_SIDED, description="备择假设类型")
    
    @field_validator('minimum_detectable_effect')
    def validate_effect(cls, v, info: ValidationInfo):
        baseline = info.data.get('baseline_conversion_rate', 0)
        if baseline + (baseline * v) > 1:
            raise ValueError("Treatment conversion rate cannot exceed 1")
        return v

# 响应模型
class PowerAnalysisResponse(ApiBaseModel):
    """功效分析响应"""
    result: Dict[str, Any] = Field(..., description="分析结果")
    interpretation: Dict[str, str] = Field(..., description="结果解释")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    message: str = Field(default="Analysis completed successfully")

class ABTestSampleSizeResponse(ApiBaseModel):
    """A/B测试样本量响应"""
    sample_size_analysis: Dict[str, Any] = Field(..., description="样本量分析结果")
    experimental_design: Dict[str, Any] = Field(..., description="实验设计建议")
    duration_estimates: Dict[str, int] = Field(..., description="实验持续时间估计")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    message: str = Field(default="A/B test sample size calculation completed")

# 辅助函数
def _interpret_power_result(result: Dict[str, Any]) -> Dict[str, str]:
    """解释功效分析结果"""
    interpretation = {}
    
    power = result.get("power", 0)
    effect_size = result.get("effect_size", 0)
    sample_size = result.get("sample_size")
    
    # 功效解释
    if power >= 0.8:
        interpretation["power_assessment"] = "统计功效充足（≥80%）"
    elif power >= 0.6:
        interpretation["power_assessment"] = "统计功效中等（60-80%）"
    else:
        interpretation["power_assessment"] = "统计功效不足（<60%）"
    
    # 效应量解释
    if effect_size < 0.2:
        interpretation["effect_size_assessment"] = "小效应量"
    elif effect_size < 0.5:
        interpretation["effect_size_assessment"] = "中等效应量"
    elif effect_size < 0.8:
        interpretation["effect_size_assessment"] = "大效应量"
    else:
        interpretation["effect_size_assessment"] = "非常大的效应量"
    
    # 样本量解释
    if isinstance(sample_size, (list, tuple)):
        total_n = sum(sample_size)
    else:
        total_n = sample_size
    
    if total_n < 30:
        interpretation["sample_size_assessment"] = "小样本"
    elif total_n < 100:
        interpretation["sample_size_assessment"] = "中等样本"
    elif total_n < 1000:
        interpretation["sample_size_assessment"] = "大样本"
    else:
        interpretation["sample_size_assessment"] = "非常大的样本"
    
    return interpretation

def _generate_power_recommendations(result: Dict[str, Any], analysis_type: str) -> List[str]:
    """生成功效分析建议"""
    recommendations = []
    
    power = result.get("power", 0)
    effect_size = result.get("effect_size", 0)
    
    if analysis_type == "power":
        if power < 0.8:
            recommendations.append("统计功效不足，建议增加样本量或提高效应量")
        else:
            recommendations.append("统计功效充足，实验设计合理")
    
    elif analysis_type == "sample_size":
        sample_size = result.get("sample_size")
        if isinstance(sample_size, (list, tuple)):
            total_n = sum(sample_size)
        else:
            total_n = sample_size
        
        if total_n > 10000:
            recommendations.append("所需样本量较大，考虑是否可以接受更大的效应量")
        elif total_n < 30:
            recommendations.append("所需样本量较小，实验容易执行")
    
    if effect_size < 0.2:
        recommendations.append("效应量较小，需要大样本量才能检测到差异")
    elif effect_size > 1.0:
        recommendations.append("效应量很大，相对容易检测到差异")
    
    recommendations.append("建议在实际实验中监控中间结果，必要时调整设计")
    
    return recommendations

# API端点
@router.post("/calculate-power", response_model=PowerAnalysisResponse)
async def calculate_power(request: PowerCalculationRequest):
    """计算统计功效"""
    try:
        service = get_power_analysis_service()
        
        # 转换样本量格式
        sample_size = tuple(request.sample_size) if isinstance(request.sample_size, list) else request.sample_size
        
        result = service.run_power_analysis(
            analysis_type=PowerAnalysisType.POWER,
            test_type=request.test_type,
            effect_size=request.effect_size,
            sample_size=sample_size,
            alpha=request.alpha,
            alternative=request.alternative
        )
        
        interpretation = _interpret_power_result(result.to_dict())
        recommendations = _generate_power_recommendations(result.to_dict(), "power")
        
        return PowerAnalysisResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"统计功效计算完成，功效 = {result.power:.3f}"
        )
        
    except Exception as e:
        logger.error(f"Power calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Power calculation failed: {str(e)}")

@router.post("/calculate-sample-size", response_model=PowerAnalysisResponse)
async def calculate_sample_size(request: SampleSizeCalculationRequest):
    """计算所需样本量"""
    try:
        service = get_power_analysis_service()
        
        result = service.run_power_analysis(
            analysis_type=PowerAnalysisType.SAMPLE_SIZE,
            test_type=request.test_type,
            effect_size=request.effect_size,
            power=request.power,
            alpha=request.alpha,
            alternative=request.alternative,
            ratio=request.ratio
        )
        
        interpretation = _interpret_power_result(result.to_dict())
        recommendations = _generate_power_recommendations(result.to_dict(), "sample_size")
        
        sample_size_str = str(result.sample_size)
        
        return PowerAnalysisResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"样本量计算完成，所需样本量 = {sample_size_str}"
        )
        
    except Exception as e:
        logger.error(f"Sample size calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sample size calculation failed: {str(e)}")

@router.post("/calculate-effect-size", response_model=PowerAnalysisResponse)
async def calculate_detectable_effect_size(request: EffectSizeCalculationRequest):
    """计算可检测的最小效应量"""
    try:
        service = get_power_analysis_service()
        
        # 转换样本量格式
        sample_size = tuple(request.sample_size) if isinstance(request.sample_size, list) else request.sample_size
        
        result = service.run_power_analysis(
            analysis_type=PowerAnalysisType.EFFECT_SIZE,
            test_type=request.test_type,
            sample_size=sample_size,
            power=request.power,
            alpha=request.alpha,
            alternative=request.alternative
        )
        
        interpretation = _interpret_power_result(result.to_dict())
        recommendations = _generate_power_recommendations(result.to_dict(), "effect_size")
        
        return PowerAnalysisResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"最小可检测效应量计算完成，效应量 = {result.effect_size:.3f}"
        )
        
    except Exception as e:
        logger.error(f"Effect size calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Effect size calculation failed: {str(e)}")

@router.post("/proportion-power", response_model=PowerAnalysisResponse)
async def calculate_proportion_power(request: ProportionPowerRequest):
    """计算比例检验统计功效"""
    try:
        if request.test_type not in {TestType.ONE_PROPORTION, TestType.TWO_PROPORTIONS}:
            raise HTTPException(status_code=400, detail="test_type必须为one_proportion或two_proportions")
        service = get_power_analysis_service()
        
        # 转换样本量格式
        sample_size = tuple(request.sample_size) if isinstance(request.sample_size, list) else request.sample_size
        
        result = service.run_power_analysis(
            analysis_type=PowerAnalysisType.POWER,
            test_type=request.test_type,
            p1=request.p1,
            p2=request.p2,
            sample_size=sample_size,
            alpha=request.alpha,
            alternative=request.alternative
        )
        
        interpretation = _interpret_power_result(result.to_dict())
        recommendations = _generate_power_recommendations(result.to_dict(), "power")
        
        return PowerAnalysisResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"比例检验功效计算完成，功效 = {result.power:.3f}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proportion power calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Proportion power calculation failed: {str(e)}")

@router.post("/proportion-sample-size", response_model=PowerAnalysisResponse)
async def calculate_proportion_sample_size(request: ProportionSampleSizeRequest):
    """计算比例检验样本量"""
    try:
        if request.test_type not in {TestType.ONE_PROPORTION, TestType.TWO_PROPORTIONS}:
            raise HTTPException(status_code=400, detail="test_type必须为one_proportion或two_proportions")
        service = get_power_analysis_service()
        
        result = service.run_power_analysis(
            analysis_type=PowerAnalysisType.SAMPLE_SIZE,
            test_type=request.test_type,
            p1=request.p1,
            p2=request.p2,
            power=request.power,
            alpha=request.alpha,
            alternative=request.alternative,
            ratio=request.ratio
        )
        
        interpretation = _interpret_power_result(result.to_dict())
        recommendations = _generate_power_recommendations(result.to_dict(), "sample_size")
        
        sample_size_str = str(result.sample_size)
        
        return PowerAnalysisResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"比例检验样本量计算完成，所需样本量 = {sample_size_str}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proportion sample size calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Proportion sample size calculation failed: {str(e)}")

@router.post("/ab-test-sample-size", response_model=ABTestSampleSizeResponse)
async def calculate_ab_test_sample_size(request: ABTestSampleSizeRequest):
    """计算A/B测试样本量"""
    try:
        service = get_power_analysis_service()
        
        result = service.calculate_ab_test_sample_size(
            baseline_conversion_rate=request.baseline_conversion_rate,
            minimum_detectable_effect=request.minimum_detectable_effect,
            power=request.power,
            alpha=request.alpha,
            alternative=request.alternative
        )
        
        # 提取关键信息
        sample_size_analysis = {
            "control_group_size": result["control_group_size"],
            "treatment_group_size": result["treatment_group_size"],
            "total_sample_size": result["total_sample_size"],
            "power_analysis": result["power_analysis_result"]
        }
        
        experimental_design = {
            "baseline_conversion_rate": result["baseline_conversion_rate"],
            "treatment_conversion_rate": result["treatment_conversion_rate"],
            "relative_effect": result["relative_effect"],
            "absolute_effect": result["absolute_effect"],
            "alpha": request.alpha,
            "power": request.power
        }
        
        return ABTestSampleSizeResponse(
            sample_size_analysis=sample_size_analysis,
            experimental_design=experimental_design,
            duration_estimates=result["estimated_duration_days"],
            recommendations=result["recommendations"],
            message=f"A/B测试样本量计算完成，总样本量 = {result['total_sample_size']}"
        )
        
    except Exception as e:
        logger.error(f"A/B test sample size calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A/B test sample size calculation failed: {str(e)}")

@router.get("/effect-size-guidelines")
async def get_effect_size_guidelines():
    """获取效应量参考指南"""
    try:
        guidelines = {
            "cohens_d": {
                "description": "Cohen's d用于比较两组均值",
                "small": {"value": 0.2, "description": "小效应量"},
                "medium": {"value": 0.5, "description": "中等效应量"},
                "large": {"value": 0.8, "description": "大效应量"}
            },
            "cohens_h": {
                "description": "Cohen's h用于比较两个比例",
                "small": {"value": 0.2, "description": "小效应量"},
                "medium": {"value": 0.5, "description": "中等效应量"},
                "large": {"value": 0.8, "description": "大效应量"}
            },
            "practical_guidelines": {
                "conversion_rate_improvements": {
                    "minimal": "1-2%相对提升",
                    "small": "5-10%相对提升",
                    "medium": "10-20%相对提升",
                    "large": "20%以上相对提升"
                },
                "business_metrics": {
                    "revenue_per_user": "通常5-15%的提升具有商业价值",
                    "engagement_metrics": "10-30%的提升较为常见",
                    "retention_rates": "5-10%的提升通常具有重要意义"
                }
            },
            "sample_size_recommendations": {
                "minimum_per_group": 30,
                "recommended_minimum": 100,
                "for_small_effects": 1000,
                "for_conversion_rates": {
                    "low_baseline": "基准转化率<5%时，需要更大样本量",
                    "high_baseline": "基准转化率>20%时，样本量要求相对较低"
                }
            }
        }
        
        return {
            "guidelines": guidelines,
            "message": "效应量参考指南获取成功"
        }
        
    except Exception as e:
        logger.error(f"Failed to get effect size guidelines: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get guidelines: {str(e)}")

@router.get("/sample-size-calculator")
async def get_sample_size_calculator_info():
    """获取样本量计算器信息和使用指南"""
    try:
        info = {
            "supported_tests": [
                {
                    "test_type": "one_sample_t",
                    "description": "单样本t检验",
                    "use_case": "比较样本均值与已知总体均值"
                },
                {
                    "test_type": "two_sample_t",
                    "description": "双样本t检验",
                    "use_case": "比较两组连续数据的均值"
                },
                {
                    "test_type": "paired_t",
                    "description": "配对t检验",
                    "use_case": "比较同一对象的前后测量值"
                },
                {
                    "test_type": "one_proportion",
                    "description": "单比例检验",
                    "use_case": "比较样本比例与已知总体比例"
                },
                {
                    "test_type": "two_proportions",
                    "description": "双比例检验",
                    "use_case": "比较两组的转化率或成功率"
                }
            ],
            "parameters": {
                "effect_size": "效应量，表示实际差异的大小",
                "power": "统计功效，通常设为0.8（80%）",
                "alpha": "显著性水平，通常设为0.05（5%）",
                "alternative": "备择假设类型：双边、单边"
            },
            "usage_tips": [
                "先确定要检测的最小重要差异",
                "根据业务需求设定合适的功效和显著性水平",
                "考虑实际可获得的样本量限制",
                "对于A/B测试，建议功效≥80%，显著性水平≤5%"
            ]
        }
        
        return {
            "calculator_info": info,
            "message": "样本量计算器信息获取成功"
        }
        
    except Exception as e:
        logger.error(f"Failed to get calculator info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get calculator info: {str(e)}")

@router.get("/health")
async def health_check():
    """功效分析服务健康检查"""
    try:
        service = get_power_analysis_service()
        
        # 简单的功能测试
        test_result = service.run_power_analysis(
            analysis_type=PowerAnalysisType.POWER,
            test_type=TestType.TWO_SAMPLE_T,
            effect_size=0.5,
            sample_size=64,  # 每组32个样本
            alpha=0.05
        )
        
        return {
            "status": "healthy",
            "service": "power-analysis",
            "test_calculation": {
                "effect_size": float(test_result.effect_size) if test_result.effect_size is not None else None,
                "power": float(test_result.power) if test_result.power is not None else None,
                "sample_size": int(test_result.sample_size) if test_result.sample_size is not None else None,
                "passed": bool(0.7 <= float(test_result.power) <= 0.9) if test_result.power is not None else False
            },
            "message": "Power analysis service is running properly"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "power-analysis",
            "error": str(e),
            "message": "Power analysis service has issues"
        }
