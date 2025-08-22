"""
多重检验校正API端点
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from core.logging import get_logger
from services.multiple_testing_correction_service import (
    get_multiple_testing_correction_service,
    CorrectionMethod,
    ErrorRateType
)

logger = get_logger(__name__)
router = APIRouter(prefix="/multiple-testing", tags=["多重检验校正"])


# 请求模型
class CorrectionRequest(BaseModel):
    """多重检验校正请求"""
    pvalues: List[float] = Field(..., min_items=1, description="p值列表")
    method: CorrectionMethod = Field(..., description="校正方法")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    
    @validator('pvalues')
    def validate_pvalues(cls, v):
        for p in v:
            if not (0 <= p <= 1):
                raise ValueError(f"P-value {p} is not between 0 and 1")
        return v


class ComparisonRequest(BaseModel):
    """方法比较请求"""
    pvalues: List[float] = Field(..., min_items=1, description="p值列表")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    
    @validator('pvalues')
    def validate_pvalues(cls, v):
        for p in v:
            if not (0 <= p <= 1):
                raise ValueError(f"P-value {p} is not between 0 and 1")
        return v


class MethodRecommendationRequest(BaseModel):
    """方法推荐请求"""
    num_tests: int = Field(..., gt=0, description="检验数量")
    study_type: str = Field("exploratory", description="研究类型")
    independence: bool = Field(True, description="检验是否独立")
    
    @validator('study_type')
    def validate_study_type(cls, v):
        allowed = ["exploratory", "confirmatory"]
        if v not in allowed:
            raise ValueError(f"Study type must be one of {allowed}")
        return v


class PowerAdjustmentRequest(BaseModel):
    """功效调整请求"""
    original_power: float = Field(..., ge=0, le=1, description="原始统计功效")
    num_tests: int = Field(..., gt=0, description="检验数量")
    method: CorrectionMethod = Field(..., description="校正方法")


class ABTestMultipleComparisonRequest(BaseModel):
    """A/B测试多重比较请求"""
    comparison_pairs: List[Dict[str, Any]] = Field(..., min_items=2, description="比较对")
    correction_method: CorrectionMethod = Field(CorrectionMethod.HOLM, description="校正方法")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    
    @validator('comparison_pairs')
    def validate_pairs(cls, v):
        for pair in v:
            if 'p_value' not in pair:
                raise ValueError("Each comparison pair must have a 'p_value' field")
            if not (0 <= pair['p_value'] <= 1):
                raise ValueError(f"P-value {pair['p_value']} is not between 0 and 1")
        return v


# 响应模型
class CorrectionResponse(BaseModel):
    """校正响应"""
    result: Dict[str, Any] = Field(..., description="校正结果")
    summary: Dict[str, Any] = Field(..., description="结果摘要")
    interpretation: Dict[str, str] = Field(..., description="结果解释")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    message: str = Field(default="Correction completed successfully")


class ComparisonResponse(BaseModel):
    """方法比较响应"""
    results: Dict[str, Dict[str, Any]] = Field(..., description="各方法结果")
    comparison_summary: Dict[str, Any] = Field(..., description="比较摘要")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    message: str = Field(default="Comparison completed successfully")


class MethodRecommendationResponse(BaseModel):
    """方法推荐响应"""
    recommended_method: str = Field(..., description="推荐方法")
    reasoning: Dict[str, str] = Field(..., description="推荐理由")
    alternatives: List[str] = Field(default_factory=list, description="备选方法")
    message: str = Field(default="Method recommendation completed")


# 辅助函数
def _interpret_correction_result(result: Dict[str, Any]) -> Dict[str, str]:
    """解释校正结果"""
    interpretation = {}
    
    num_rejected = result.get("num_rejected", 0)
    num_tests = result.get("num_tests", 0)
    method = result.get("correction_method", "unknown")
    error_rate_type = result.get("error_rate_type", "unknown")
    
    # 基本统计
    interpretation["summary"] = f"在{num_tests}个检验中，有{num_rejected}个显著"
    
    # 错误率类型解释
    if error_rate_type == "fwer":
        interpretation["error_control"] = "控制家族错误率（FWER）"
    elif error_rate_type == "fdr":
        interpretation["error_control"] = "控制错误发现率（FDR）"
    else:
        interpretation["error_control"] = "每次比较错误率"
    
    # 方法特点
    method_descriptions = {
        "bonferroni": "最保守的方法，适用于少量关键检验",
        "holm": "改进的Bonferroni，保持FWER控制同时提高功效",
        "fdr_bh": "Benjamini-Hochberg方法，在探索性研究中有更高功效",
        "fdr_by": "更保守的FDR方法，不需要独立性假设",
        "sidak": "假设独立性，略微不如Bonferroni保守",
        "hochberg": "逐步向上方法，某些情况下比Holm更有功效"
    }
    
    interpretation["method_description"] = method_descriptions.get(
        method, "标准多重检验校正方法"
    )
    
    # 结果评价
    if num_rejected == 0:
        interpretation["result_assessment"] = "校正后无显著结果，可能需要更大样本量"
    elif num_rejected < num_tests * 0.1:
        interpretation["result_assessment"] = "少数检验显著，结果较为可靠"
    elif num_rejected < num_tests * 0.5:
        interpretation["result_assessment"] = "中等数量的显著结果"
    else:
        interpretation["result_assessment"] = "大量显著结果，注意验证重要发现"
    
    return interpretation


def _generate_correction_recommendations(result: Dict[str, Any], method: str) -> List[str]:
    """生成校正建议"""
    recommendations = []
    
    num_rejected = result.get("num_rejected", 0)
    num_tests = result.get("num_tests", 0)
    
    # 基于结果的建议
    if num_rejected == 0:
        recommendations.append("考虑使用更宽松的FDR方法或增加样本量")
        recommendations.append("检查原始效应量是否具有实际意义")
    
    # 基于方法的建议
    if method in ["bonferroni", "sidak"]:
        if num_tests > 10:
            recommendations.append("检验数量较多，考虑使用Holm或FDR方法提高功效")
    elif method in ["fdr_bh", "fdr_by"]:
        recommendations.append("FDR方法适合探索性分析，重要发现需要独立验证")
    
    # 通用建议
    if num_tests > 20:
        recommendations.append("大量多重比较，建议预先指定主要假设")
    
    recommendations.append("记录所有检验结果，包括不显著的")
    
    return recommendations


# API端点
@router.post("/apply-correction", response_model=CorrectionResponse)
async def apply_correction(request: CorrectionRequest):
    """应用多重检验校正"""
    try:
        service = get_multiple_testing_correction_service()
        
        result = service.apply_correction(
            pvalues=request.pvalues,
            method=request.method,
            alpha=request.alpha
        )
        
        # 创建摘要
        summary = {
            "total_tests": result.num_tests,
            "significant_tests": result.num_rejected,
            "rejection_rate": result.num_rejected / result.num_tests if result.num_tests > 0 else 0,
            "method": request.method.value,
            "alpha": request.alpha,
            "adjusted_alpha": result.adjusted_alpha
        }
        
        # 解释和建议
        interpretation = _interpret_correction_result(result.to_dict())
        recommendations = _generate_correction_recommendations(result.to_dict(), request.method.value)
        
        return CorrectionResponse(
            result=result.to_dict(),
            summary=summary,
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"{request.method.value}校正完成，{result.num_rejected}/{result.num_tests}个检验显著"
        )
        
    except Exception as e:
        logger.error(f"Correction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Correction failed: {str(e)}")


@router.post("/compare-methods", response_model=ComparisonResponse)
async def compare_correction_methods(request: ComparisonRequest):
    """比较不同校正方法"""
    try:
        service = get_multiple_testing_correction_service()
        
        results = service.compare_methods(
            pvalues=request.pvalues,
            alpha=request.alpha
        )
        
        # 转换结果格式
        results_dict = {}
        for method_name, result in results.items():
            results_dict[method_name] = {
                "num_rejected": result.num_rejected,
                "corrected_pvalues": result.corrected_pvalues[:5],  # 只显示前5个
                "error_rate_type": result.error_rate_type.value
            }
        
        # 创建比较摘要
        comparison_summary = {
            "total_tests": len(request.pvalues),
            "alpha": request.alpha,
            "method_rankings": sorted(
                [(method, res.num_rejected) for method, res in results.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            "most_conservative": min(results.items(), key=lambda x: x[1].num_rejected)[0],
            "most_liberal": max(results.items(), key=lambda x: x[1].num_rejected)[0]
        }
        
        # 生成建议
        recommendations = [
            f"最保守方法：{comparison_summary['most_conservative']}",
            f"最宽松方法：{comparison_summary['most_liberal']}",
            "FWER方法（Bonferroni、Holm）适合验证性研究",
            "FDR方法（BH、BY）适合探索性研究",
            "根据研究目的和错误控制需求选择合适方法"
        ]
        
        return ComparisonResponse(
            results=results_dict,
            comparison_summary=comparison_summary,
            recommendations=recommendations,
            message=f"比较了{len(results)}种校正方法"
        )
        
    except Exception as e:
        logger.error(f"Method comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/recommend-method", response_model=MethodRecommendationResponse)
async def recommend_correction_method(request: MethodRecommendationRequest):
    """推荐校正方法"""
    try:
        service = get_multiple_testing_correction_service()
        
        recommended = service.recommend_method(
            num_tests=request.num_tests,
            study_type=request.study_type,
            independence=request.independence
        )
        
        # 生成推荐理由
        reasoning = {}
        
        if request.study_type == "confirmatory":
            reasoning["study_type"] = "验证性研究需要严格控制FWER"
        else:
            reasoning["study_type"] = "探索性研究可以使用FDR方法获得更高功效"
        
        if request.num_tests <= 5:
            reasoning["num_tests"] = "检验数量较少，Bonferroni方法功效损失可接受"
        elif request.num_tests <= 20:
            reasoning["num_tests"] = "中等数量检验，推荐逐步调整方法"
        else:
            reasoning["num_tests"] = "大量检验，需要平衡错误控制和统计功效"
        
        if not request.independence:
            reasoning["independence"] = "检验不独立，需要更保守的方法"
        
        # 备选方法
        alternatives = []
        if recommended == CorrectionMethod.HOLM:
            alternatives = [CorrectionMethod.BONFERRONI.value, CorrectionMethod.HOCHBERG.value]
        elif recommended == CorrectionMethod.FDR_BH:
            alternatives = [CorrectionMethod.FDR_BY.value, CorrectionMethod.HOLM.value]
        elif recommended == CorrectionMethod.BONFERRONI:
            alternatives = [CorrectionMethod.HOLM.value, CorrectionMethod.SIDAK.value]
        
        return MethodRecommendationResponse(
            recommended_method=recommended.value,
            reasoning=reasoning,
            alternatives=alternatives,
            message=f"推荐使用{recommended.value}方法"
        )
        
    except Exception as e:
        logger.error(f"Method recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.post("/adjust-power", response_model=Dict[str, Any])
async def calculate_adjusted_power(request: PowerAdjustmentRequest):
    """计算校正后的统计功效"""
    try:
        service = get_multiple_testing_correction_service()
        
        adjusted_power = service.calculate_adjusted_power(
            original_power=request.original_power,
            num_tests=request.num_tests,
            method=request.method
        )
        
        power_loss = request.original_power - adjusted_power
        
        return {
            "original_power": request.original_power,
            "adjusted_power": adjusted_power,
            "power_loss": power_loss,
            "power_loss_percentage": power_loss / request.original_power * 100 if request.original_power > 0 else 0,
            "method": request.method.value,
            "num_tests": request.num_tests,
            "interpretation": {
                "power_assessment": "充足" if adjusted_power >= 0.8 else "不足",
                "recommendation": "可接受" if power_loss < 0.2 else "考虑增加样本量"
            },
            "message": f"校正后功效为{adjusted_power:.3f}"
        }
        
    except Exception as e:
        logger.error(f"Power adjustment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Power adjustment failed: {str(e)}")


@router.post("/ab-test-multiple-comparison", response_model=CorrectionResponse)
async def handle_ab_test_multiple_comparisons(request: ABTestMultipleComparisonRequest):
    """处理A/B测试的多重比较"""
    try:
        service = get_multiple_testing_correction_service()
        
        # 提取p值
        pvalues = [pair['p_value'] for pair in request.comparison_pairs]
        
        # 应用校正
        result = service.apply_correction(
            pvalues=pvalues,
            method=request.correction_method,
            alpha=request.alpha
        )
        
        # 将校正结果映射回比较对
        corrected_pairs = []
        for i, pair in enumerate(request.comparison_pairs):
            corrected_pair = pair.copy()
            corrected_pair['corrected_p_value'] = result.corrected_pvalues[i]
            corrected_pair['rejected'] = result.rejected[i]
            corrected_pairs.append(corrected_pair)
        
        # 创建摘要
        summary = {
            "total_comparisons": len(request.comparison_pairs),
            "significant_comparisons": result.num_rejected,
            "method": request.correction_method.value,
            "alpha": request.alpha,
            "corrected_pairs": corrected_pairs
        }
        
        # 解释和建议
        interpretation = _interpret_correction_result(result.to_dict())
        
        recommendations = [
            "A/B测试多重比较需要谨慎解释",
            f"使用{request.correction_method.value}方法控制错误率",
            "关注效应量和实际业务意义，而不仅是统计显著性"
        ]
        
        if result.num_rejected == 0:
            recommendations.append("无显著差异，考虑延长实验时间或增加样本量")
        else:
            recommendations.append(f"发现{result.num_rejected}个显著差异，建议进一步验证")
        
        return CorrectionResponse(
            result=result.to_dict(),
            summary=summary,
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"A/B测试多重比较校正完成"
        )
        
    except Exception as e:
        logger.error(f"A/B test multiple comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A/B test comparison failed: {str(e)}")


@router.get("/correction-methods")
async def get_correction_methods():
    """获取所有可用的校正方法"""
    try:
        methods = {
            "fwer_methods": {
                "bonferroni": {
                    "name": "Bonferroni校正",
                    "description": "最保守的FWER控制方法",
                    "formula": "p_adjusted = p * n",
                    "use_case": "少量重要假设检验"
                },
                "holm": {
                    "name": "Holm-Bonferroni校正",
                    "description": "逐步调整的Bonferroni改进",
                    "formula": "p_adjusted_i = p_i * (n - i + 1)",
                    "use_case": "中等数量的假设检验"
                },
                "sidak": {
                    "name": "Šidák校正",
                    "description": "基于独立性假设的方法",
                    "formula": "p_adjusted = 1 - (1 - p)^n",
                    "use_case": "独立检验"
                },
                "hochberg": {
                    "name": "Hochberg校正",
                    "description": "逐步向上的方法",
                    "formula": "从大到小检查p值",
                    "use_case": "某些条件下比Holm更有功效"
                }
            },
            "fdr_methods": {
                "fdr_bh": {
                    "name": "Benjamini-Hochberg FDR",
                    "description": "控制错误发现率",
                    "formula": "p_adjusted_i = p_i * n / i",
                    "use_case": "探索性研究，大量检验"
                },
                "fdr_by": {
                    "name": "Benjamini-Yekutieli FDR",
                    "description": "更保守的FDR方法",
                    "formula": "p_adjusted_i = p_i * n * C(n) / i",
                    "use_case": "非独立检验的FDR控制"
                }
            },
            "guidelines": {
                "sample_size": {
                    "small": "< 5个检验：Bonferroni",
                    "medium": "5-20个检验：Holm",
                    "large": "> 20个检验：FDR方法"
                },
                "study_type": {
                    "confirmatory": "使用FWER方法",
                    "exploratory": "使用FDR方法"
                }
            }
        }
        
        return {
            "methods": methods,
            "message": "校正方法信息获取成功"
        }
        
    except Exception as e:
        logger.error(f"Failed to get correction methods: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get methods: {str(e)}")


@router.get("/health")
async def health_check():
    """多重检验校正服务健康检查"""
    try:
        service = get_multiple_testing_correction_service()
        
        # 简单的功能测试
        test_pvalues = [0.01, 0.04, 0.03, 0.05, 0.20]
        test_result = service.apply_correction(
            pvalues=test_pvalues,
            method=CorrectionMethod.HOLM,
            alpha=0.05
        )
        
        return {
            "status": "healthy",
            "service": "multiple-testing-correction",
            "test_calculation": {
                "input_pvalues": test_pvalues,
                "method": "holm",
                "num_rejected": test_result.num_rejected,
                "passed": test_result.num_tests == len(test_pvalues)
            },
            "message": "Multiple testing correction service is running properly"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "multiple-testing-correction",
            "error": str(e),
            "message": "Multiple testing correction service has issues"
        }