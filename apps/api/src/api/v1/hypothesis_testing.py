"""
假设检验API端点 - 提供t检验、卡方检验等统计推断功能
"""
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from core.logging import get_logger
from services.hypothesis_testing_service import (
    get_hypothesis_testing_service,
    HypothesisType,
    HypothesisTestResult
)
from services.statistical_analysis_service import MetricType

logger = get_logger(__name__)
router = APIRouter(prefix="/hypothesis-testing", tags=["假设检验"])


# 请求模型
class OneSampleTTestRequest(BaseModel):
    """单样本t检验请求"""
    sample: List[float] = Field(..., min_items=2, description="样本数据")
    population_mean: float = Field(..., description="总体均值（零假设值）")
    hypothesis_type: HypothesisType = Field(HypothesisType.TWO_SIDED, description="假设检验类型")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")


class TwoSampleTTestRequest(BaseModel):
    """双样本t检验请求"""
    sample1: List[float] = Field(..., min_items=2, description="样本1数据")
    sample2: List[float] = Field(..., min_items=2, description="样本2数据")
    equal_variances: bool = Field(True, description="是否假定方差相等")
    hypothesis_type: HypothesisType = Field(HypothesisType.TWO_SIDED, description="假设检验类型")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")


class PairedTTestRequest(BaseModel):
    """配对t检验请求"""
    sample1: List[float] = Field(..., min_items=2, description="配对样本1")
    sample2: List[float] = Field(..., min_items=2, description="配对样本2")
    hypothesis_type: HypothesisType = Field(HypothesisType.TWO_SIDED, description="假设检验类型")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    
    @validator('sample2')
    def validate_paired_samples(cls, v, values):
        sample1 = values.get('sample1', [])
        if len(v) != len(sample1):
            raise ValueError("Paired samples must have the same length")
        return v


class ChiSquareGoodnessOfFitRequest(BaseModel):
    """卡方拟合优度检验请求"""
    observed: List[int] = Field(..., min_items=2, description="观测频数")
    expected: List[float] = Field(..., min_items=2, description="期望频数")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    
    @validator('expected')
    def validate_expected_frequencies(cls, v, values):
        observed = values.get('observed', [])
        if len(v) != len(observed):
            raise ValueError("Observed and expected frequencies must have the same length")
        if any(e <= 0 for e in v):
            raise ValueError("All expected frequencies must be positive")
        return v


class ChiSquareIndependenceRequest(BaseModel):
    """卡方独立性检验请求"""
    contingency_table: List[List[int]] = Field(..., description="列联表")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    
    @validator('contingency_table')
    def validate_contingency_table(cls, v):
        if len(v) < 2 or len(v[0]) < 2:
            raise ValueError("Contingency table must be at least 2x2")
        
        cols = len(v[0])
        if not all(len(row) == cols for row in v):
            raise ValueError("All rows must have the same number of columns")
        
        if any(any(cell < 0 for cell in row) for row in v):
            raise ValueError("All frequencies must be non-negative")
        
        return v


class TwoProportionTestRequest(BaseModel):
    """两比例检验请求"""
    successes1: int = Field(..., ge=0, description="组1成功数")
    total1: int = Field(..., gt=0, description="组1总数")
    successes2: int = Field(..., ge=0, description="组2成功数")
    total2: int = Field(..., gt=0, description="组2总数")
    hypothesis_type: HypothesisType = Field(HypothesisType.TWO_SIDED, description="假设检验类型")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    
    @validator('successes1')
    def validate_successes1(cls, v, values):
        total1 = values.get('total1', 0)
        if v > total1:
            raise ValueError("successes1 cannot exceed total1")
        return v
    
    @validator('successes2')
    def validate_successes2(cls, v, values):
        total2 = values.get('total2', 0)
        if v > total2:
            raise ValueError("successes2 cannot exceed total2")
        return v


class ABTestComparisonRequest(BaseModel):
    """A/B测试比较请求"""
    control_group: Dict[str, Any] = Field(..., description="对照组数据")
    treatment_group: Dict[str, Any] = Field(..., description="实验组数据")
    metric_type: MetricType = Field(..., description="指标类型")
    hypothesis_type: HypothesisType = Field(HypothesisType.TWO_SIDED, description="假设检验类型")
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="显著性水平")
    equal_variances: bool = Field(True, description="是否假定等方差（仅对连续指标有效）")
    
    @validator('control_group')
    def validate_control_group(cls, v, values):
        metric_type = values.get('metric_type')
        if metric_type == MetricType.CONVERSION:
            required_fields = ['conversions', 'total_users']
            if not all(field in v for field in required_fields):
                raise ValueError(f"Conversion metric requires {required_fields} in control_group")
        else:
            if 'values' not in v:
                raise ValueError("Non-conversion metrics require 'values' in control_group")
        return v
    
    @validator('treatment_group')
    def validate_treatment_group(cls, v, values):
        metric_type = values.get('metric_type')
        if metric_type == MetricType.CONVERSION:
            required_fields = ['conversions', 'total_users']
            if not all(field in v for field in required_fields):
                raise ValueError(f"Conversion metric requires {required_fields} in treatment_group")
        else:
            if 'values' not in v:
                raise ValueError("Non-conversion metrics require 'values' in treatment_group")
        return v


# 响应模型
class HypothesisTestResponse(BaseModel):
    """假设检验响应"""
    result: Dict[str, Any] = Field(..., description="检验结果")
    interpretation: Dict[str, str] = Field(..., description="结果解释")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    message: str = Field(default="Test completed successfully")


class ABTestComparisonResponse(BaseModel):
    """A/B测试比较响应"""
    test_result: Dict[str, Any] = Field(..., description="检验结果")
    control_stats: Dict[str, Any] = Field(..., description="对照组统计")
    treatment_stats: Dict[str, Any] = Field(..., description="实验组统计")
    practical_significance: Dict[str, Any] = Field(..., description="实际显著性分析")
    interpretation: Dict[str, str] = Field(..., description="结果解释")
    recommendations: List[str] = Field(default_factory=list, description="建议")
    message: str = Field(default="A/B test comparison completed successfully")


# 辅助函数
def _interpret_test_result(result: HypothesisTestResult, test_context: str = "") -> Dict[str, str]:
    """解释检验结果"""
    interpretation = {}
    
    # 基础结论
    if result.is_significant:
        interpretation["statistical_conclusion"] = f"在α={result.alpha}的显著性水平下，拒绝零假设，结果具有统计显著性"
    else:
        interpretation["statistical_conclusion"] = f"在α={result.alpha}的显著性水平下，未能拒绝零假设，结果不具有统计显著性"
    
    # p值解释
    if result.p_value < 0.001:
        interpretation["p_value_interpretation"] = "p值小于0.001，证据极强"
    elif result.p_value < 0.01:
        interpretation["p_value_interpretation"] = "p值小于0.01，证据很强"
    elif result.p_value < 0.05:
        interpretation["p_value_interpretation"] = "p值小于0.05，证据中等"
    elif result.p_value < 0.1:
        interpretation["p_value_interpretation"] = "p值小于0.1，证据较弱"
    else:
        interpretation["p_value_interpretation"] = "p值大于0.1，证据不足"
    
    # 效应量解释
    if result.effect_size is not None:
        if result.effect_size < 0.2:
            interpretation["effect_size_interpretation"] = "效应量很小"
        elif result.effect_size < 0.5:
            interpretation["effect_size_interpretation"] = "效应量小"
        elif result.effect_size < 0.8:
            interpretation["effect_size_interpretation"] = "效应量中等"
        else:
            interpretation["effect_size_interpretation"] = "效应量大"
    
    return interpretation


def _generate_recommendations(result: HypothesisTestResult, metric_type: MetricType = None) -> List[str]:
    """生成建议"""
    recommendations = []
    
    if result.is_significant:
        recommendations.append("检测到统计显著差异，建议进一步分析实际业务意义")
        if result.effect_size and result.effect_size < 0.2:
            recommendations.append("虽然统计显著，但效应量很小，需要评估实际业务价值")
    else:
        recommendations.append("未检测到统计显著差异，可能需要增加样本量或延长实验时间")
        if result.power and result.power < 0.8:
            recommendations.append("统计功效较低，建议增加样本量以提高检验功效")
    
    # 基于指标类型的建议
    if metric_type == MetricType.CONVERSION:
        if result.is_significant:
            recommendations.append("转化率存在显著差异，建议评估对业务指标的长期影响")
        else:
            recommendations.append("转化率差异不显著，建议检查实验设计和样本分布")
    
    return recommendations


# API端点
@router.post("/t-test/one-sample", response_model=HypothesisTestResponse)
async def one_sample_t_test(request: OneSampleTTestRequest):
    """单样本t检验"""
    try:
        service = get_hypothesis_testing_service()
        
        result = service.run_t_test(
            test_type="one_sample",
            sample=request.sample,
            population_mean=request.population_mean,
            hypothesis_type=request.hypothesis_type,
            alpha=request.alpha
        )
        
        interpretation = _interpret_test_result(result)
        recommendations = _generate_recommendations(result)
        
        return HypothesisTestResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"单样本t检验完成，样本量={len(request.sample)}"
        )
        
    except Exception as e:
        logger.error(f"One-sample t-test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"One-sample t-test failed: {str(e)}")


@router.post("/t-test/two-sample", response_model=HypothesisTestResponse)
async def two_sample_t_test(request: TwoSampleTTestRequest):
    """双样本t检验"""
    try:
        service = get_hypothesis_testing_service()
        
        result = service.run_t_test(
            test_type="independent_two_sample",
            sample1=request.sample1,
            sample2=request.sample2,
            equal_variances=request.equal_variances,
            hypothesis_type=request.hypothesis_type,
            alpha=request.alpha
        )
        
        interpretation = _interpret_test_result(result)
        recommendations = _generate_recommendations(result)
        
        test_name = "Student t检验" if request.equal_variances else "Welch t检验"
        
        return HypothesisTestResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"{test_name}完成，样本量={len(request.sample1)}和{len(request.sample2)}"
        )
        
    except Exception as e:
        logger.error(f"Two-sample t-test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Two-sample t-test failed: {str(e)}")


@router.post("/t-test/paired", response_model=HypothesisTestResponse)
async def paired_t_test(request: PairedTTestRequest):
    """配对t检验"""
    try:
        service = get_hypothesis_testing_service()
        
        result = service.run_t_test(
            test_type="paired",
            sample1=request.sample1,
            sample2=request.sample2,
            hypothesis_type=request.hypothesis_type,
            alpha=request.alpha
        )
        
        interpretation = _interpret_test_result(result)
        recommendations = _generate_recommendations(result)
        
        return HypothesisTestResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"配对t检验完成，配对样本量={len(request.sample1)}"
        )
        
    except Exception as e:
        logger.error(f"Paired t-test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Paired t-test failed: {str(e)}")


@router.post("/chi-square/goodness-of-fit", response_model=HypothesisTestResponse)
async def chi_square_goodness_of_fit(request: ChiSquareGoodnessOfFitRequest):
    """卡方拟合优度检验"""
    try:
        service = get_hypothesis_testing_service()
        
        result = service.run_chi_square_test(
            test_type="goodness_of_fit",
            observed=request.observed,
            expected=request.expected,
            alpha=request.alpha
        )
        
        interpretation = _interpret_test_result(result)
        recommendations = _generate_recommendations(result)
        
        return HypothesisTestResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"卡方拟合优度检验完成，类别数={len(request.observed)}"
        )
        
    except Exception as e:
        logger.error(f"Chi-square goodness of fit test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chi-square goodness of fit test failed: {str(e)}")


@router.post("/chi-square/independence", response_model=HypothesisTestResponse)
async def chi_square_independence(request: ChiSquareIndependenceRequest):
    """卡方独立性检验"""
    try:
        service = get_hypothesis_testing_service()
        
        result = service.run_chi_square_test(
            test_type="independence",
            contingency_table=request.contingency_table,
            alpha=request.alpha
        )
        
        interpretation = _interpret_test_result(result)
        recommendations = _generate_recommendations(result)
        
        rows = len(request.contingency_table)
        cols = len(request.contingency_table[0])
        
        return HypothesisTestResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"卡方独立性检验完成，列联表维度={rows}x{cols}"
        )
        
    except Exception as e:
        logger.error(f"Chi-square independence test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chi-square independence test failed: {str(e)}")


@router.post("/proportion-test", response_model=HypothesisTestResponse)
async def two_proportion_test(request: TwoProportionTestRequest):
    """两比例检验"""
    try:
        service = get_hypothesis_testing_service()
        
        result = service.run_chi_square_test(
            test_type="proportion",
            successes1=request.successes1,
            total1=request.total1,
            successes2=request.successes2,
            total2=request.total2,
            hypothesis_type=request.hypothesis_type,
            alpha=request.alpha
        )
        
        interpretation = _interpret_test_result(result)
        recommendations = _generate_recommendations(result, MetricType.CONVERSION)
        
        p1 = request.successes1 / request.total1
        p2 = request.successes2 / request.total2
        
        return HypothesisTestResponse(
            result=result.to_dict(),
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"两比例检验完成，比例分别为{p1:.4f}和{p2:.4f}"
        )
        
    except Exception as e:
        logger.error(f"Two proportion test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Two proportion test failed: {str(e)}")


@router.post("/ab-test-comparison", response_model=ABTestComparisonResponse)
async def ab_test_comparison(request: ABTestComparisonRequest):
    """A/B测试比较分析"""
    try:
        service = get_hypothesis_testing_service()
        stats_service = get_stats_calculator()
        
        # 执行假设检验
        test_result = service.compare_two_groups(
            group1_data=request.control_group,
            group2_data=request.treatment_group,
            metric_type=request.metric_type,
            hypothesis_type=request.hypothesis_type,
            alpha=request.alpha,
            equal_variances=request.equal_variances
        )
        
        # 计算各组统计信息
        if request.metric_type == MetricType.CONVERSION:
            control_stats = stats_service.calculate_conversion_group_stats(
                "control", "Control Group",
                request.control_group["conversions"],
                request.control_group["total_users"]
            )
            treatment_stats = stats_service.calculate_conversion_group_stats(
                "treatment", "Treatment Group", 
                request.treatment_group["conversions"],
                request.treatment_group["total_users"]
            )
        else:
            control_stats = stats_service.calculate_group_stats(
                "control", "Control Group",
                request.control_group["values"],
                request.metric_type
            )
            treatment_stats = stats_service.calculate_group_stats(
                "treatment", "Treatment Group",
                request.treatment_group["values"],
                request.metric_type
            )
        
        # 实际显著性分析
        practical_significance = {}
        if request.metric_type == MetricType.CONVERSION:
            control_rate = control_stats.stats.mean
            treatment_rate = treatment_stats.stats.mean
            relative_change = ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0
            
            practical_significance = {
                "control_conversion_rate": control_rate,
                "treatment_conversion_rate": treatment_rate,
                "absolute_difference": treatment_rate - control_rate,
                "relative_change_percent": relative_change,
                "minimum_detectable_effect": 0.01  # 1%的最小检测效应
            }
        else:
            control_mean = control_stats.stats.mean
            treatment_mean = treatment_stats.stats.mean
            relative_change = ((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0
            
            practical_significance = {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "absolute_difference": treatment_mean - control_mean,
                "relative_change_percent": relative_change
            }
        
        # 解释和建议
        interpretation = _interpret_test_result(test_result, f"{request.metric_type.value} A/B test")
        recommendations = _generate_recommendations(test_result, request.metric_type)
        
        # 添加A/B测试特定的建议
        if test_result.is_significant:
            if abs(practical_significance.get("relative_change_percent", 0)) < 5:
                recommendations.append("虽然统计显著，但相对变化小于5%，需要评估实施成本")
            recommendations.append("建议在更大样本上验证结果")
        else:
            recommendations.append("考虑延长实验时间或优化实验设计")
        
        return ABTestComparisonResponse(
            test_result=test_result.to_dict(),
            control_stats=control_stats.to_dict(),
            treatment_stats=treatment_stats.to_dict(),
            practical_significance=practical_significance,
            interpretation=interpretation,
            recommendations=recommendations,
            message=f"A/B测试比较完成，指标类型：{request.metric_type.value}"
        )
        
    except Exception as e:
        logger.error(f"A/B test comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A/B test comparison failed: {str(e)}")


@router.get("/health")
async def health_check():
    """假设检验服务健康检查"""
    try:
        service = get_hypothesis_testing_service()
        
        # 简单的功能测试
        test_data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        test_data2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        test_result = service.run_t_test(
            test_type="independent_two_sample",
            sample1=test_data1,
            sample2=test_data2,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        return {
            "status": "healthy",
            "service": "hypothesis-testing",
            "test_calculation": {
                "test_type": test_result.test_type,
                "p_value": test_result.p_value,
                "is_significant": test_result.is_significant,
                "passed": test_result.p_value is not None
            },
            "message": "Hypothesis testing service is running properly"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "hypothesis-testing",
            "error": str(e),
            "message": "Hypothesis testing service has issues"
        }


# 导入统计分析服务
from services.statistical_analysis_service import get_stats_calculator