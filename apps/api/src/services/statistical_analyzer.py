"""
A/B测试统计分析引擎 - 基于SciPy和statsmodels实现统计检验和功效分析
"""

from typing import Dict, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

from src.core.logging import get_logger
logger = get_logger(__name__)

class StatisticalAnalyzer:
    """统计分析引擎"""
    
    def __init__(self):
        """初始化统计分析器"""
        from statsmodels.stats.power import TTestIndPower

        self._power = TTestIndPower()
    
    async def analyze_metric(self, experiment_id: str, metric_name: str, 
                           metric_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验指标"""
        import numpy as np
        from scipy import stats

        logger.info(f"Analyzing metric {metric_name} for experiment {experiment_id}")

        variant_data = (metric_data or {}).get("variant_data") or {}
        values_by_variant: Dict[str, np.ndarray] = {}
        for variant_id, rows in variant_data.items():
            values = []
            for row in rows or []:
                try:
                    values.append(float(row.get("value", 0.0)))
                except Exception:
                    continue
            values_by_variant[str(variant_id)] = np.asarray(values, dtype=float)

        if len(values_by_variant) < 2:
            now = utc_now()
            return {
                "variant_results": {
                    vid: {"mean": float(v.mean()) if v.size else 0.0, "count": float(v.size)}
                    for vid, v in values_by_variant.items()
                },
                "statistical_test": "ttest_ind_welch",
                "p_value": 1.0,
                "is_significant": False,
                "effect_size": 0.0,
                "confidence_interval": (0.0, 0.0),
                "sample_size": int(sum(v.size for v in values_by_variant.values())),
                "statistical_power": 0.0,
                "data_window_start": now,
                "data_window_end": now,
            }

        control_id = "control" if "control" in values_by_variant else sorted(values_by_variant.keys())[0]
        control_values = values_by_variant.get(control_id, np.asarray([], dtype=float))

        variant_results: Dict[str, Dict[str, float]] = {}
        for vid, arr in values_by_variant.items():
            mean = float(arr.mean()) if arr.size else 0.0
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            variant_results[vid] = {"mean": mean, "std": std, "count": float(arr.size)}

        best = None
        for vid, arr in values_by_variant.items():
            if vid == control_id:
                continue
            if control_values.size < 2 or arr.size < 2:
                continue

            t_stat, p_value = stats.ttest_ind(arr, control_values, equal_var=False, nan_policy="omit")
            diff = float(arr.mean() - control_values.mean())

            control_var = float(control_values.var(ddof=1))
            treat_var = float(arr.var(ddof=1))
            se = float(np.sqrt(control_var / control_values.size + treat_var / arr.size))
            ci = (diff - 1.96 * se, diff + 1.96 * se) if se > 0 else (0.0, 0.0)

            pooled = (control_var + treat_var) / 2
            effect = diff / float(np.sqrt(pooled)) if pooled > 0 else 0.0

            if best is None or float(p_value) < best["p_value"]:
                best = {
                    "variant_id": vid,
                    "p_value": float(p_value) if p_value == p_value else 1.0,
                    "effect_size": float(effect),
                    "confidence_interval": (float(ci[0]), float(ci[1])),
                    "ratio": float(arr.size / control_values.size),
                    "nobs1": int(control_values.size),
                }

        if best is None:
            p_value = 1.0
            effect_size = 0.0
            confidence_interval = (0.0, 0.0)
            statistical_power = 0.0
        else:
            p_value = best["p_value"]
            effect_size = best["effect_size"]
            confidence_interval = best["confidence_interval"]
            try:
                statistical_power = float(
                    self._power.power(
                        effect_size=abs(effect_size),
                        nobs1=best["nobs1"],
                        ratio=best["ratio"],
                        alpha=0.05,
                    )
                )
            except Exception:
                statistical_power = 0.0

        now = utc_now()
        sample_size = int(sum(v.size for v in values_by_variant.values()))
        return {
            "variant_results": variant_results,
            "statistical_test": "ttest_ind_welch",
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
            "effect_size": float(effect_size),
            "confidence_interval": confidence_interval,
            "sample_size": sample_size,
            "statistical_power": statistical_power,
            "data_window_start": metric_data.get("data_window_start") or now,
            "data_window_end": metric_data.get("data_window_end") or now,
        }
