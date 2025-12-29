from __future__ import annotations

from typing import Dict, List
from sqlalchemy import select
from src.core.database import get_db_session
from src.models.database.experiment import Experiment, ExperimentVariant

async def set_experiment_status(experiment_id: str, status: str) -> None:
    if status not in {"draft", "running", "paused", "completed", "terminated"}:
        raise ValueError("无效实验状态")

    async with get_db_session() as session:
        exp = await session.get(Experiment, experiment_id)
        if not exp:
            raise ValueError("实验不存在")
        exp.status = status
        await session.commit()

def _normalize_variant_traffic(
    variants: List[ExperimentVariant], target_variant_id: str, percentage: float
) -> None:
    if not 0 <= percentage <= 100:
        raise ValueError("percentage必须在0-100之间")

    target = next((v for v in variants if v.variant_id == target_variant_id), None)
    if not target:
        raise ValueError("变体不存在")

    others = [v for v in variants if v.variant_id != target_variant_id]
    if not others:
        raise ValueError("实验至少需要2个变体")

    remaining = 100.0 - float(percentage)
    sum_others = float(sum(v.traffic_percentage for v in others))

    target.traffic_percentage = float(percentage)
    if sum_others > 0:
        for v in others:
            v.traffic_percentage = remaining * float(v.traffic_percentage) / sum_others
    else:
        avg = remaining / len(others)
        for v in others:
            v.traffic_percentage = avg

    total = float(sum(v.traffic_percentage for v in variants))
    diff = 100.0 - total
    if abs(diff) > 1e-6:
        others[-1].traffic_percentage = float(others[-1].traffic_percentage) + diff

async def set_variant_traffic(experiment_id: str, variant_id: str, percentage: float) -> Dict[str, float]:
    async with get_db_session() as session:
        variants = (
            (await session.execute(select(ExperimentVariant).where(ExperimentVariant.experiment_id == experiment_id)))
            .scalars()
            .all()
        )
        if not variants:
            raise ValueError("实验不存在或无变体")

        _normalize_variant_traffic(variants, variant_id, percentage)
        await session.commit()
        return {v.variant_id: float(v.traffic_percentage) for v in variants}

async def route_all_traffic_to_control(experiment_id: str) -> Dict[str, float]:
    async with get_db_session() as session:
        variants = (
            (await session.execute(select(ExperimentVariant).where(ExperimentVariant.experiment_id == experiment_id)))
            .scalars()
            .all()
        )
        if not variants:
            raise ValueError("实验不存在或无变体")

        control = next((v for v in variants if bool(v.is_control)), None)
        if not control:
            raise ValueError("找不到对照组变体")

        for v in variants:
            v.traffic_percentage = 100.0 if v.variant_id == control.variant_id else 0.0

        await session.commit()
        return {v.variant_id: float(v.traffic_percentage) for v in variants}

