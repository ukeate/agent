"""
实验平台 API
去除静态返回与未实现占位，全部落库并通过事件流计算指标。
"""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import Field
from sqlalchemy import and_, asc, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.core.utils.timezone_utils import utc_now
from src.models.database.event_tracking import EventStream
from src.models.database.experiment import Experiment, ExperimentAssignment, ExperimentVariant
from src.services.report_generation_service import ExperimentNotFoundError, ReportFormat, ReportGenerationService
from src.services.realtime_metrics_service import TimeWindow, get_realtime_metrics_service
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])

class VariantConfig(ApiBaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    traffic: float = Field(..., ge=0, le=100)
    isControl: Optional[bool] = None
    config: Dict[str, Any] = Field(default_factory=dict)

class CreateExperimentBody(ApiBaseModel):
    name: str
    description: str
    type: str = "A/B Testing"
    status: str = "draft"
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    variants: List[VariantConfig]
    metrics: List[Any] = Field(default_factory=list)
    targetingRules: List[Any] = Field(default_factory=list)
    sampleSize: Optional[int] = None
    confidenceLevel: float = 0.95
    power: float = 0.8
    tags: List[str] = Field(default_factory=list)
    enableDataQualityChecks: bool = False
    enableAutoStop: bool = False
    autoStopThreshold: Optional[float] = None
    layer: Optional[str] = None
    hypothesis: Optional[str] = None

class UpdateExperimentBody(ApiBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    endDate: Optional[datetime] = None
    metrics: Optional[List[Any]] = None
    targetingRules: Optional[List[Any]] = None
    tags: Optional[List[str]] = None
    enableDataQualityChecks: Optional[bool] = None
    enableAutoStop: Optional[bool] = None
    autoStopThreshold: Optional[float] = None
    layer: Optional[str] = None
    hypothesis: Optional[str] = None

class ListExperimentsResponse(ApiBaseModel):
    experiments: List[Dict[str, Any]]
    total: int
    page: int
    pageSize: int

class CalculateSampleSizeBody(ApiBaseModel):
    baselineRate: float = Field(..., ge=0, le=1)
    minimumDetectableEffect: float = Field(..., gt=0, le=1)
    confidenceLevel: float = Field(0.95, ge=0.5, le=0.999)
    power: float = Field(0.8, ge=0.5, le=0.99)

class SearchBody(ApiBaseModel):
    filters: Optional[Dict[str, Any]] = None
    sort: Optional[Dict[str, Any]] = None
    pagination: Optional[Dict[str, Any]] = None

class CloneBody(ApiBaseModel):
    name: Optional[str] = None

class ShareBody(ApiBaseModel):
    users: List[str]

class CommentBody(ApiBaseModel):
    text: str
    type: str = "comment"

def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "variant"

def _normalize_metrics(metrics: List[Any]) -> List[str]:
    names: List[str] = []
    for m in metrics:
        if isinstance(m, str):
            n = m.strip()
        elif isinstance(m, dict):
            n = str(m.get("name", "")).strip()
        else:
            n = str(m).strip()
        if n:
            names.append(n)
    return names or ["conversion_rate"]

async def _create_audit_event(
    db: AsyncSession,
    *,
    experiment_id: str,
    user_id: str,
    event_name: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    now = utc_now()
    db.add(
        EventStream(
            event_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            variant_id=None,
            user_id=user_id,
            session_id=None,
            event_type="system",
            event_name=event_name,
            event_category="experiment_audit",
            event_timestamp=now,
            properties=properties or {},
            partition_key=now.strftime("%Y-%m"),
            status="processed",
            data_quality="high",
        )
    )

async def _load_experiment(db: AsyncSession, experiment_id: str) -> Experiment:
    exp = (await db.execute(select(Experiment).where(Experiment.id == experiment_id))).scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="实验不存在")
    return exp

async def _load_variants(db: AsyncSession, experiment_id: str) -> List[ExperimentVariant]:
    rows = (
        await db.execute(
            select(ExperimentVariant)
            .where(ExperimentVariant.experiment_id == experiment_id)
            .order_by(asc(ExperimentVariant.created_at))
        )
    ).scalars().all()
    return list(rows)

async def _assignment_counts(db: AsyncSession, experiment_id: str) -> Dict[str, int]:
    rows = (
        await db.execute(
            select(ExperimentAssignment.variant_id, func.count(ExperimentAssignment.id))
            .where(ExperimentAssignment.experiment_id == experiment_id)
            .group_by(ExperimentAssignment.variant_id)
        )
    ).all()
    return {variant_id: int(count) for variant_id, count in rows}

async def _primary_metric_counts(db: AsyncSession, experiment_id: str, metric_name: str) -> Dict[str, int]:
    rows = (
        await db.execute(
            select(EventStream.variant_id, func.count(EventStream.id))
            .where(
                and_(
                    EventStream.experiment_id == experiment_id,
                    EventStream.event_name == metric_name,
                    EventStream.variant_id.is_not(None),
                )
            )
            .group_by(EventStream.variant_id)
        )
    ).all()
    return {variant_id: int(count) for variant_id, count in rows}

def _status_to_db(status: str) -> str:
    s = (status or "").strip().lower()
    if s in {"draft", "running", "paused", "completed", "terminated"}:
        return s
    raise HTTPException(status_code=400, detail="不支持的实验状态")

async def _to_experiment_dict(db: AsyncSession, exp: Experiment) -> Dict[str, Any]:
    variants_db = await _load_variants(db, exp.id)
    assignments = await _assignment_counts(db, exp.id)
    metrics = exp.success_metrics or []
    primary_metric = metrics[0] if metrics else "conversion_rate"
    conversions = await _primary_metric_counts(db, exp.id, primary_metric)

    variants: List[Dict[str, Any]] = []
    control_rate = 0.0
    best_rate = 0.0
    control_variant_id = None

    for v in variants_db:
        users = float(assignments.get(v.variant_id, 0))
        conv = float(conversions.get(v.variant_id, 0))
        rate = conv / users if users > 0 else 0.0
        if v.is_control and control_variant_id is None:
            control_variant_id = v.variant_id
            control_rate = rate
        best_rate = max(best_rate, rate)

        variants.append(
            {
                "id": v.variant_id,
                "name": v.name,
                "traffic": v.traffic_percentage,
                "isControl": bool(v.is_control),
                "description": v.description,
                "config": v.config or {},
                "sampleSize": int(users),
                "conversions": int(conv),
                "conversionRate": rate,
            }
        )

    participants = int(sum(assignments.values()))
    total_conversions = int(sum(conversions.values()))
    overall_rate = total_conversions / participants if participants > 0 else 0.0
    lift = 0.0
    if control_variant_id and control_rate > 0:
        lift = ((best_rate - control_rate) / control_rate) * 100

    meta = exp.metadata_ or {}
    confidence = 1 - float(exp.significance_level or 0.05)
    if meta.get("confidenceLevel") is not None:
        try:
            confidence = float(meta["confidenceLevel"])
        except (TypeError, ValueError):
            logger.warning("confidenceLevel无效，使用默认值", exc_info=True)

    layer = None
    if exp.layers:
        layer = exp.layers[0]

    return {
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "type": meta.get("type", "A/B Testing"),
        "owner": exp.owner,
        "owners": [exp.owner],
        "status": exp.status,
        "startDate": exp.start_date,
        "endDate": exp.end_date,
        "variants": variants,
        "metrics": metrics,
        "targetingRules": exp.targeting_rules or [],
        "sampleSize": {
            "current": participants,
            "required": int(exp.minimum_sample_size or 0),
        },
        "confidenceLevel": confidence,
        "tags": meta.get("tags", []),
        "enableDataQualityChecks": bool(meta.get("enableDataQualityChecks", False)),
        "enableAutoStop": bool(meta.get("enableAutoStop", False)),
        "autoStopThreshold": meta.get("autoStopThreshold"),
        "layer": layer,
        "hypothesis": exp.hypothesis,
        "participants": participants,
        "total_conversions": total_conversions,
        "conversion_rate": overall_rate,
        "lift": lift,
        "created_at": exp.created_at,
        "updated_at": exp.updated_at,
    }

@router.get("", response_model=ListExperimentsResponse)
async def list_experiments(
    search: Optional[str] = None,
    status: Optional[str] = None,
    owner: Optional[str] = None,
    startDateFrom: Optional[datetime] = None,
    startDateTo: Optional[datetime] = None,
    page: int = Query(1, ge=1),
    pageSize: int = Query(10, ge=1, le=200),
    sortBy: str = Query("created_at"),
    sortOrder: str = Query("desc"),
    db: AsyncSession = Depends(get_db),
) -> ListExperimentsResponse:
    if startDateFrom and startDateTo and startDateFrom > startDateTo:
        raise HTTPException(status_code=400, detail="开始日期不能晚于结束日期")
    stmt = select(Experiment).where(True)
    if search:
        like = f"%{search.strip()}%"
        stmt = stmt.where(or_(Experiment.name.ilike(like), Experiment.description.ilike(like)))
    if status:
        stmt = stmt.where(Experiment.status == _status_to_db(status))
    if owner:
        stmt = stmt.where(Experiment.owner == owner)
    if startDateFrom:
        stmt = stmt.where(Experiment.start_date >= startDateFrom)
    if startDateTo:
        stmt = stmt.where(Experiment.start_date <= startDateTo)

    sort_col = getattr(Experiment, sortBy, None) or Experiment.created_at
    stmt = stmt.order_by(desc(sort_col) if sortOrder.lower() == "desc" else asc(sort_col))

    total = int((await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one())
    rows = (
        await db.execute(stmt.offset((page - 1) * pageSize).limit(pageSize))
    ).scalars().all()

    experiments = [await _to_experiment_dict(db, exp) for exp in rows]
    return ListExperimentsResponse(experiments=experiments, total=total, page=page, pageSize=pageSize)

@router.post("")
async def create_experiment(
    body: CreateExperimentBody, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    if len(body.variants) < 2:
        raise HTTPException(status_code=400, detail="至少需要2个变体")
    traffic_sum = sum(float(v.traffic) for v in body.variants)
    if abs(traffic_sum - 100.0) > 0.01:
        raise HTTPException(status_code=400, detail="变体流量之和必须为100")

    owner = getattr(request.state, "client_id", None) or "unknown"
    start_date = body.startDate or utc_now()
    minimum_sample_size = max(int(body.sampleSize or 100), 100)
    significance_level = max(min(1 - float(body.confidenceLevel), 0.1), 0.01)
    metrics = _normalize_metrics(body.metrics)

    exp = Experiment(
        name=body.name,
        description=body.description,
        hypothesis=body.hypothesis or body.description,
        owner=owner,
        status=_status_to_db(body.status),
        start_date=start_date,
        end_date=body.endDate,
        success_metrics=metrics,
        guardrail_metrics=[],
        minimum_sample_size=minimum_sample_size,
        significance_level=significance_level,
        power=float(body.power),
        layers=[body.layer] if body.layer else [],
        targeting_rules=body.targetingRules or [],
        metadata_={
            "type": body.type,
            "tags": body.tags,
            "enableDataQualityChecks": body.enableDataQualityChecks,
            "enableAutoStop": body.enableAutoStop,
            "autoStopThreshold": body.autoStopThreshold,
            "confidenceLevel": body.confidenceLevel,
        },
    )
    db.add(exp)
    await db.flush()

    used: set[str] = set()
    control_set = False
    for idx, v in enumerate(body.variants):
        variant_id = v.id.strip() if v.id else _slugify(v.name)
        while variant_id in used:
            variant_id = f"{variant_id}_{idx}"
        used.add(variant_id)

        is_control = bool(v.isControl) if v.isControl is not None else (not control_set)
        if is_control and control_set:
            raise HTTPException(status_code=400, detail="只能有一个对照组变体")
        control_set = control_set or is_control

        db.add(
            ExperimentVariant(
                experiment_id=exp.id,
                variant_id=variant_id,
                name=v.name,
                description=v.description,
                config=v.config or {},
                is_control=is_control,
                traffic_percentage=float(v.traffic),
            )
        )

    await _create_audit_event(
        db,
        experiment_id=exp.id,
        user_id=owner,
        event_name="experiment_created",
        properties={"name": exp.name, "status": exp.status},
    )
    await db.commit()
    await db.refresh(exp)
    logger.info("创建实验成功", experiment_id=exp.id)
    return await _to_experiment_dict(db, exp)

@router.put("/{experiment_id}")
async def update_experiment(
    experiment_id: str, body: UpdateExperimentBody, request: Request, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    if body.name is not None:
        exp.name = body.name
    if body.description is not None:
        exp.description = body.description
    if body.hypothesis is not None:
        exp.hypothesis = body.hypothesis
    if body.endDate is not None:
        exp.end_date = body.endDate
    if body.metrics is not None:
        exp.success_metrics = _normalize_metrics(body.metrics)
    if body.targetingRules is not None:
        exp.targeting_rules = body.targetingRules
    if body.layer is not None:
        exp.layers = [body.layer] if body.layer else []

    meta = exp.metadata_ or {}
    if body.tags is not None:
        meta["tags"] = body.tags
    if body.enableDataQualityChecks is not None:
        meta["enableDataQualityChecks"] = body.enableDataQualityChecks
    if body.enableAutoStop is not None:
        meta["enableAutoStop"] = body.enableAutoStop
    if body.autoStopThreshold is not None:
        meta["autoStopThreshold"] = body.autoStopThreshold
    exp.metadata_ = meta

    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(
        db,
        experiment_id=exp.id,
        user_id=owner,
        event_name="experiment_updated",
        properties={"status": exp.status},
    )
    await db.commit()
    await db.refresh(exp)
    return await _to_experiment_dict(db, exp)

@router.delete("/{experiment_id}", status_code=204, response_class=Response)
async def delete_experiment(experiment_id: str, request: Request, db: AsyncSession = Depends(get_db)) -> Response:
    exp = await _load_experiment(db, experiment_id)
    if exp.status != "draft":
        raise HTTPException(status_code=400, detail="仅允许删除草稿实验")
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(
        db, experiment_id=exp.id, user_id=owner, event_name="experiment_deleted", properties={}
    )
    await db.delete(exp)
    await db.commit()
    return Response(status_code=204)

@router.post("/{experiment_id}/start")
async def start_experiment(experiment_id: str, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    exp.status = "running"
    if not exp.start_date:
        exp.start_date = utc_now()
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(db, experiment_id=exp.id, user_id=owner, event_name="experiment_started", properties={})
    await db.commit()
    await db.refresh(exp)
    return await _to_experiment_dict(db, exp)

@router.post("/{experiment_id}/pause")
async def pause_experiment(experiment_id: str, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    exp.status = "paused"
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(db, experiment_id=exp.id, user_id=owner, event_name="experiment_paused", properties={})
    await db.commit()
    await db.refresh(exp)
    return await _to_experiment_dict(db, exp)

@router.post("/{experiment_id}/resume")
async def resume_experiment(experiment_id: str, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    exp.status = "running"
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(db, experiment_id=exp.id, user_id=owner, event_name="experiment_resumed", properties={})
    await db.commit()
    await db.refresh(exp)
    return await _to_experiment_dict(db, exp)

@router.post("/{experiment_id}/stop")
async def stop_experiment(experiment_id: str, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    exp.status = "completed"
    exp.end_date = exp.end_date or utc_now()
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(db, experiment_id=exp.id, user_id=owner, event_name="experiment_completed", properties={})
    await db.commit()
    await db.refresh(exp)
    return await _to_experiment_dict(db, exp)

@router.post("/{experiment_id}/archive")
async def archive_experiment(experiment_id: str, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    exp.status = "terminated"
    meta = exp.metadata_ or {}
    meta["archived"] = True
    exp.metadata_ = meta
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(db, experiment_id=exp.id, user_id=owner, event_name="experiment_archived", properties={})
    await db.commit()
    await db.refresh(exp)
    return await _to_experiment_dict(db, exp)

@router.post("/{experiment_id}/clone")
async def clone_experiment(
    experiment_id: str, body: Optional[CloneBody] = None, request: Request = None, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    variants_db = await _load_variants(db, experiment_id)
    meta = exp.metadata_ or {}

    new_exp = Experiment(
        name=body.name if body and body.name else f"{exp.name} (copy)",
        description=exp.description,
        hypothesis=exp.hypothesis,
        owner=getattr(request.state, "client_id", None) if request else exp.owner,
        status="draft",
        start_date=utc_now(),
        end_date=exp.end_date,
        success_metrics=exp.success_metrics or [],
        guardrail_metrics=exp.guardrail_metrics or [],
        minimum_sample_size=exp.minimum_sample_size,
        significance_level=exp.significance_level,
        power=exp.power,
        layers=exp.layers or [],
        targeting_rules=exp.targeting_rules or [],
        metadata_=meta,
    )
    db.add(new_exp)
    await db.flush()

    for v in variants_db:
        db.add(
            ExperimentVariant(
                experiment_id=new_exp.id,
                variant_id=v.variant_id,
                name=v.name,
                description=v.description,
                config=v.config or {},
                is_control=v.is_control,
                traffic_percentage=v.traffic_percentage,
            )
        )

    owner = getattr(request.state, "client_id", None) if request else new_exp.owner
    await _create_audit_event(
        db, experiment_id=new_exp.id, user_id=owner or new_exp.owner, event_name="experiment_cloned", properties={}
    )
    await db.commit()
    await db.refresh(new_exp)
    return await _to_experiment_dict(db, new_exp)

@router.post("/{experiment_id}/share")
async def share_experiment(experiment_id: str, body: ShareBody, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    meta = exp.metadata_ or {}
    shared = set(meta.get("shared_with", []))
    shared.update(body.users)
    meta["shared_with"] = sorted(shared)
    exp.metadata_ = meta
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(
        db, experiment_id=exp.id, user_id=owner, event_name="experiment_shared", properties={"users": body.users}
    )
    await db.commit()
    await db.refresh(exp)
    return {"success": True}

@router.post("/{experiment_id}/comments")
async def add_comment(experiment_id: str, body: CommentBody, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(
        db,
        experiment_id=exp.id,
        user_id=owner,
        event_name="experiment_comment",
        properties={"text": body.text, "type": body.type},
    )
    await db.commit()
    return {"success": True}

@router.put("/{experiment_id}/settings")
async def update_settings(experiment_id: str, settings: Dict[str, Any], request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    meta = exp.metadata_ or {}
    meta.update(settings)
    exp.metadata_ = meta
    owner = getattr(request.state, "client_id", None) or exp.owner
    await _create_audit_event(
        db,
        experiment_id=exp.id,
        user_id=owner,
        event_name="experiment_settings_updated",
        properties={"keys": list(settings.keys())},
    )
    await db.commit()
    await db.refresh(exp)
    return {"success": True}

@router.get("/{experiment_id}/audit")
async def get_audit_log(
    experiment_id: str, limit: int = Query(200, ge=1, le=1000), db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    await _load_experiment(db, experiment_id)
    rows = (
        await db.execute(
            select(EventStream)
            .where(and_(EventStream.experiment_id == experiment_id, EventStream.event_category == "experiment_audit"))
            .order_by(desc(EventStream.event_timestamp))
            .limit(limit)
        )
    ).scalars().all()
    return [
        {
            "id": e.event_id,
            "event_name": e.event_name,
            "timestamp": e.event_timestamp,
            "user_id": e.user_id,
            "properties": e.properties,
        }
        for e in rows
    ]

@router.get("/{experiment_id}/events")
async def get_events(
    experiment_id: str, limit: int = Query(200, ge=1, le=5000), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    await _load_experiment(db, experiment_id)
    rows = (
        await db.execute(
            select(EventStream)
            .where(EventStream.experiment_id == experiment_id)
            .order_by(desc(EventStream.event_timestamp))
            .limit(limit)
        )
    ).scalars().all()
    return {
        "events": [
            {
                "event_id": e.event_id,
                "variant_id": e.variant_id,
                "user_id": e.user_id,
                "event_type": e.event_type,
                "event_name": e.event_name,
                "event_category": e.event_category,
                "event_timestamp": e.event_timestamp,
                "properties": e.properties,
            }
            for e in rows
        ],
        "total": len(rows),
    }

@router.get("/{experiment_id}/metrics")
async def get_metrics(experiment_id: str) -> Dict[str, Any]:
    service = await get_realtime_metrics_service()
    metrics = await service.calculate_metrics(experiment_id, TimeWindow.CUMULATIVE)
    return {k: v.to_dict() for k, v in metrics.items()}

@router.get("/{experiment_id}/monitoring")
async def get_monitoring(
    experiment_id: str, metric: str = Query("conversion_rate"), granularity: str = Query("hourly")
) -> Dict[str, Any]:
    service = await get_realtime_metrics_service()
    window = TimeWindow.HOURLY if granularity == "hourly" else TimeWindow.DAILY
    trends = await service.get_metric_trends(experiment_id, metric, window)
    latest = await service.calculate_metrics(experiment_id, TimeWindow.REALTIME)
    return {"latest": {k: v.to_dict() for k, v in latest.items()}, "trends": [t.to_dict() for t in trends]}

@router.get("/{experiment_id}/analysis")
async def get_analysis(experiment_id: str) -> Dict[str, Any]:
    service = await get_realtime_metrics_service()
    metrics = await service.calculate_metrics(experiment_id, TimeWindow.CUMULATIVE)
    groups = list(metrics.keys())
    comparison = None
    if len(groups) >= 2:
        comparison = await service.compare_groups(experiment_id, groups[0], groups[1])
        comparison = {k: v.to_dict() for k, v in comparison.items()}
    return {"metrics": {k: v.to_dict() for k, v in metrics.items()}, "comparison": comparison}

@router.get("/{experiment_id}/cost-analysis")
async def get_cost_analysis(experiment_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    await _load_experiment(db, experiment_id)
    total_events = int(
        (
            await db.execute(
                select(func.count()).select_from(EventStream).where(EventStream.experiment_id == experiment_id)
            )
        ).scalar_one()
    )
    return {
        "experiment_id": experiment_id,
        "event_count": total_events,
        "estimated_cost": {
            "currency": "USD",
            "amount": round(total_events * 0.00001, 6),
            "unit_cost_per_event": 0.00001,
        },
    }

@router.get("/{experiment_id}/report")
async def get_report(experiment_id: str, format: str = Query("json")) -> Dict[str, Any]:
    fmt = format.lower()
    report_format = ReportFormat.JSON if fmt == "json" else ReportFormat.HTML if fmt == "html" else ReportFormat.PDF
    try:
        report = await ReportGenerationService().generate_report(experiment_id=experiment_id, format=report_format)
        return report
    except ExperimentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/templates")
async def get_templates() -> List[Dict[str, Any]]:
    return [
        {
            "id": "ab_basic",
            "name": "基础A/B测试",
            "description": "标准对照组/实验组两变体实验",
            "default": {
                "type": "A/B Testing",
                "variants": [
                    {"name": "Control", "traffic": 50, "isControl": True},
                    {"name": "Treatment", "traffic": 50},
                ],
                "metrics": ["conversion_rate"],
                "confidenceLevel": 0.95,
                "power": 0.8,
            },
        }
    ]

@router.post("/from-template")
async def create_from_template(body: Dict[str, Any], request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    template_id = str(body.get("templateId", "")).strip()
    overrides = body.get("overrides") or {}
    templates = {t["id"]: t for t in await get_templates()}
    if template_id not in templates:
        raise HTTPException(status_code=404, detail="模板不存在")
    default = templates[template_id]["default"]
    payload = {**default, **overrides}
    return await create_experiment(CreateExperimentBody(**payload), request, db)

@router.post("/validate")
async def validate_config(body: CreateExperimentBody) -> Dict[str, Any]:
    errors: List[str] = []
    if len(body.variants) < 2:
        errors.append("至少需要2个变体")
    if abs(sum(float(v.traffic) for v in body.variants) - 100.0) > 0.01:
        errors.append("变体流量之和必须为100")
    if not body.name.strip():
        errors.append("实验名称不能为空")
    return {"valid": not errors, "errors": errors or None}

@router.post("/calculate-sample-size")
async def calculate_sample_size(body: CalculateSampleSizeBody) -> Dict[str, Any]:
    from src.services.power_analysis_service import AlternativeHypothesis, get_power_analysis_service

    try:
        alpha = 1 - body.confidenceLevel
        service = get_power_analysis_service()
        return service.calculate_ab_test_sample_size(
            baseline_conversion_rate=body.baselineRate,
            minimum_detectable_effect=body.minimumDetectableEffect,
            power=body.power,
            alpha=alpha,
            alternative=AlternativeHypothesis.TWO_SIDED,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/search")
async def search_experiments(body: SearchBody, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    filters = body.filters or {}
    pagination = body.pagination or {}
    page = int(pagination.get("page", 1))
    limit = int(pagination.get("limit", 20))

    stmt = select(Experiment).where(True)
    if statuses := filters.get("status"):
        stmt = stmt.where(Experiment.status.in_([_status_to_db(s) for s in statuses]))
    if owners := filters.get("owner"):
        stmt = stmt.where(Experiment.owner.in_(owners))
    if types := filters.get("type"):
        stmt = stmt.where(Experiment.metadata_.op("->>")("type").in_(types))

    total = int((await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one())
    rows = (
        await db.execute(
            stmt.order_by(desc(Experiment.created_at)).offset((page - 1) * limit).limit(limit)
        )
    ).scalars().all()
    return {"experiments": [await _to_experiment_dict(db, e) for e in rows], "total": total, "page": page}

def _export_bytes(data: bytes, filename: str, content_type: str) -> Response:
    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@router.get("/{experiment_id}/export")
async def export_experiment(experiment_id: str, format: str = Query("json"), db: AsyncSession = Depends(get_db)) -> Response:
    exp = await _load_experiment(db, experiment_id)
    exp_dict = await _to_experiment_dict(db, exp)
    fmt = format.lower()

    if fmt == "json":
        payload = json.dumps(exp_dict, default=str, ensure_ascii=False).encode("utf-8")
        return _export_bytes(payload, f"experiment_{experiment_id}.json", "application/json")

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "name", "status", "startDate", "endDate", "type"])
        w.writerow(
            [
                exp_dict["id"],
                exp_dict["name"],
                exp_dict["status"],
                exp_dict["startDate"],
                exp_dict["endDate"],
                exp_dict["type"],
            ]
        )
        payload = buf.getvalue().encode("utf-8")
        return _export_bytes(payload, f"experiment_{experiment_id}.csv", "text/csv")

    if fmt == "xlsx":
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "experiment"
        ws.append(["id", "name", "status", "startDate", "endDate", "type"])
        ws.append(
            [
                exp_dict["id"],
                exp_dict["name"],
                exp_dict["status"],
                str(exp_dict["startDate"]),
                str(exp_dict["endDate"]),
                exp_dict["type"],
            ]
        )
        out = io.BytesIO()
        wb.save(out)
        return _export_bytes(
            out.getvalue(),
            f"experiment_{experiment_id}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    raise HTTPException(status_code=400, detail="不支持的导出格式")

@router.post("/export")
async def export_experiments(body: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Response:
    ids = body.get("ids") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="ids不能为空")
    experiments: List[Dict[str, Any]] = []
    for exp_id in ids:
        exp = await _load_experiment(db, str(exp_id))
        experiments.append(await _to_experiment_dict(db, exp))
    payload = json.dumps(experiments, default=str, ensure_ascii=False).encode("utf-8")
    return _export_bytes(payload, "experiments.json", "application/json")

@router.post("/import")
async def import_experiments(file: UploadFile = File(...), request: Request = None, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    raw = await file.read()
    imported = 0
    failed = 0
    try:
        items = json.loads(raw.decode("utf-8"))
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise ValueError("无效导入格式")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析失败: {e}")

    for item in items:
        try:
            await create_experiment(CreateExperimentBody(**item), request, db)
            imported += 1
        except Exception:
            await db.rollback()
            failed += 1

    return {"imported": imported, "failed": failed}

@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    exp = await _load_experiment(db, experiment_id)
    return await _to_experiment_dict(db, exp)
