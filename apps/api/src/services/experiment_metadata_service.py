"""
A/B测试实验元数据管理服务 - 管理实验的元数据、标签、分类等信息
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass
import json
import re
from src.models.schemas.experiment import ExperimentConfig, ExperimentStatus

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class ExperimentMetadata:
    """实验元数据"""
    experiment_id: str
    tags: Set[str]
    category: str
    business_unit: str
    product: str
    feature_flag: Optional[str]
    jira_ticket: Optional[str]
    confluence_page: Optional[str]
    stakeholders: List[str]
    custom_fields: Dict[str, Any]
    created_by: str
    created_at: datetime
    updated_by: str
    updated_at: datetime

@dataclass
class ExperimentTemplate:
    """实验模板"""
    template_id: str
    name: str
    description: str
    category: str
    variants_template: List[Dict[str, Any]]
    metrics_template: List[str]
    default_config: Dict[str, Any]
    tags: Set[str]
    created_by: str
    created_at: datetime

@dataclass
class ExperimentArchive:
    """实验归档信息"""
    experiment_id: str
    archived_at: datetime
    archived_by: str
    archive_reason: str
    final_results: Dict[str, Any]
    lessons_learned: str
    retention_policy: str

class ExperimentMetadataService:
    """实验元数据管理服务"""
    
    def __init__(self):
        # 在实际实现中，这些应该连接到数据库
        self.metadata_store: Dict[str, ExperimentMetadata] = {}
        self.templates_store: Dict[str, ExperimentTemplate] = {}
        self.archives_store: Dict[str, ExperimentArchive] = {}
        
        # 预定义的分类和标签
        self.predefined_categories = {
            "ui_ux": "用户界面/体验",
            "algorithm": "算法优化", 
            "pricing": "定价策略",
            "marketing": "营销活动",
            "onboarding": "用户引导",
            "retention": "用户留存",
            "conversion": "转化优化",
            "performance": "性能优化",
            "feature": "功能测试",
            "content": "内容测试"
        }
        
        self.predefined_tags = {
            "mobile", "web", "ios", "android", "desktop",
            "homepage", "checkout", "signup", "login",
            "recommendation", "search", "notification",
            "premium", "freemium", "enterprise",
            "high_priority", "low_risk", "quick_win"
        }
    
    async def create_metadata(self, experiment_id: str, metadata_request: Dict[str, Any], 
                            created_by: str) -> ExperimentMetadata:
        """创建实验元数据"""
        try:
            # 验证和清理标签
            tags = set(metadata_request.get("tags", []))
            tags = self._validate_tags(tags)
            
            # 验证分类
            category = metadata_request.get("category", "feature")
            if category not in self.predefined_categories:
                logger.warning(f"Unknown category '{category}' for experiment {experiment_id}")
            
            # 创建元数据对象
            metadata = ExperimentMetadata(
                experiment_id=experiment_id,
                tags=tags,
                category=category,
                business_unit=metadata_request.get("business_unit", ""),
                product=metadata_request.get("product", ""),
                feature_flag=metadata_request.get("feature_flag"),
                jira_ticket=metadata_request.get("jira_ticket"),
                confluence_page=metadata_request.get("confluence_page"),
                stakeholders=metadata_request.get("stakeholders", []),
                custom_fields=metadata_request.get("custom_fields", {}),
                created_by=created_by,
                created_at=utc_now(),
                updated_by=created_by,
                updated_at=utc_now()
            )
            
            # 存储元数据
            self.metadata_store[experiment_id] = metadata
            
            logger.info(f"Created metadata for experiment {experiment_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to create metadata for experiment {experiment_id}: {str(e)}")
            raise
    
    async def update_metadata(self, experiment_id: str, metadata_updates: Dict[str, Any], 
                            updated_by: str) -> Optional[ExperimentMetadata]:
        """更新实验元数据"""
        try:
            metadata = self.metadata_store.get(experiment_id)
            if not metadata:
                return None
            
            # 更新字段
            if "tags" in metadata_updates:
                metadata.tags = self._validate_tags(set(metadata_updates["tags"]))
            
            if "category" in metadata_updates:
                metadata.category = metadata_updates["category"]
            
            if "business_unit" in metadata_updates:
                metadata.business_unit = metadata_updates["business_unit"]
            
            if "product" in metadata_updates:
                metadata.product = metadata_updates["product"]
            
            if "feature_flag" in metadata_updates:
                metadata.feature_flag = metadata_updates["feature_flag"]
            
            if "jira_ticket" in metadata_updates:
                metadata.jira_ticket = metadata_updates["jira_ticket"]
            
            if "confluence_page" in metadata_updates:
                metadata.confluence_page = metadata_updates["confluence_page"]
            
            if "stakeholders" in metadata_updates:
                metadata.stakeholders = metadata_updates["stakeholders"]
            
            if "custom_fields" in metadata_updates:
                metadata.custom_fields.update(metadata_updates["custom_fields"])
            
            # 更新时间戳
            metadata.updated_by = updated_by
            metadata.updated_at = utc_now()
            
            logger.info(f"Updated metadata for experiment {experiment_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to update metadata for experiment {experiment_id}: {str(e)}")
            raise
    
    async def get_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """获取实验元数据"""
        return self.metadata_store.get(experiment_id)
    
    async def search_experiments_by_metadata(self, search_criteria: Dict[str, Any]) -> List[str]:
        """根据元数据搜索实验"""
        try:
            matching_experiments = []
            
            for experiment_id, metadata in self.metadata_store.items():
                if self._matches_criteria(metadata, search_criteria):
                    matching_experiments.append(experiment_id)
            
            return matching_experiments
            
        except Exception as e:
            logger.error(f"Failed to search experiments by metadata: {str(e)}")
            return []
    
    async def get_experiments_by_tags(self, tags: List[str]) -> List[str]:
        """根据标签获取实验列表"""
        try:
            tag_set = set(tags)
            matching_experiments = []
            
            for experiment_id, metadata in self.metadata_store.items():
                if tag_set.intersection(metadata.tags):
                    matching_experiments.append(experiment_id)
            
            return matching_experiments
            
        except Exception as e:
            logger.error(f"Failed to get experiments by tags: {str(e)}")
            return []
    
    async def get_experiments_by_category(self, category: str) -> List[str]:
        """根据分类获取实验列表"""
        try:
            matching_experiments = []
            
            for experiment_id, metadata in self.metadata_store.items():
                if metadata.category == category:
                    matching_experiments.append(experiment_id)
            
            return matching_experiments
            
        except Exception as e:
            logger.error(f"Failed to get experiments by category: {str(e)}")
            return []
    
    async def get_experiment_analytics(self) -> Dict[str, Any]:
        """获取实验元数据分析"""
        try:
            analytics = {
                "total_experiments": len(self.metadata_store),
                "categories": {},
                "tags": {},
                "business_units": {},
                "products": {},
                "stakeholders": {},
                "creation_trends": {}
            }
            
            # 统计分类
            for metadata in self.metadata_store.values():
                # 分类统计
                category = metadata.category
                analytics["categories"][category] = analytics["categories"].get(category, 0) + 1
                
                # 标签统计
                for tag in metadata.tags:
                    analytics["tags"][tag] = analytics["tags"].get(tag, 0) + 1
                
                # 业务单元统计
                if metadata.business_unit:
                    bu = metadata.business_unit
                    analytics["business_units"][bu] = analytics["business_units"].get(bu, 0) + 1
                
                # 产品统计
                if metadata.product:
                    product = metadata.product
                    analytics["products"][product] = analytics["products"].get(product, 0) + 1
                
                # 利益相关者统计
                for stakeholder in metadata.stakeholders:
                    analytics["stakeholders"][stakeholder] = analytics["stakeholders"].get(stakeholder, 0) + 1
                
                # 创建趋势（按月统计）
                month_key = metadata.created_at.strftime("%Y-%m")
                analytics["creation_trends"][month_key] = analytics["creation_trends"].get(month_key, 0) + 1
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get experiment analytics: {str(e)}")
            return {}
    
    # 模板管理
    async def create_template(self, template_request: Dict[str, Any], created_by: str) -> ExperimentTemplate:
        """创建实验模板"""
        try:
            template = ExperimentTemplate(
                template_id=f"template_{utc_now().strftime('%Y%m%d_%H%M%S')}",
                name=template_request["name"],
                description=template_request["description"],
                category=template_request.get("category", "feature"),
                variants_template=template_request.get("variants_template", []),
                metrics_template=template_request.get("metrics_template", []),
                default_config=template_request.get("default_config", {}),
                tags=set(template_request.get("tags", [])),
                created_by=created_by,
                created_at=utc_now()
            )
            
            self.templates_store[template.template_id] = template
            
            logger.info(f"Created experiment template {template.template_id}")
            return template
            
        except Exception as e:
            logger.error(f"Failed to create experiment template: {str(e)}")
            raise
    
    async def get_template(self, template_id: str) -> Optional[ExperimentTemplate]:
        """获取实验模板"""
        return self.templates_store.get(template_id)
    
    async def list_templates(self, category: Optional[str] = None) -> List[ExperimentTemplate]:
        """列出实验模板"""
        try:
            templates = list(self.templates_store.values())
            
            if category:
                templates = [t for t in templates if t.category == category]
            
            return sorted(templates, key=lambda t: t.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list templates: {str(e)}")
            return []
    
    async def apply_template(self, template_id: str, experiment_id: str) -> Dict[str, Any]:
        """将模板应用到实验"""
        try:
            template = self.templates_store.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # 构建实验配置建议
            config_suggestion = {
                "variants_template": template.variants_template,
                "success_metrics": template.metrics_template,
                "default_config": template.default_config,
                "suggested_tags": list(template.tags),
                "category": template.category
            }
            
            logger.info(f"Applied template {template_id} to experiment {experiment_id}")
            return config_suggestion
            
        except Exception as e:
            logger.error(f"Failed to apply template {template_id}: {str(e)}")
            raise
    
    # 归档管理
    async def archive_experiment(self, experiment_id: str, archive_request: Dict[str, Any], 
                               archived_by: str) -> ExperimentArchive:
        """归档实验"""
        try:
            archive = ExperimentArchive(
                experiment_id=experiment_id,
                archived_at=utc_now(),
                archived_by=archived_by,
                archive_reason=archive_request.get("reason", "experiment_completed"),
                final_results=archive_request.get("final_results", {}),
                lessons_learned=archive_request.get("lessons_learned", ""),
                retention_policy=archive_request.get("retention_policy", "keep_forever")
            )
            
            self.archives_store[experiment_id] = archive
            
            logger.info(f"Archived experiment {experiment_id}")
            return archive
            
        except Exception as e:
            logger.error(f"Failed to archive experiment {experiment_id}: {str(e)}")
            raise
    
    async def get_archived_experiments(self, days_back: int = 30) -> List[ExperimentArchive]:
        """获取归档的实验"""
        try:
            cutoff_date = utc_now() - timedelta(days=days_back)
            
            archives = [
                archive for archive in self.archives_store.values()
                if archive.archived_at >= cutoff_date
            ]
            
            return sorted(archives, key=lambda a: a.archived_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get archived experiments: {str(e)}")
            return []
    
    # 辅助方法
    def _validate_tags(self, tags: Set[str]) -> Set[str]:
        """验证和清理标签"""
        validated_tags = set()
        
        for tag in tags:
            # 清理标签格式
            clean_tag = tag.lower().strip().replace(" ", "_")
            if len(clean_tag) > 0 and len(clean_tag) <= 50:
                validated_tags.add(clean_tag)
            else:
                logger.warning(f"Invalid tag format: '{tag}'")
        
        return validated_tags
    
    def _matches_criteria(self, metadata: ExperimentMetadata, criteria: Dict[str, Any]) -> bool:
        """检查元数据是否匹配搜索条件"""
        try:
            # 分类匹配
            if "category" in criteria and metadata.category != criteria["category"]:
                return False
            
            # 标签匹配
            if "tags" in criteria:
                required_tags = set(criteria["tags"])
                if not required_tags.intersection(metadata.tags):
                    return False
            
            # 业务单元匹配
            if "business_unit" in criteria and metadata.business_unit != criteria["business_unit"]:
                return False
            
            # 产品匹配
            if "product" in criteria and metadata.product != criteria["product"]:
                return False
            
            # 利益相关者匹配
            if "stakeholder" in criteria and criteria["stakeholder"] not in metadata.stakeholders:
                return False
            
            # 创建时间范围匹配
            if "created_after" in criteria and metadata.created_at < criteria["created_after"]:
                return False
            
            if "created_before" in criteria and metadata.created_at > criteria["created_before"]:
                return False
            
            # 自定义字段匹配
            if "custom_fields" in criteria:
                for key, value in criteria["custom_fields"].items():
                    if key not in metadata.custom_fields or metadata.custom_fields[key] != value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching criteria: {str(e)}")
            return False
    
    async def get_metadata_schema(self) -> Dict[str, Any]:
        """获取元数据字段schema"""
        return {
            "categories": {
                "type": "enum",
                "values": list(self.predefined_categories.keys()),
                "descriptions": self.predefined_categories
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "predefined": list(self.predefined_tags),
                "max_items": 20
            },
            "business_unit": {
                "type": "string",
                "max_length": 100
            },
            "product": {
                "type": "string",
                "max_length": 100
            },
            "feature_flag": {
                "type": "string",
                "max_length": 100,
                "description": "关联的功能开关名称"
            },
            "jira_ticket": {
                "type": "string",
                "pattern": r"^[A-Z]+-\d+$",
                "description": "JIRA ticket格式: PROJECT-123"
            },
            "confluence_page": {
                "type": "string",
                "format": "url",
                "description": "Confluence页面URL"
            },
            "stakeholders": {
                "type": "array",
                "items": {"type": "string"},
                "max_items": 10
            },
            "custom_fields": {
                "type": "object",
                "description": "自定义字段，支持任意键值对"
            }
        }
    
    async def validate_metadata_request(self, metadata_request: Dict[str, Any]) -> List[str]:
        """验证元数据请求"""
        errors = []
        
        try:
            # 验证分类
            if "category" in metadata_request:
                category = metadata_request["category"]
                if category not in self.predefined_categories:
                    errors.append(f"Invalid category: {category}")
            
            # 验证标签
            if "tags" in metadata_request:
                tags = metadata_request["tags"]
                if not isinstance(tags, list) or len(tags) > 20:
                    errors.append("Tags must be a list with at most 20 items")
            
            # 验证JIRA ticket格式
            if "jira_ticket" in metadata_request:
                jira_ticket = metadata_request["jira_ticket"]
                if jira_ticket and not re.match(r"^[A-Z]+-\d+$", jira_ticket):
                    errors.append("JIRA ticket must match format: PROJECT-123")
            
            # 验证利益相关者数量
            if "stakeholders" in metadata_request:
                stakeholders = metadata_request["stakeholders"]
                if not isinstance(stakeholders, list) or len(stakeholders) > 10:
                    errors.append("Stakeholders must be a list with at most 10 items")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
