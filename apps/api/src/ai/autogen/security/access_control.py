"""
分布式安全框架 - 访问控制引擎
支持RBAC、ABAC模型，动态权限评估和策略管理
"""

import time
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from src.core.logging import get_logger
class AccessDecision(Enum):
    PERMIT = "permit"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"

class ResourceType(Enum):
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    MESSAGE_QUEUE = "message_queue"
    COMPUTE_RESOURCE = "compute_resource"
    AI_MODEL = "ai_model"
    AGENT_SERVICE = "agent_service"

@dataclass
class AccessRequest:
    subject_id: str  # 智能体ID
    resource_id: str  # 资源ID
    action: str  # 操作类型
    resource_type: ResourceType
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class AccessPolicy:
    policy_id: str
    name: str
    description: str
    target: Dict[str, Any]  # 策略适用范围
    rules: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

@dataclass
class Role:
    role_id: str
    name: str
    description: str
    permissions: Set[str]
    parent_roles: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Subject:
    subject_id: str
    roles: Set[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

class PolicyEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, request: AccessRequest, policies: List[AccessPolicy]) -> AccessDecision:
        ...

class RBACEvaluator(PolicyEvaluator):
    """基于角色的访问控制评估器"""
    
    def __init__(self, roles: Dict[str, Role], subjects: Dict[str, Subject]):
        self.roles = roles
        self.subjects = subjects
        self.logger = get_logger(__name__)
    
    async def evaluate(self, request: AccessRequest, policies: List[AccessPolicy]) -> AccessDecision:
        subject = self.subjects.get(request.subject_id)
        if not subject or not subject.active:
            self.logger.warning(f"Subject not found or inactive: {request.subject_id}")
            return AccessDecision.DENY
        
        # 获取主体的所有有效权限
        permissions = await self._get_subject_permissions(subject)
        
        # 构造所需权限
        required_permission = f"{request.resource_type.value}:{request.action}:{request.resource_id}"
        
        # 检查权限匹配
        for permission in permissions:
            if await self._permission_matches(permission, required_permission):
                self.logger.info(f"RBAC access granted for {request.subject_id}")
                return AccessDecision.PERMIT
        
        self.logger.warning(f"RBAC access denied for {request.subject_id}")
        return AccessDecision.DENY
    
    async def _get_subject_permissions(self, subject: Subject) -> Set[str]:
        """获取主体的所有权限"""
        permissions = set()
        visited_roles = set()
        
        async def collect_role_permissions(role_id: str):
            if role_id in visited_roles:
                return
            visited_roles.add(role_id)
            
            role = self.roles.get(role_id)
            if role:
                permissions.update(role.permissions)
                for parent_role_id in role.parent_roles:
                    await collect_role_permissions(parent_role_id)
        
        for role_id in subject.roles:
            await collect_role_permissions(role_id)
        
        return permissions
    
    async def _permission_matches(self, permission: str, required_permission: str) -> bool:
        """检查权限是否匹配"""
        # 支持通配符匹配
        permission_pattern = permission.replace('*', '.*')
        return bool(re.match(f"^{permission_pattern}$", required_permission))

class ABACEvaluator(PolicyEvaluator):
    """基于属性的访问控制评估器"""
    
    def __init__(self, attribute_provider):
        self.attribute_provider = attribute_provider
        self.logger = get_logger(__name__)
    
    async def evaluate(self, request: AccessRequest, policies: List[AccessPolicy]) -> AccessDecision:
        # 收集所有相关属性
        attributes = await self._collect_attributes(request)
        
        # 评估所有适用的策略
        decisions = []
        for policy in policies:
            if await self._is_policy_applicable(policy, request, attributes):
                decision = await self._evaluate_policy_rules(policy, attributes)
                decisions.append((policy.priority, decision))
        
        # 根据优先级和组合算法确定最终决策
        final_decision = await self._combine_decisions(decisions)
        
        if final_decision == AccessDecision.PERMIT:
            self.logger.info(f"ABAC access granted for {request.subject_id}")
        else:
            self.logger.warning(f"ABAC access denied for {request.subject_id}")
        
        return final_decision
    
    async def _collect_attributes(self, request: AccessRequest) -> Dict[str, Any]:
        """收集访问请求相关的所有属性"""
        attributes = {
            'subject': await self.attribute_provider.get_subject_attributes(request.subject_id),
            'resource': await self.attribute_provider.get_resource_attributes(request.resource_id),
            'action': {'name': request.action},
            'environment': await self.attribute_provider.get_environment_attributes(),
            'context': request.context
        }
        return attributes
    
    async def _is_policy_applicable(
        self, 
        policy: AccessPolicy, 
        request: AccessRequest,
        attributes: Dict[str, Any]
    ) -> bool:
        """判断策略是否适用于当前请求"""
        target = policy.target
        
        # 检查主体匹配
        if 'subject' in target:
            if not await self._match_attributes(target['subject'], attributes['subject']):
                return False
        
        # 检查资源匹配
        if 'resource' in target:
            if not await self._match_attributes(target['resource'], attributes['resource']):
                return False
        
        # 检查动作匹配
        if 'action' in target:
            if not await self._match_attributes(target['action'], attributes['action']):
                return False
        
        return True
    
    async def _match_attributes(self, target_attrs: Dict[str, Any], actual_attrs: Dict[str, Any]) -> bool:
        """匹配属性"""
        for key, expected_value in target_attrs.items():
            actual_value = actual_attrs.get(key)
            if not await self._compare_values(actual_value, expected_value, 'equals'):
                return False
        return True
    
    async def _evaluate_policy_rules(
        self, 
        policy: AccessPolicy, 
        attributes: Dict[str, Any]
    ) -> AccessDecision:
        """评估策略规则"""
        for rule in policy.rules:
            condition = rule.get('condition', {})
            effect = rule.get('effect', AccessDecision.DENY.value)
            
            if await self._evaluate_condition(condition, attributes):
                return AccessDecision(effect)
        
        return AccessDecision.NOT_APPLICABLE
    
    async def _evaluate_condition(
        self, 
        condition: Dict[str, Any], 
        attributes: Dict[str, Any]
    ) -> bool:
        """评估条件表达式"""
        if not condition:
            return True
        
        # 支持逻辑操作符
        if 'and' in condition:
            return all(await self._evaluate_condition(c, attributes) for c in condition['and'])
        
        if 'or' in condition:
            return any(await self._evaluate_condition(c, attributes) for c in condition['or'])
        
        if 'not' in condition:
            return not await self._evaluate_condition(condition['not'], attributes)
        
        # 属性比较
        if 'attribute' in condition:
            attr_path = condition['attribute']
            expected_value = condition.get('value')
            operator = condition.get('operator', 'equals')
            
            actual_value = await self._get_attribute_value(attr_path, attributes)
            return await self._compare_values(actual_value, expected_value, operator)
        
        return False
    
    async def _get_attribute_value(self, attr_path: str, attributes: Dict[str, Any]) -> Any:
        """根据属性路径获取属性值"""
        parts = attr_path.split('.')
        current = attributes
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    async def _compare_values(self, actual: Any, expected: Any, operator: str) -> bool:
        """比较属性值"""
        if operator == 'equals':
            return actual == expected
        elif operator == 'not_equals':
            return actual != expected
        elif operator == 'greater_than':
            return actual > expected
        elif operator == 'less_than':
            return actual < expected
        elif operator == 'contains':
            return expected in actual if isinstance(actual, (list, str)) else False
        elif operator == 'matches':
            return bool(re.match(expected, str(actual))) if expected else False
        else:
            return False
    
    async def _combine_decisions(self, decisions: List[Tuple[int, AccessDecision]]) -> AccessDecision:
        """组合决策"""
        if not decisions:
            return AccessDecision.DENY
        
        # 按优先级排序，拒绝优先
        sorted_decisions = sorted(decisions, key=lambda x: x[0], reverse=True)
        
        for priority, decision in sorted_decisions:
            if decision == AccessDecision.DENY:
                return AccessDecision.DENY
        
        for priority, decision in sorted_decisions:
            if decision == AccessDecision.PERMIT:
                return AccessDecision.PERMIT
        
        return AccessDecision.DENY

class AccessControlEngine:
    """访问控制引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policies: Dict[str, AccessPolicy] = {}
        self.roles: Dict[str, Role] = {}
        self.subjects: Dict[str, Subject] = {}
        self.evaluators: Dict[str, PolicyEvaluator] = {}
        self.access_logs: List[Dict[str, Any]] = []
        self.logger = get_logger(__name__)
        
    async def initialize(self):
        """初始化访问控制引擎"""
        await self._load_policies()
        await self._load_roles()
        await self._load_subjects()
        await self._setup_evaluators()
        self.logger.info("Access control engine initialized")
        
    async def _setup_evaluators(self):
        """设置策略评估器"""
        self.evaluators['rbac'] = RBACEvaluator(self.roles, self.subjects)
        self.evaluators['abac'] = ABACEvaluator(AttributeProvider())
        
    async def evaluate_access(self, request: AccessRequest) -> Dict[str, Any]:
        """评估访问请求"""
        start_time = time.time()
        
        try:
            # 预检查
            if not await self._pre_check_request(request):
                decision = AccessDecision.DENY
                reason = "Request pre-check failed"
            else:
                # 获取适用的策略
                applicable_policies = await self._get_applicable_policies(request)
                
                if not applicable_policies:
                    decision = AccessDecision.DENY
                    reason = "No applicable policies found"
                else:
                    # 评估策略
                    decision, reason = await self._evaluate_policies(request, applicable_policies)
            
            # 记录访问决策
            access_log = {
                'request_id': f"{request.subject_id}_{request.resource_id}_{int(request.timestamp)}",
                'subject_id': request.subject_id,
                'resource_id': request.resource_id,
                'action': request.action,
                'decision': decision.value,
                'reason': reason,
                'timestamp': request.timestamp,
                'evaluation_time_ms': (time.time() - start_time) * 1000,
                'context': request.context
            }
            
            self.access_logs.append(access_log)
            
            # 限制日志大小
            max_log_entries = self.config.get('max_log_entries', 10000)
            if len(self.access_logs) > max_log_entries:
                self.access_logs = self.access_logs[-8000:]  # 保留最近8000条
            
            return {
                'decision': decision,
                'reason': reason,
                'request_id': access_log['request_id'],
                'evaluation_time_ms': access_log['evaluation_time_ms']
            }
            
        except Exception as e:
            # 安全失败时拒绝访问
            self.logger.error(f"Access evaluation error: {e}")
            error_log = {
                'request_id': f"{request.subject_id}_{request.resource_id}_{int(request.timestamp)}_error",
                'subject_id': request.subject_id,
                'resource_id': request.resource_id,
                'action': request.action,
                'decision': AccessDecision.DENY.value,
                'reason': f"Evaluation error: {str(e)}",
                'timestamp': request.timestamp,
                'evaluation_time_ms': (time.time() - start_time) * 1000,
                'error': True
            }
            
            self.access_logs.append(error_log)
            
            return {
                'decision': AccessDecision.DENY,
                'reason': f"Access evaluation failed: {str(e)}",
                'request_id': error_log['request_id'],
                'evaluation_time_ms': error_log['evaluation_time_ms']
            }
    
    async def _pre_check_request(self, request: AccessRequest) -> bool:
        """预检查访问请求"""
        # 检查主体是否存在且活跃
        subject = self.subjects.get(request.subject_id)
        if not subject or not subject.active:
            return False
        
        # 检查请求格式
        if not request.resource_id or not request.action:
            return False
        
        # 检查时间窗口
        current_time = time.time()
        max_time_skew = self.config.get('max_time_skew', 300)
        if abs(current_time - request.timestamp) > max_time_skew:
            return False
        
        return True
    
    async def _get_applicable_policies(self, request: AccessRequest) -> List[AccessPolicy]:
        """获取适用的策略"""
        applicable_policies = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            if await self._is_policy_applicable_to_request(policy, request):
                applicable_policies.append(policy)
        
        # 按优先级排序
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)
        
        return applicable_policies
    
    async def _is_policy_applicable_to_request(
        self, 
        policy: AccessPolicy, 
        request: AccessRequest
    ) -> bool:
        """判断策略是否适用于请求"""
        target = policy.target
        
        # 检查资源类型
        if 'resource_type' in target:
            if target['resource_type'] != request.resource_type.value:
                return False
        
        # 检查主体
        if 'subjects' in target:
            if request.subject_id not in target['subjects']:
                return False
        
        # 检查资源
        if 'resources' in target:
            resource_patterns = target['resources']
            if not any(re.match(pattern, request.resource_id) for pattern in resource_patterns):
                return False
        
        # 检查动作
        if 'actions' in target:
            if request.action not in target['actions']:
                return False
        
        return True
    
    async def _evaluate_policies(
        self, 
        request: AccessRequest, 
        policies: List[AccessPolicy]
    ) -> Tuple[AccessDecision, str]:
        """评估策略列表"""
        decisions = []
        reasons = []
        
        for policy in policies:
            evaluator_type = policy.target.get('evaluator', 'rbac')
            evaluator = self.evaluators.get(evaluator_type)
            
            if evaluator:
                decision = await evaluator.evaluate(request, [policy])
                decisions.append((policy.priority, decision, policy.name))
                
                if decision == AccessDecision.PERMIT:
                    reasons.append(f"Policy '{policy.name}' permits access")
                elif decision == AccessDecision.DENY:
                    reasons.append(f"Policy '{policy.name}' denies access")
        
        # 组合决策 - 拒绝优先
        for priority, decision, policy_name in sorted(decisions, reverse=True):
            if decision == AccessDecision.DENY:
                return AccessDecision.DENY, f"Access denied by policy '{policy_name}'"
        
        for priority, decision, policy_name in sorted(decisions, reverse=True):
            if decision == AccessDecision.PERMIT:
                return AccessDecision.PERMIT, f"Access permitted by policy '{policy_name}'"
        
        return AccessDecision.DENY, "No policy permits access"
    
    async def add_policy(self, policy: AccessPolicy) -> bool:
        """添加访问策略"""
        try:
            if await self._validate_policy(policy):
                self.policies[policy.policy_id] = policy
                await self._log_policy_change('add', policy.policy_id)
                self.logger.info(f"Policy added: {policy.policy_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to add policy: {e}")
            return False
    
    async def _validate_policy(self, policy: AccessPolicy) -> bool:
        """验证策略格式"""
        if not policy.policy_id or not policy.name:
            return False
        
        if not policy.target or not policy.rules:
            return False
        
        return True
    
    async def add_role(self, role: Role) -> bool:
        """添加角色"""
        try:
            self.roles[role.role_id] = role
            self.logger.info(f"Role added: {role.role_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add role: {e}")
            return False
    
    async def add_subject(self, subject: Subject) -> bool:
        """添加主体"""
        try:
            self.subjects[subject.subject_id] = subject
            self.logger.info(f"Subject added: {subject.subject_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add subject: {e}")
            return False
    
    async def get_access_logs(
        self, 
        subject_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """获取访问日志"""
        logs = self.access_logs
        
        if subject_id:
            logs = [log for log in logs if log['subject_id'] == subject_id]
        
        if resource_id:
            logs = [log for log in logs if log['resource_id'] == resource_id]
        
        return logs[-limit:]
    
    async def _load_policies(self):
        """加载默认策略"""
        # 管理员策略
        admin_policy = AccessPolicy(
            policy_id='admin_policy',
            name='Administrator Policy',
            description='Full access for administrators',
            target={'subjects': ['admin']},
            rules=[{
                'condition': {},
                'effect': 'permit'
            }],
            priority=100
        )
        self.policies[admin_policy.policy_id] = admin_policy
        
        # 基础访问策略
        basic_policy = AccessPolicy(
            policy_id='basic_policy',
            name='Basic Access Policy',
            description='Basic access for authenticated agents',
            target={'resource_type': 'api_endpoint'},
            rules=[{
                'condition': {
                    'attribute': 'subject.authenticated',
                    'value': True,
                    'operator': 'equals'
                },
                'effect': 'permit'
            }],
            priority=10
        )
        self.policies[basic_policy.policy_id] = basic_policy
    
    async def _load_roles(self):
        """加载默认角色"""
        # 管理员角色
        admin_role = Role(
            role_id='admin',
            name='Administrator',
            description='System administrator',
            permissions={'*:*:*'}
        )
        self.roles[admin_role.role_id] = admin_role
        
        # 智能体角色
        agent_role = Role(
            role_id='agent',
            name='Agent',
            description='AI agent',
            permissions={
                'api_endpoint:read:*',
                'api_endpoint:write:own_data/*',
                'ai_model:invoke:*'
            }
        )
        self.roles[agent_role.role_id] = agent_role
        
        # 用户角色
        user_role = Role(
            role_id='user',
            name='User',
            description='Regular user',
            permissions={
                'api_endpoint:read:public/*',
                'api_endpoint:write:user_data/*'
            }
        )
        self.roles[user_role.role_id] = user_role
    
    async def _load_subjects(self):
        """加载默认主体"""
        # 管理员主体
        admin_subject = Subject(
            subject_id='admin',
            roles={'admin'},
            attributes={'authenticated': True, 'security_level': 'high'}
        )
        self.subjects[admin_subject.subject_id] = admin_subject
    
    async def _log_policy_change(self, action: str, policy_id: str):
        """记录策略变更"""
        self.logger.info(f"Policy {action}: {policy_id}")

class AttributeProvider:
    """属性提供器"""
    
    def __init__(self):
        self.subject_attributes = {}
        self.resource_attributes = {}
        self.environment_attributes = {}
    
    async def get_subject_attributes(self, subject_id: str) -> Dict[str, Any]:
        """获取主体属性"""
        return self.subject_attributes.get(subject_id, {
            'authenticated': True,
            'agent_type': 'standard',
            'security_level': 'medium'
        })
    
    async def get_resource_attributes(self, resource_id: str) -> Dict[str, Any]:
        """获取资源属性"""
        return self.resource_attributes.get(resource_id, {
            'classification': 'internal',
            'sensitivity': 'medium'
        })
    
    async def get_environment_attributes(self) -> Dict[str, Any]:
        """获取环境属性"""
        return {
            'current_time': time.time(),
            'day_of_week': time.strftime('%A'),
            'hour_of_day': int(time.strftime('%H')),
            'network_zone': 'internal'
        }

class DynamicPermissionEvaluator:
    """动态权限评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.permission_cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5分钟
        self.logger = get_logger(__name__)
    
    async def evaluate_dynamic_permission(
        self,
        subject_id: str,
        resource_id: str,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """动态评估权限"""
        cache_key = f"{subject_id}:{resource_id}:{action}"
        
        # 检查缓存
        if cache_key in self.permission_cache:
            cached_result, cached_time = self.permission_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result
        
        # 动态评估
        result = await self._perform_dynamic_evaluation(subject_id, resource_id, action, context)
        
        # 更新缓存
        self.permission_cache[cache_key] = (result, time.time())
        
        return result
    
    async def _perform_dynamic_evaluation(
        self,
        subject_id: str,
        resource_id: str,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """执行动态评估"""
        # 基于上下文的动态权限评估
        risk_factors = []
        
        # 时间因素
        current_hour = int(time.strftime('%H'))
        if current_hour < 6 or current_hour > 22:
            risk_factors.append('off_hours_access')
        
        # 地理位置因素
        client_ip = context.get('client_ip', '')
        if client_ip and not client_ip.startswith('192.168.'):
            risk_factors.append('external_access')
        
        # 访问频率因素
        request_rate = context.get('request_rate', 0)
        if request_rate > 50:
            risk_factors.append('high_frequency_access')
        
        # 根据风险因素调整权限
        if len(risk_factors) >= 2:
            self.logger.warning(f"High risk access attempt: {subject_id} -> {resource_id}")
            return False
        
        return True
