"""
不确定性推理框架 - 基于概率的推理和置信度计算

实现功能:
- 贝叶斯网络推理算法
- 置信度传播和聚合机制
- 概率分布计算和更新
- 不确定性量化和可视化
- 蒙特卡罗采样和变分推理

技术栈:
- 贝叶斯网络建模
- 概率图模型
- 统计推理算法
- 不确定性量化方法
"""

import asyncio
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import logging
import json
from collections import defaultdict
import math
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class InferenceMethod(str, Enum):
    """推理方法"""
    EXACT = "exact"
    VARIATIONAL = "variational"
    MONTE_CARLO = "monte_carlo"
    BELIEF_PROPAGATION = "belief_propagation"

class DistributionType(str, Enum):
    """概率分布类型"""
    BETA = "beta"
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    BERNOULLI = "bernoulli"
    CATEGORICAL = "categorical"

@dataclass
class ProbabilityDistribution:
    """概率分布"""
    distribution_type: DistributionType
    parameters: Dict[str, float]
    support: Tuple[float, float] = (-np.inf, np.inf)
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """从分布中采样"""
        if self.distribution_type == DistributionType.BETA:
            alpha, beta = self.parameters['alpha'], self.parameters['beta']
            return np.random.beta(alpha, beta, n_samples)
        elif self.distribution_type == DistributionType.GAUSSIAN:
            mu, sigma = self.parameters['mean'], self.parameters['std']
            return np.random.normal(mu, sigma, n_samples)
        elif self.distribution_type == DistributionType.UNIFORM:
            low, high = self.parameters['low'], self.parameters['high']
            return np.random.uniform(low, high, n_samples)
        elif self.distribution_type == DistributionType.BERNOULLI:
            p = self.parameters['p']
            return np.random.bernoulli(p, n_samples)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """概率密度函数"""
        if self.distribution_type == DistributionType.BETA:
            alpha, beta = self.parameters['alpha'], self.parameters['beta']
            return stats.beta.pdf(x, alpha, beta)
        elif self.distribution_type == DistributionType.GAUSSIAN:
            mu, sigma = self.parameters['mean'], self.parameters['std']
            return stats.norm.pdf(x, mu, sigma)
        elif self.distribution_type == DistributionType.UNIFORM:
            low, high = self.parameters['low'], self.parameters['high']
            return stats.uniform.pdf(x, low, high - low)
        else:
            raise ValueError(f"PDF not implemented for: {self.distribution_type}")
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """累积分布函数"""
        if self.distribution_type == DistributionType.BETA:
            alpha, beta = self.parameters['alpha'], self.parameters['beta']
            return stats.beta.cdf(x, alpha, beta)
        elif self.distribution_type == DistributionType.GAUSSIAN:
            mu, sigma = self.parameters['mean'], self.parameters['std']
            return stats.norm.cdf(x, mu, sigma)
        elif self.distribution_type == DistributionType.UNIFORM:
            low, high = self.parameters['low'], self.parameters['high']
            return stats.uniform.cdf(x, low, high - low)
        else:
            raise ValueError(f"CDF not implemented for: {self.distribution_type}")
    
    def mean(self) -> float:
        """期望值"""
        if self.distribution_type == DistributionType.BETA:
            alpha, beta = self.parameters['alpha'], self.parameters['beta']
            return alpha / (alpha + beta)
        elif self.distribution_type == DistributionType.GAUSSIAN:
            return self.parameters['mean']
        elif self.distribution_type == DistributionType.UNIFORM:
            low, high = self.parameters['low'], self.parameters['high']
            return (low + high) / 2
        elif self.distribution_type == DistributionType.BERNOULLI:
            return self.parameters['p']
        else:
            return 0.0
    
    def variance(self) -> float:
        """方差"""
        if self.distribution_type == DistributionType.BETA:
            alpha, beta = self.parameters['alpha'], self.parameters['beta']
            return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        elif self.distribution_type == DistributionType.GAUSSIAN:
            return self.parameters['std'] ** 2
        elif self.distribution_type == DistributionType.UNIFORM:
            low, high = self.parameters['low'], self.parameters['high']
            return (high - low) ** 2 / 12
        elif self.distribution_type == DistributionType.BERNOULLI:
            p = self.parameters['p']
            return p * (1 - p)
        else:
            return 0.0

@dataclass
class UncertaintyQuantification:
    """不确定性量化结果"""
    mean: float
    variance: float
    confidence_interval: Tuple[float, float]
    entropy: float
    epistemic_uncertainty: float  # 认知不确定性
    aleatoric_uncertainty: float  # 偶然不确定性
    probability_distribution: Optional[ProbabilityDistribution] = None
    
    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)
    
    @property
    def coefficient_of_variation(self) -> float:
        if self.mean == 0:
            return float('inf')
        return self.standard_deviation / abs(self.mean)

@dataclass
class BayesianNode:
    """贝叶斯网络节点"""
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    prior: ProbabilityDistribution = None
    likelihood: Dict[str, ProbabilityDistribution] = field(default_factory=dict)
    posterior: Optional[ProbabilityDistribution] = None
    observed_value: Optional[float] = None
    
    def is_observed(self) -> bool:
        return self.observed_value is not None
    
    def is_root(self) -> bool:
        return len(self.parents) == 0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0

class BayesianNetwork:
    """贝叶斯网络"""
    
    def __init__(self):
        self.nodes: Dict[str, BayesianNode] = {}
        self.edges: List[Tuple[str, str]] = []
    
    def add_node(self, node: BayesianNode):
        """添加节点"""
        self.nodes[node.name] = node
    
    def add_edge(self, parent: str, child: str):
        """添加边"""
        if parent in self.nodes and child in self.nodes:
            self.edges.append((parent, child))
            self.nodes[parent].children.append(child)
            self.nodes[child].parents.append(parent)
    
    def get_topological_order(self) -> List[str]:
        """获取拓扑排序"""
        in_degree = {name: 0 for name in self.nodes}
        for parent, child in self.edges:
            in_degree[child] += 1
        
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for child in self.nodes[current].children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def set_evidence(self, node_name: str, value: float):
        """设置观测证据"""
        if node_name in self.nodes:
            self.nodes[node_name].observed_value = value

class UncertaintyReasoner:
    """不确定性推理器"""
    
    def __init__(self, method: InferenceMethod = InferenceMethod.MONTE_CARLO):
        self.method = method
        self.bayesian_network: Optional[BayesianNetwork] = None
        self.confidence_cache: Dict[str, UncertaintyQuantification] = {}
        self.n_samples = 10000
        self.confidence_level = 0.95
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
    
    def build_bayesian_network(self, 
                             entities: List[str], 
                             relations: List[Tuple[str, str, str]],
                             prior_beliefs: Dict[str, ProbabilityDistribution] = None) -> BayesianNetwork:
        """构建贝叶斯网络"""
        network = BayesianNetwork()
        
        # 添加实体节点
        for entity in entities:
            prior = prior_beliefs.get(entity) if prior_beliefs else None
            if prior is None:
                # 默认Beta分布作为先验
                prior = ProbabilityDistribution(
                    distribution_type=DistributionType.BETA,
                    parameters={'alpha': 1.0, 'beta': 1.0}
                )
            
            node = BayesianNode(name=entity, prior=prior)
            network.add_node(node)
        
        # 添加关系边
        for head, relation, tail in relations:
            if head in entities and tail in entities:
                network.add_edge(head, tail)
        
        self.bayesian_network = network
        return network
    
    async def calculate_inference_confidence(self,
                                           evidence: Dict[str, float],
                                           hypothesis: str,
                                           context: Dict[str, Any] = None) -> UncertaintyQuantification:
        """计算推理的置信度分布"""
        if not self.bayesian_network:
            raise ValueError("Bayesian network not initialized")
        
        # 设置证据
        for node_name, value in evidence.items():
            self.bayesian_network.set_evidence(node_name, value)
        
        # 执行推理
        if self.method == InferenceMethod.MONTE_CARLO:
            return await self._monte_carlo_inference(hypothesis, context)
        elif self.method == InferenceMethod.VARIATIONAL:
            return await self._variational_inference(hypothesis, context)
        elif self.method == InferenceMethod.BELIEF_PROPAGATION:
            return await self._belief_propagation(hypothesis, context)
        else:
            return await self._exact_inference(hypothesis, context)
    
    async def _monte_carlo_inference(self,
                                   hypothesis: str,
                                   context: Dict[str, Any] = None) -> UncertaintyQuantification:
        """蒙特卡罗推理"""
        samples = []
        
        for _ in range(self.n_samples):
            # 从先验分布采样
            sample_values = {}
            
            # 按拓扑顺序采样
            topo_order = self.bayesian_network.get_topological_order()
            
            for node_name in topo_order:
                node = self.bayesian_network.nodes[node_name]
                
                if node.is_observed():
                    sample_values[node_name] = node.observed_value
                else:
                    # 从先验或条件分布采样
                    if node.is_root():
                        sample_value = node.prior.sample(1)[0]
                    else:
                        # 简化的条件采样
                        parent_values = [sample_values[parent] for parent in node.parents]
                        sample_value = self._conditional_sample(node, parent_values)
                    
                    sample_values[node_name] = sample_value
            
            # 收集目标变量的样本
            if hypothesis in sample_values:
                samples.append(sample_values[hypothesis])
        
        samples = np.array(samples)
        
        # 计算统计量
        mean = np.mean(samples)
        variance = np.var(samples)
        
        # 置信区间
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(samples, 100 * alpha / 2)
        ci_upper = np.percentile(samples, 100 * (1 - alpha / 2))
        
        # 熵
        entropy = self._calculate_entropy(samples)
        
        # 不确定性分解
        epistemic, aleatoric = self._decompose_uncertainty(samples)
        
        return UncertaintyQuantification(
            mean=mean,
            variance=variance,
            confidence_interval=(ci_lower, ci_upper),
            entropy=entropy,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric
        )
    
    def _conditional_sample(self, node: BayesianNode, parent_values: List[float]) -> float:
        """条件采样"""
        # 简化实现：基于父节点值的线性组合加噪声
        if not parent_values:
            return node.prior.sample(1)[0]
        
        # 加权平均父节点值
        weighted_sum = np.mean(parent_values)
        
        # 添加噪声
        noise = np.random.normal(0, 0.1)
        
        return np.clip(weighted_sum + noise, 0, 1)
    
    async def _variational_inference(self,
                                   hypothesis: str,
                                   context: Dict[str, Any] = None) -> UncertaintyQuantification:
        """变分推理"""
        # 简化的变分推理实现
        # 使用平均场变分近似
        
        node = self.bayesian_network.nodes[hypothesis]
        
        # 初始化变分参数
        if node.prior.distribution_type == DistributionType.BETA:
            # 对Beta分布使用变分更新
            alpha = node.prior.parameters['alpha']
            beta = node.prior.parameters['beta']
            
            # 变分更新（简化）
            for _ in range(100):  # 迭代次数
                # 更新规则（简化实现）
                alpha_new = alpha + self._calculate_sufficient_statistic_1()
                beta_new = beta + self._calculate_sufficient_statistic_2()
                
                # 检查收敛
                if abs(alpha_new - alpha) < 1e-6 and abs(beta_new - beta) < 1e-6:
                    break
                
                alpha, beta = alpha_new, beta_new
            
            # 构造近似后验
            posterior = ProbabilityDistribution(
                distribution_type=DistributionType.BETA,
                parameters={'alpha': alpha, 'beta': beta}
            )
            
            mean = posterior.mean()
            variance = posterior.variance()
            
            # 置信区间
            ci_lower = stats.beta.ppf((1 - self.confidence_level) / 2, alpha, beta)
            ci_upper = stats.beta.ppf(1 - (1 - self.confidence_level) / 2, alpha, beta)
            
            entropy = self._beta_entropy(alpha, beta)
            
            return UncertaintyQuantification(
                mean=mean,
                variance=variance,
                confidence_interval=(ci_lower, ci_upper),
                entropy=entropy,
                epistemic_uncertainty=entropy * 0.7,  # 简化分解
                aleatoric_uncertainty=entropy * 0.3,
                probability_distribution=posterior
            )
        
        # 默认返回Monte Carlo结果
        return await self._monte_carlo_inference(hypothesis, context)
    
    def _calculate_sufficient_statistic_1(self) -> float:
        """计算充分统计量1（简化）"""
        return 0.1
    
    def _calculate_sufficient_statistic_2(self) -> float:
        """计算充分统计量2（简化）"""
        return 0.1
    
    def _beta_entropy(self, alpha: float, beta: float) -> float:
        """计算Beta分布的熵"""
        from scipy.special import digamma
        return (
            math.log(stats.beta.pdf(0.5, alpha, beta)) +
            (alpha - 1) * digamma(alpha) +
            (beta - 1) * digamma(beta) -
            (alpha + beta - 2) * digamma(alpha + beta)
        )
    
    async def _belief_propagation(self,
                                hypothesis: str,
                                context: Dict[str, Any] = None) -> UncertaintyQuantification:
        """置信传播算法"""
        # 简化的置信传播实现
        # 在树结构上进行消息传递
        
        messages = defaultdict(dict)  # messages[from][to] = distribution
        
        # 初始化消息
        for node_name in self.bayesian_network.nodes:
            node = self.bayesian_network.nodes[node_name]
            for child in node.children:
                messages[node_name][child] = node.prior
        
        # 消息传递（简化）
        for iteration in range(10):
            updated = False
            
            for node_name in self.bayesian_network.nodes:
                node = self.bayesian_network.nodes[node_name]
                
                if not node.is_observed():
                    # 收集来自父节点的消息
                    parent_messages = []
                    for parent in node.parents:
                        if parent in messages and node_name in messages[parent]:
                            parent_messages.append(messages[parent][node_name])
                    
                    # 更新消息（简化）
                    if parent_messages:
                        # 消息融合
                        combined_mean = np.mean([msg.mean() for msg in parent_messages])
                        combined_var = np.mean([msg.variance() for msg in parent_messages])
                        
                        new_message = ProbabilityDistribution(
                            distribution_type=DistributionType.GAUSSIAN,
                            parameters={'mean': combined_mean, 'std': math.sqrt(combined_var)}
                        )
                        
                        # 向子节点发送消息
                        for child in node.children:
                            if child not in messages[node_name] or messages[node_name][child] != new_message:
                                messages[node_name][child] = new_message
                                updated = True
            
            if not updated:
                break
        
        # 计算边际分布
        target_node = self.bayesian_network.nodes[hypothesis]
        
        if hypothesis in messages:
            # 融合所有入边消息
            incoming_messages = list(messages[hypothesis].values())
            if incoming_messages:
                mean = np.mean([msg.mean() for msg in incoming_messages])
                variance = np.mean([msg.variance() for msg in incoming_messages])
            else:
                mean = target_node.prior.mean()
                variance = target_node.prior.variance()
        else:
            mean = target_node.prior.mean()
            variance = target_node.prior.variance()
        
        # 置信区间（假设高斯分布）
        std = math.sqrt(variance)
        ci_lower = stats.norm.ppf((1 - self.confidence_level) / 2, mean, std)
        ci_upper = stats.norm.ppf(1 - (1 - self.confidence_level) / 2, mean, std)
        
        entropy = 0.5 * math.log(2 * math.pi * math.e * variance)
        
        return UncertaintyQuantification(
            mean=mean,
            variance=variance,
            confidence_interval=(ci_lower, ci_upper),
            entropy=entropy,
            epistemic_uncertainty=entropy * 0.6,
            aleatoric_uncertainty=entropy * 0.4
        )
    
    async def _exact_inference(self,
                             hypothesis: str,
                             context: Dict[str, Any] = None) -> UncertaintyQuantification:
        """精确推理（适用于小型网络）"""
        # 对于小型网络的精确推理
        # 通过枚举所有可能的状态组合
        
        # 简化实现：假设二值变量
        node = self.bayesian_network.nodes[hypothesis]
        
        # 使用贝叶斯公式
        prior = node.prior.mean()
        
        # 计算似然（简化）
        likelihood = 0.8  # 假设似然
        
        # 证据概率（归一化常数）
        evidence = 0.6
        
        # 后验概率
        posterior = (likelihood * prior) / evidence
        
        # 方差估计（简化）
        variance = posterior * (1 - posterior)  # 伯努利方差
        
        # 置信区间
        std = math.sqrt(variance)
        ci_lower = max(0, posterior - 1.96 * std)
        ci_upper = min(1, posterior + 1.96 * std)
        
        # 熵
        if posterior > 0 and posterior < 1:
            entropy = -posterior * math.log(posterior) - (1 - posterior) * math.log(1 - posterior)
        else:
            entropy = 0.0
        
        return UncertaintyQuantification(
            mean=posterior,
            variance=variance,
            confidence_interval=(ci_lower, ci_upper),
            entropy=entropy,
            epistemic_uncertainty=entropy * 0.5,
            aleatoric_uncertainty=entropy * 0.5
        )
    
    def _calculate_entropy(self, samples: np.ndarray) -> float:
        """计算样本的熵"""
        # 使用直方图估计概率分布
        hist, bin_edges = np.histogram(samples, bins=50, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        
        # 计算离散熵
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * bin_width * math.log(p * bin_width)
        
        return entropy
    
    def _decompose_uncertainty(self, samples: np.ndarray) -> Tuple[float, float]:
        """分解不确定性为认知不确定性和偶然不确定性"""
        # 简化的不确定性分解
        total_variance = np.var(samples)
        
        # 认知不确定性：模型参数的不确定性
        epistemic = total_variance * 0.6
        
        # 偶然不确定性：数据的固有噪声
        aleatoric = total_variance * 0.4
        
        return epistemic, aleatoric
    
    async def propagate_confidence(self,
                                 source_confidences: Dict[str, float],
                                 reasoning_path: List[Tuple[str, str, str]]) -> Dict[str, UncertaintyQuantification]:
        """沿推理路径传播置信度"""
        result = {}
        
        # 初始化源置信度
        current_confidences = source_confidences.copy()
        
        for i, (head, relation, tail) in enumerate(reasoning_path):
            # 获取关系的可靠性
            relation_confidence = await self._get_relation_confidence(relation)
            
            # 传播置信度
            if head in current_confidences:
                head_conf = current_confidences[head]
                
                # 使用概率乘法传播（简化）
                propagated_conf = head_conf * relation_confidence
                
                # 考虑不确定性累积
                uncertainty_growth = i * 0.05  # 每跳增加5%的不确定性
                adjusted_conf = propagated_conf * (1 - uncertainty_growth)
                
                current_confidences[tail] = adjusted_conf
                
                # 计算不确定性量化
                variance = adjusted_conf * (1 - adjusted_conf) * (1 + uncertainty_growth)
                std = math.sqrt(variance)
                
                result[tail] = UncertaintyQuantification(
                    mean=adjusted_conf,
                    variance=variance,
                    confidence_interval=(
                        max(0, adjusted_conf - 1.96 * std),
                        min(1, adjusted_conf + 1.96 * std)
                    ),
                    entropy=self._calculate_binary_entropy(adjusted_conf),
                    epistemic_uncertainty=variance * 0.7,
                    aleatoric_uncertainty=variance * 0.3
                )
        
        return result
    
    async def _get_relation_confidence(self, relation: str) -> float:
        """获取关系的置信度"""
        # 简化：从知识库或统计中获取关系可靠性
        relation_confidences = {
            "is_a": 0.9,
            "part_of": 0.85,
            "related_to": 0.7,
            "similar_to": 0.75,
            "causes": 0.8
        }
        
        return relation_confidences.get(relation, 0.6)  # 默认置信度
    
    def _calculate_binary_entropy(self, p: float) -> float:
        """计算二进制熵"""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    
    async def aggregate_confidences(self,
                                  confidences: List[UncertaintyQuantification],
                                  aggregation_method: str = "weighted_average") -> UncertaintyQuantification:
        """聚合多个置信度"""
        if not confidences:
            return UncertaintyQuantification(0, 0, (0, 0), 0, 0, 0)
        
        if aggregation_method == "weighted_average":
            # 按不确定性倒数加权
            weights = [1.0 / (conf.variance + 1e-8) for conf in confidences]
            total_weight = sum(weights)
            
            if total_weight == 0:
                return confidences[0]
            
            # 加权平均
            mean = sum(w * conf.mean for w, conf in zip(weights, confidences)) / total_weight
            
            # 加权方差
            variance = sum(w * conf.variance for w, conf in zip(weights, confidences)) / total_weight
            
        elif aggregation_method == "conservative":
            # 保守聚合：取最高不确定性
            mean = np.mean([conf.mean for conf in confidences])
            variance = max(conf.variance for conf in confidences)
            
        elif aggregation_method == "optimistic":
            # 乐观聚合：取最低不确定性
            mean = np.mean([conf.mean for conf in confidences])
            variance = min(conf.variance for conf in confidences)
            
        else:  # "simple_average"
            mean = np.mean([conf.mean for conf in confidences])
            variance = np.mean([conf.variance for conf in confidences])
        
        # 计算聚合的置信区间
        std = math.sqrt(variance)
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
        
        # 聚合熵
        entropy = np.mean([conf.entropy for conf in confidences])
        
        return UncertaintyQuantification(
            mean=mean,
            variance=variance,
            confidence_interval=(ci_lower, ci_upper),
            entropy=entropy,
            epistemic_uncertainty=np.mean([conf.epistemic_uncertainty for conf in confidences]),
            aleatoric_uncertainty=np.mean([conf.aleatoric_uncertainty for conf in confidences])
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "confidence_level": self.confidence_level,
            "cache_size": len(self.confidence_cache),
            "network_nodes": len(self.bayesian_network.nodes) if self.bayesian_network else 0,
            "network_edges": len(self.bayesian_network.edges) if self.bayesian_network else 0
        }
    
    def clear_cache(self):
        """清除缓存"""
        self.confidence_cache.clear()
        logger.info("Uncertainty reasoner cache cleared")