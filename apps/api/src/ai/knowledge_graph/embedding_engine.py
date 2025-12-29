"""
知识图谱嵌入推理引擎 - 基于向量表示的关系预测和推理

实现功能:
- TransE、RotatE等嵌入模型集成
- 实体和关系嵌入训练
- 基于相似度的关系预测
- 嵌入向量存储和索引
- 动态嵌入更新机制

技术栈:
- PyTorch深度学习框架
- FAISS高效向量检索
- 异步训练和推理
- GPU/CPU自适应计算
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from src.core.utils import secure_pickle as pickle
import json
from pathlib import Path
import faiss
from concurrent.futures import ThreadPoolExecutor
import threading

from src.core.logging import get_logger
logger = get_logger(__name__)

class EmbeddingModel(str, Enum):
    """嵌入模型类型"""
    TRANSE = "TransE"
    ROTATE = "RotatE"
    COMPLEX = "ComplEx"
    DISTMULT = "DistMult"

class DistanceMetric(str, Enum):
    """距离度量类型"""
    L1 = "L1"  # 曼哈顿距离
    L2 = "L2"  # 欧几里得距离
    COSINE = "cosine"  # 余弦相似度

@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_type: EmbeddingModel = EmbeddingModel.TRANSE
    embedding_dim: int = 100
    learning_rate: float = 0.001
    batch_size: int = 1024
    max_epochs: int = 100
    negative_sampling_rate: int = 10
    margin: float = 1.0
    regularization_weight: float = 0.001
    distance_metric: DistanceMetric = DistanceMetric.L2
    device: str = "auto"  # "auto", "cpu", "cuda"

@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    loss: float
    hits_at_1: float = 0.0
    hits_at_10: float = 0.0
    mean_rank: float = 0.0
    mean_reciprocal_rank: float = 0.0
    timestamp: datetime = field(default_factory=utc_now)

@dataclass
class EmbeddingPrediction:
    """嵌入预测结果"""
    head: str
    relation: str
    tail: str
    score: float
    confidence: float
    rank: int

@dataclass
class SimilarEntity:
    """相似实体结果"""
    entity: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimilarityResult:
    """相似度查询返回"""
    query_entity: str
    similar_entities: List[SimilarEntity]
    execution_time: float = 0.0
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class TransE(nn.Module):
    """TransE知识图谱嵌入模型"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, margin: float = 1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # 实体和关系嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # 归一化关系向量
        self._normalize_relations()
    
    def _normalize_relations(self):
        """归一化关系嵌入向量"""
        with torch.no_grad():
            rel_norms = self.relation_embeddings.weight.norm(dim=1, keepdim=True)
            self.relation_embeddings.weight.div_(rel_norms + 1e-8)
    
    def forward(self, heads, relations, tails):
        """前向传播计算三元组得分"""
        head_embeds = self.entity_embeddings(heads)
        rel_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)
        
        # TransE: h + r ≈ t
        scores = head_embeds + rel_embeds - tail_embeds
        return torch.norm(scores, p=2, dim=1)
    
    def predict_tail(self, head_id: int, relation_id: int) -> torch.Tensor:
        """预测尾实体"""
        head_embed = self.entity_embeddings(torch.tensor([head_id]))
        rel_embed = self.relation_embeddings(torch.tensor([relation_id]))
        
        # 计算与所有实体的距离
        target = head_embed + rel_embed
        entity_embeds = self.entity_embeddings.weight
        
        distances = torch.norm(target - entity_embeds, p=2, dim=1)
        return distances

class RotatE(nn.Module):
    """RotatE知识图谱嵌入模型"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, margin: float = 6.0):
        super(RotatE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # 实体嵌入（复数）
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)
        
        # 关系嵌入（旋转角度）
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, heads, relations, tails):
        """前向传播"""
        head_embeds = self.entity_embeddings(heads)
        rel_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)
        
        # 分离实部和虚部
        head_re, head_im = torch.chunk(head_embeds, 2, dim=1)
        tail_re, tail_im = torch.chunk(tail_embeds, 2, dim=1)
        
        # 计算旋转
        rel_cos = torch.cos(rel_embeds)
        rel_sin = torch.sin(rel_embeds)
        
        # 复数乘法: h * r
        rotated_head_re = head_re * rel_cos - head_im * rel_sin
        rotated_head_im = head_re * rel_sin + head_im * rel_cos
        
        # 计算距离
        diff_re = rotated_head_re - tail_re
        diff_im = rotated_head_im - tail_im
        
        scores = torch.sqrt(diff_re ** 2 + diff_im ** 2).sum(dim=1)
        return scores

class EmbeddingEngine:
    """知识图谱嵌入推理引擎"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.device = self._get_device()
        
        # 模型和索引
        self.model: Optional[nn.Module] = None
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_relation: Dict[int, str] = {}
        self.faiss_index: Optional[faiss.Index] = None
        
        # 训练状态
        self.is_trained = False
        self.training_metrics: List[TrainingMetrics] = []
        self.optimizer = None
        
        # 异步执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    async def prepare_training_data(self, graph_data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        """准备训练数据"""
        entities = set()
        relations = set()
        triples = []
        
        # 提取三元组数据
        if "triples" in graph_data:
            for triple in graph_data["triples"]:
                head, relation, tail = triple
                entities.update([head, tail])
                relations.add(relation)
                triples.append((head, relation, tail))
        
        # 构建实体和关系映射
        self.entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
        self.id_to_entity = {idx: entity for entity, idx in self.entity_to_id.items()}
        self.relation_to_id = {relation: idx for idx, relation in enumerate(sorted(relations))}
        self.id_to_relation = {idx: relation for relation, idx in self.relation_to_id.items()}
        
        # 转换为张量
        triple_tensor = torch.tensor([
            [self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]]
            for h, r, t in triples
        ], dtype=torch.long)
        
        metadata = {
            "num_entities": len(entities),
            "num_relations": len(relations),
            "num_triples": len(triples)
        }
        
        logger.info(f"Prepared training data: {metadata}")
        return triple_tensor, metadata
    
    async def train_embeddings(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """训练知识图谱嵌入"""
        try:
            # 准备数据
            triples, metadata = await self.prepare_training_data(graph_data)
            
            # 初始化模型
            self.model = self._create_model(
                metadata["num_entities"], 
                metadata["num_relations"]
            )
            self.model.to(self.device)
            
            # 初始化优化器
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
            
            # 训练循环
            training_results = await self._training_loop(triples, metadata)
            
            # 构建FAISS索引
            await self._build_faiss_index()
            
            self.is_trained = True
            logger.info("Embedding training completed successfully")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def _create_model(self, num_entities: int, num_relations: int) -> nn.Module:
        """创建嵌入模型"""
        if self.config.model_type == EmbeddingModel.TRANSE:
            return TransE(
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=self.config.embedding_dim,
                margin=self.config.margin
            )
        elif self.config.model_type == EmbeddingModel.ROTATE:
            return RotatE(
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=self.config.embedding_dim,
                margin=self.config.margin
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    async def _training_loop(self, triples: torch.Tensor, metadata: Dict) -> Dict[str, Any]:
        """训练循环"""
        triples = triples.to(self.device)
        num_triples = len(triples)
        
        for epoch in range(self.config.max_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # 随机打乱训练数据
            perm = torch.randperm(num_triples)
            triples_shuffled = triples[perm]
            
            for i in range(0, num_triples, self.config.batch_size):
                batch = triples_shuffled[i:i + self.config.batch_size]
                
                # 生成负样本
                negative_batch = self._generate_negative_samples(batch, metadata)
                
                # 计算损失
                loss = self._compute_loss(batch, negative_batch)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # 评估模型（每10个epoch）
            metrics = TrainingMetrics(epoch=epoch, loss=avg_loss)
            if epoch % 10 == 0:
                eval_results = await self._evaluate_model(triples[:1000])  # 采样评估
                metrics.hits_at_1 = eval_results.get("hits@1", 0.0)
                metrics.hits_at_10 = eval_results.get("hits@10", 0.0)
                metrics.mean_rank = eval_results.get("mean_rank", 0.0)
                metrics.mean_reciprocal_rank = eval_results.get("mrr", 0.0)
            
            self.training_metrics.append(metrics)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Hits@1={metrics.hits_at_1:.3f}")
        
        return {
            "final_loss": self.training_metrics[-1].loss,
            "best_hits_at_1": max(m.hits_at_1 for m in self.training_metrics),
            "training_epochs": len(self.training_metrics),
            "model_parameters": sum(p.numel() for p in self.model.parameters())
        }
    
    def _generate_negative_samples(self, positive_batch: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """生成负样本"""
        batch_size = len(positive_batch)
        neg_size = batch_size * self.config.negative_sampling_rate
        
        negative_samples = []
        
        for _ in range(self.config.negative_sampling_rate):
            # 复制正样本
            neg_batch = positive_batch.clone()
            
            # 随机替换头或尾实体
            for i in range(batch_size):
                if torch.rand(1) < 0.5:
                    # 替换头实体
                    neg_batch[i, 0] = torch.randint(0, metadata["num_entities"], (1,))
                else:
                    # 替换尾实体
                    neg_batch[i, 2] = torch.randint(0, metadata["num_entities"], (1,))
            
            negative_samples.append(neg_batch)
        
        return torch.cat(negative_samples, dim=0)
    
    def _compute_loss(self, positive_batch: torch.Tensor, negative_batch: torch.Tensor) -> torch.Tensor:
        """计算训练损失"""
        # 正样本得分
        pos_heads, pos_rels, pos_tails = positive_batch.t()
        pos_scores = self.model(pos_heads, pos_rels, pos_tails)
        
        # 负样本得分
        neg_heads, neg_rels, neg_tails = negative_batch.t()
        neg_scores = self.model(neg_heads, neg_rels, neg_tails)
        
        # Margin ranking loss
        margin = self.config.margin
        if self.config.model_type == EmbeddingModel.ROTATE:
            # RotatE使用不同的损失函数
            pos_loss = -F.logsigmoid(-pos_scores + margin).mean()
            neg_loss = -F.logsigmoid(neg_scores - margin).mean()
            loss = pos_loss + neg_loss
        else:
            # TransE等使用margin ranking loss
            loss = F.relu(pos_scores - neg_scores + margin).mean()
        
        # 添加正则化
        if self.config.regularization_weight > 0:
            l2_reg = torch.tensor(0.0, device=self.device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param, p=2) ** 2
            loss += self.config.regularization_weight * l2_reg
        
        return loss
    
    async def _evaluate_model(self, test_triples: torch.Tensor) -> Dict[str, float]:
        """评估模型性能"""
        self.model.eval()
        
        hits_at_1 = 0
        hits_at_10 = 0
        total_rank = 0
        reciprocal_rank = 0
        
        with torch.no_grad():
            for triple in test_triples:
                head, relation, tail = triple
                
                # 预测尾实体
                if hasattr(self.model, 'predict_tail'):
                    scores = self.model.predict_tail(head.item(), relation.item())
                    
                    # 排序获得排名
                    _, sorted_indices = torch.sort(scores)
                    rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1
                    
                    if rank == 1:
                        hits_at_1 += 1
                    if rank <= 10:
                        hits_at_10 += 1
                    
                    total_rank += rank
                    reciprocal_rank += 1.0 / rank
        
        num_samples = len(test_triples)
        
        self.model.train()
        
        return {
            "hits@1": hits_at_1 / num_samples,
            "hits@10": hits_at_10 / num_samples,
            "mean_rank": total_rank / num_samples,
            "mrr": reciprocal_rank / num_samples
        }
    
    async def _build_faiss_index(self):
        """构建FAISS向量索引"""
        if not self.model or not self.is_trained:
            return
        
        try:
            # 获取所有实体嵌入
            entity_embeddings = self.model.entity_embeddings.weight.detach().cpu().numpy()
            
            if self.config.model_type == EmbeddingModel.ROTATE:
                # RotatE需要处理复数嵌入
                entity_embeddings = entity_embeddings[:, :self.config.embedding_dim]
            
            # 创建FAISS索引
            dimension = entity_embeddings.shape[1]
            if self.config.distance_metric == DistanceMetric.COSINE:
                self.faiss_index = faiss.IndexFlatIP(dimension)
                # 归一化向量用于余弦相似度
                faiss.normalize_L2(entity_embeddings)
            else:
                self.faiss_index = faiss.IndexFlatL2(dimension)
            
            # 添加向量到索引
            self.faiss_index.add(entity_embeddings.astype(np.float32))
            
            logger.info(f"Built FAISS index with {len(entity_embeddings)} entities")
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {str(e)}")
    
    async def predict_relations(self, head: str, tail: str, top_k: int = 5) -> List[EmbeddingPrediction]:
        """预测实体间可能的关系"""
        if not self.is_trained or head not in self.entity_to_id or tail not in self.entity_to_id:
            return []
        
        head_id = self.entity_to_id[head]
        tail_id = self.entity_to_id[tail]
        
        predictions = []
        
        try:
            with torch.no_grad():
                head_embed = self.model.entity_embeddings(torch.tensor([head_id], device=self.device))
                tail_embed = self.model.entity_embeddings(torch.tensor([tail_id], device=self.device))
                
                # 计算与所有关系的得分
                relation_scores = []
                for rel_name, rel_id in self.relation_to_id.items():
                    rel_tensor = torch.tensor([rel_id], device=self.device)
                    
                    if self.config.model_type == EmbeddingModel.TRANSE:
                        # TransE: h + r ≈ t
                        rel_embed = self.model.relation_embeddings(rel_tensor)
                        score = torch.norm(head_embed + rel_embed - tail_embed, p=2).item()
                        # 转换为相似度（距离越小相似度越高）
                        similarity = 1.0 / (1.0 + score)
                    else:
                        # 其他模型的得分计算
                        score = self.model(torch.tensor([head_id]), rel_tensor, torch.tensor([tail_id])).item()
                        similarity = 1.0 / (1.0 + score)
                    
                    relation_scores.append((rel_name, score, similarity))
                
                # 按相似度排序
                relation_scores.sort(key=lambda x: x[2], reverse=True)
                
                # 构造预测结果
                for rank, (relation, score, confidence) in enumerate(relation_scores[:top_k], 1):
                    prediction = EmbeddingPrediction(
                        head=head,
                        relation=relation,
                        tail=tail,
                        score=score,
                        confidence=confidence,
                        rank=rank
                    )
                    predictions.append(prediction)
        
        except Exception as e:
            logger.error(f"Relation prediction failed: {str(e)}")
        
        return predictions
    
    async def find_similar_entities(self, entity: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """查找相似实体"""
        if not self.faiss_index or entity not in self.entity_to_id:
            return []
        
        try:
            entity_id = self.entity_to_id[entity]
            
            # 获取实体嵌入
            entity_embed = self.model.entity_embeddings.weight[entity_id].detach().cpu().numpy()
            
            if self.config.model_type == EmbeddingModel.ROTATE:
                entity_embed = entity_embed[:self.config.embedding_dim]
            
            # 归一化（如果使用余弦相似度）
            if self.config.distance_metric == DistanceMetric.COSINE:
                entity_embed = entity_embed / np.linalg.norm(entity_embed)
            
            # 搜索相似向量
            scores, indices = self.faiss_index.search(
                entity_embed.reshape(1, -1).astype(np.float32), 
                top_k + 1  # +1因为会包含自身
            )
            
            # 构造结果（排除自身）
            similar_entities = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != entity_id:  # 排除自身
                    similar_entity = self.id_to_entity[idx]
                    
                    if self.config.distance_metric == DistanceMetric.COSINE:
                        similarity = float(score)  # FAISS返回的是余弦相似度
                    else:
                        similarity = 1.0 / (1.0 + float(score))  # 距离转换为相似度
                    
                    similar_entities.append((similar_entity, similarity))
            
            return similar_entities[:top_k]
            
        except Exception as e:
            logger.error(f"Similar entity search failed: {str(e)}")
            return []
    
    async def save_model(self, model_path: str):
        """保存模型"""
        if not self.model:
            raise ValueError("No model to save")
        
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "entity_to_id": self.entity_to_id,
            "relation_to_id": self.relation_to_id,
            "training_metrics": [m.__dict__ for m in self.training_metrics]
        }
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    async def load_model(self, model_path: str):
        """加载模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 恢复配置
            config_dict = model_data["config"]
            self.config = EmbeddingConfig(**config_dict)
            
            # 恢复映射
            self.entity_to_id = model_data["entity_to_id"]
            self.relation_to_id = model_data["relation_to_id"]
            self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
            self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
            
            # 重建模型
            num_entities = len(self.entity_to_id)
            num_relations = len(self.relation_to_id)
            self.model = self._create_model(num_entities, num_relations)
            self.model.load_state_dict(model_data["model_state_dict"])
            self.model.to(self.device)
            
            # 恢复训练指标
            metrics_data = model_data.get("training_metrics", [])
            self.training_metrics = [TrainingMetrics(**m) for m in metrics_data]
            
            # 重建索引
            await self._build_faiss_index()
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.training_metrics:
            return {}
        
        latest = self.training_metrics[-1]
        best_performance = max(self.training_metrics, key=lambda m: m.hits_at_1)
        
        return {
            "total_epochs": len(self.training_metrics),
            "final_loss": latest.loss,
            "final_hits_at_1": latest.hits_at_1,
            "final_hits_at_10": latest.hits_at_10,
            "final_mrr": latest.mean_reciprocal_rank,
            "best_hits_at_1": best_performance.hits_at_1,
            "best_epoch": best_performance.epoch,
            "num_entities": len(self.entity_to_id),
            "num_relations": len(self.relation_to_id),
            "model_type": self.config.model_type,
            "embedding_dim": self.config.embedding_dim
        }
