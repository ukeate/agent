"""
冷启动策略实现

处理新用户和新物品的推荐策略，解决多臂老虎机算法中的冷启动问题。
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod
from .bandits.base import MultiArmedBandit

class ColdStartStrategy(ABC):
    """冷启动策略抽象基类"""
    
    @abstractmethod
    def handle_new_user(
        self, 
        user_id: str, 
        user_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理新用户的冷启动
        
        Args:
            user_id: 用户ID
            user_features: 用户特征信息
            
        Returns:
            冷启动策略结果
        """
        ...
    
    @abstractmethod
    def handle_new_item(
        self,
        item_id: str,
        item_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理新物品的冷启动
        
        Args:
            item_id: 物品ID
            item_features: 物品特征信息
            
        Returns:
            冷启动策略结果
        """
        ...

class ContentBasedColdStart(ColdStartStrategy):
    """基于内容的冷启动策略"""
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        初始化基于内容的冷启动策略
        
        Args:
            similarity_threshold: 相似性阈值
        """
        self.similarity_threshold = similarity_threshold
        self.user_profiles = {}  # 用户画像
        self.item_profiles = {}  # 物品画像
        self.user_item_history = {}  # 用户-物品交互历史
    
    def handle_new_user(
        self, 
        user_id: str, 
        user_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        基于内容相似性处理新用户
        
        Args:
            user_id: 用户ID
            user_features: 用户特征信息
            
        Returns:
            推荐策略结果
        """
        if user_features is None:
            # 使用随机策略或流行物品策略
            return {
                "strategy": "random",
                "confidence": 0.1,
                "recommendations": self._get_popular_items(),
                "reason": "没有用户特征，使用流行物品策略"
            }
        
        # 存储用户画像
        self.user_profiles[user_id] = user_features
        
        # 查找相似用户
        similar_users = self._find_similar_users(user_features)
        
        if similar_users:
            # 基于相似用户的推荐
            recommendations = self._recommend_from_similar_users(similar_users)
            return {
                "strategy": "similar_users",
                "confidence": 0.7,
                "recommendations": recommendations,
                "similar_users": [u[0] for u in similar_users[:5]],
                "reason": f"基于{len(similar_users)}个相似用户的推荐"
            }
        else:
            # 基于内容特征的推荐
            recommendations = self._recommend_by_content_features(user_features)
            return {
                "strategy": "content_based", 
                "confidence": 0.5,
                "recommendations": recommendations,
                "reason": "基于用户特征的内容推荐"
            }
    
    def handle_new_item(
        self,
        item_id: str,
        item_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        基于内容相似性处理新物品
        
        Args:
            item_id: 物品ID
            item_features: 物品特征信息
            
        Returns:
            物品推荐策略结果
        """
        if item_features is None:
            return {
                "strategy": "explore",
                "confidence": 0.2,
                "target_users": [],
                "reason": "没有物品特征，需要探索"
            }
        
        # 存储物品画像
        self.item_profiles[item_id] = item_features
        
        # 查找相似物品
        similar_items = self._find_similar_items(item_features)
        
        if similar_items:
            # 找到喜欢相似物品的用户
            target_users = self._find_users_liking_similar_items(similar_items)
            return {
                "strategy": "similar_items",
                "confidence": 0.8,
                "target_users": target_users,
                "similar_items": [i[0] for i in similar_items[:5]],
                "reason": f"基于{len(similar_items)}个相似物品推荐给目标用户"
            }
        else:
            return {
                "strategy": "explore",
                "confidence": 0.3,
                "target_users": list(self.user_profiles.keys()),
                "reason": "没有找到相似物品，向所有用户探索"
            }
    
    def _find_similar_users(
        self, 
        user_features: Dict[str, Any], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        查找相似用户
        
        Args:
            user_features: 用户特征
            top_k: 返回相似用户数量
            
        Returns:
            相似用户列表，每个元素为(用户ID, 相似度)
        """
        similarities = []
        user_vector = self._features_to_vector(user_features)
        
        for uid, profile in self.user_profiles.items():
            profile_vector = self._features_to_vector(profile)
            similarity = self._calculate_cosine_similarity(user_vector, profile_vector)
            
            if similarity > self.similarity_threshold:
                similarities.append((uid, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _find_similar_items(
        self,
        item_features: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        查找相似物品
        
        Args:
            item_features: 物品特征
            top_k: 返回相似物品数量
            
        Returns:
            相似物品列表，每个元素为(物品ID, 相似度)
        """
        similarities = []
        item_vector = self._features_to_vector(item_features)
        
        for iid, profile in self.item_profiles.items():
            profile_vector = self._features_to_vector(profile)
            similarity = self._calculate_cosine_similarity(item_vector, profile_vector)
            
            if similarity > self.similarity_threshold:
                similarities.append((iid, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        将特征字典转换为向量
        
        Args:
            features: 特征字典
            
        Returns:
            特征向量
        """
        # 简单实现：提取数值型特征
        vector = []
        
        for key, value in sorted(features.items()):
            if isinstance(value, (int, float)):
                vector.append(value)
            elif isinstance(value, str):
                # 字符串特征用哈希值表示
                vector.append(hash(value) % 1000 / 1000.0)
            elif isinstance(value, list):
                # 列表特征用平均值表示
                if value and all(isinstance(v, (int, float)) for v in value):
                    vector.append(np.mean(value))
                else:
                    vector.append(0.0)
            else:
                vector.append(0.0)
        
        return np.array(vector) if vector else np.array([0.0])
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度
        """
        # 确保向量长度一致
        min_len = min(len(vec1), len(vec2))
        if min_len == 0:
            return 0.0
            
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _recommend_from_similar_users(
        self, 
        similar_users: List[Tuple[str, float]]
    ) -> List[str]:
        """
        基于相似用户推荐物品
        
        Args:
            similar_users: 相似用户列表
            
        Returns:
            推荐物品列表
        """
        item_scores = {}
        
        for user_id, similarity in similar_users:
            if user_id in self.user_item_history:
                for item_id, rating in self.user_item_history[user_id].items():
                    if item_id not in item_scores:
                        item_scores[item_id] = 0.0
                    item_scores[item_id] += similarity * rating
        
        # 按分数排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, score in sorted_items[:10]]
    
    def _recommend_by_content_features(self, user_features: Dict[str, Any]) -> List[str]:
        """
        基于用户特征推荐物品
        
        Args:
            user_features: 用户特征
            
        Returns:
            推荐物品列表
        """
        item_scores = {}
        user_vector = self._features_to_vector(user_features)
        
        for item_id, item_features in self.item_profiles.items():
            item_vector = self._features_to_vector(item_features)
            similarity = self._calculate_cosine_similarity(user_vector, item_vector)
            item_scores[item_id] = similarity
        
        # 按相似度排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, score in sorted_items[:10]]
    
    def _find_users_liking_similar_items(
        self, 
        similar_items: List[Tuple[str, float]]
    ) -> List[str]:
        """
        查找喜欢相似物品的用户
        
        Args:
            similar_items: 相似物品列表
            
        Returns:
            目标用户列表
        """
        target_users = set()
        
        for item_id, similarity in similar_items:
            for user_id, history in self.user_item_history.items():
                if item_id in history and history[item_id] > 0.5:  # 喜欢的阈值
                    target_users.add(user_id)
        
        return list(target_users)
    
    def _get_popular_items(self) -> List[str]:
        """
        获取流行物品列表
        
        Returns:
            流行物品列表
        """
        item_popularity = {}
        
        for user_id, history in self.user_item_history.items():
            for item_id, rating in history.items():
                if item_id not in item_popularity:
                    item_popularity[item_id] = []
                item_popularity[item_id].append(rating)
        
        # 计算平均评分和交互次数
        item_scores = {}
        for item_id, ratings in item_popularity.items():
            avg_rating = np.mean(ratings)
            num_interactions = len(ratings)
            # 综合评分和交互次数
            item_scores[item_id] = avg_rating * np.log(1 + num_interactions)
        
        # 按流行度排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, score in sorted_items[:10]]
    
    def update_user_item_interaction(
        self, 
        user_id: str, 
        item_id: str, 
        rating: float
    ):
        """
        更新用户-物品交互记录
        
        Args:
            user_id: 用户ID
            item_id: 物品ID  
            rating: 评分或偏好值
        """
        if user_id not in self.user_item_history:
            self.user_item_history[user_id] = {}
        self.user_item_history[user_id][item_id] = rating
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取冷启动策略统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "num_users": len(self.user_profiles),
            "num_items": len(self.item_profiles),
            "num_interactions": sum(len(hist) for hist in self.user_item_history.values()),
            "similarity_threshold": self.similarity_threshold
        }

class PopularityBasedColdStart(ColdStartStrategy):
    """基于流行度的冷启动策略"""
    
    def __init__(self, decay_factor: float = 0.95):
        """
        初始化基于流行度的冷启动策略
        
        Args:
            decay_factor: 时间衰减因子
        """
        self.decay_factor = decay_factor
        self.item_popularity = {}  # 物品流行度分数
        self.item_categories = {}  # 物品分类
        self.user_preferences = {}  # 用户偏好
        self.time_weights = {}  # 时间权重
    
    def handle_new_user(
        self, 
        user_id: str, 
        user_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        基于流行度处理新用户
        
        Args:
            user_id: 用户ID
            user_features: 用户特征信息
            
        Returns:
            推荐策略结果
        """
        if user_features and 'categories' in user_features:
            # 基于用户感兴趣的分类推荐流行物品
            preferred_categories = user_features['categories']
            recommendations = self._get_popular_items_by_category(preferred_categories)
            
            return {
                "strategy": "category_popularity",
                "confidence": 0.6,
                "recommendations": recommendations,
                "categories": preferred_categories,
                "reason": f"基于{len(preferred_categories)}个偏好分类的流行物品推荐"
            }
        else:
            # 推荐整体最流行的物品
            recommendations = self._get_top_popular_items()
            
            return {
                "strategy": "global_popularity",
                "confidence": 0.4,
                "recommendations": recommendations,
                "reason": "基于全局流行度的物品推荐"
            }
    
    def handle_new_item(
        self,
        item_id: str,
        item_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        基于流行度处理新物品
        
        Args:
            item_id: 物品ID
            item_features: 物品特征信息
            
        Returns:
            物品推荐策略结果
        """
        # 初始化新物品的流行度
        self.item_popularity[item_id] = 0.0
        
        if item_features and 'category' in item_features:
            category = item_features['category']
            self.item_categories[item_id] = category
            
            # 找到对该分类感兴趣的用户
            target_users = self._find_users_interested_in_category(category)
            
            return {
                "strategy": "category_targeting",
                "confidence": 0.7,
                "target_users": target_users,
                "category": category,
                "reason": f"向对{category}分类感兴趣的{len(target_users)}个用户推荐"
            }
        else:
            # 随机推荐给一部分用户
            all_users = list(self.user_preferences.keys())
            sample_size = min(len(all_users), 10)  # 最多10个用户
            target_users = np.random.choice(all_users, sample_size, replace=False).tolist() if all_users else []
            
            return {
                "strategy": "random_sampling",
                "confidence": 0.2,
                "target_users": target_users,
                "reason": f"随机向{len(target_users)}个用户推荐以收集反馈"
            }
    
    def _get_popular_items_by_category(self, categories: List[str]) -> List[str]:
        """
        获取指定分类的流行物品
        
        Args:
            categories: 分类列表
            
        Returns:
            流行物品列表
        """
        category_items = {}
        
        for item_id, item_category in self.item_categories.items():
            if item_category in categories:
                popularity = self.item_popularity.get(item_id, 0.0)
                if item_category not in category_items:
                    category_items[item_category] = []
                category_items[item_category].append((item_id, popularity))
        
        # 从每个分类选择最流行的物品
        recommendations = []
        for category in categories:
            if category in category_items:
                # 按流行度排序
                sorted_items = sorted(category_items[category], key=lambda x: x[1], reverse=True)
                recommendations.extend([item_id for item_id, _ in sorted_items[:3]])  # 每个分类取前3个
        
        return recommendations[:10]  # 最多返回10个
    
    def _get_top_popular_items(self, top_k: int = 10) -> List[str]:
        """
        获取全局最流行的物品
        
        Args:
            top_k: 返回数量
            
        Returns:
            最流行物品列表
        """
        sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:top_k]]
    
    def _find_users_interested_in_category(self, category: str) -> List[str]:
        """
        查找对指定分类感兴趣的用户
        
        Args:
            category: 分类名称
            
        Returns:
            感兴趣的用户列表
        """
        interested_users = []
        
        for user_id, preferences in self.user_preferences.items():
            if 'categories' in preferences and category in preferences['categories']:
                interested_users.append(user_id)
        
        return interested_users
    
    def update_item_popularity(self, item_id: str, interaction_score: float):
        """
        更新物品流行度
        
        Args:
            item_id: 物品ID
            interaction_score: 交互分数
        """
        if item_id not in self.item_popularity:
            self.item_popularity[item_id] = 0.0
        
        # 使用时间衰减更新流行度
        current_popularity = self.item_popularity[item_id]
        self.item_popularity[item_id] = current_popularity * self.decay_factor + interaction_score
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        更新用户偏好
        
        Args:
            user_id: 用户ID
            preferences: 偏好信息
        """
        self.user_preferences[user_id] = preferences
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取流行度策略统计信息
        
        Returns:
            统计信息字典
        """
        categories = list(set(self.item_categories.values())) if self.item_categories else []
        
        return {
            "num_items": len(self.item_popularity),
            "num_users": len(self.user_preferences),
            "num_categories": len(categories),
            "categories": categories,
            "decay_factor": self.decay_factor,
            "top_items": self._get_top_popular_items(5)
        }

class HybridColdStart(ColdStartStrategy):
    """混合冷启动策略"""
    
    def __init__(
        self, 
        content_strategy: ContentBasedColdStart,
        popularity_strategy: PopularityBasedColdStart,
        content_weight: float = 0.6,
        popularity_weight: float = 0.4
    ):
        """
        初始化混合冷启动策略
        
        Args:
            content_strategy: 基于内容的策略
            popularity_strategy: 基于流行度的策略
            content_weight: 内容策略权重
            popularity_weight: 流行度策略权重
        """
        self.content_strategy = content_strategy
        self.popularity_strategy = popularity_strategy
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        
        # 确保权重和为1
        total_weight = content_weight + popularity_weight
        self.content_weight = content_weight / total_weight
        self.popularity_weight = popularity_weight / total_weight
    
    def handle_new_user(
        self, 
        user_id: str, 
        user_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        混合策略处理新用户
        
        Args:
            user_id: 用户ID
            user_features: 用户特征信息
            
        Returns:
            推荐策略结果
        """
        # 获取两个策略的结果
        content_result = self.content_strategy.handle_new_user(user_id, user_features)
        popularity_result = self.popularity_strategy.handle_new_user(user_id, user_features)
        
        # 合并推荐结果
        content_recs = set(content_result["recommendations"])
        popularity_recs = set(popularity_result["recommendations"])
        
        # 计算综合置信度
        combined_confidence = (
            content_result["confidence"] * self.content_weight +
            popularity_result["confidence"] * self.popularity_weight
        )
        
        # 合并推荐，优先内容策略的推荐
        combined_recs = []
        
        # 先加入内容策略的推荐
        for item in content_result["recommendations"]:
            if len(combined_recs) < 10:
                combined_recs.append(item)
        
        # 再加入流行度策略的推荐（避免重复）
        for item in popularity_result["recommendations"]:
            if item not in combined_recs and len(combined_recs) < 10:
                combined_recs.append(item)
        
        return {
            "strategy": "hybrid",
            "confidence": combined_confidence,
            "recommendations": combined_recs,
            "content_result": content_result,
            "popularity_result": popularity_result,
            "weights": {
                "content": self.content_weight,
                "popularity": self.popularity_weight
            },
            "reason": f"混合策略：{self.content_weight:.1%}内容 + {self.popularity_weight:.1%}流行度"
        }
    
    def handle_new_item(
        self,
        item_id: str,
        item_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        混合策略处理新物品
        
        Args:
            item_id: 物品ID
            item_features: 物品特征信息
            
        Returns:
            物品推荐策略结果
        """
        # 获取两个策略的结果
        content_result = self.content_strategy.handle_new_item(item_id, item_features)
        popularity_result = self.popularity_strategy.handle_new_item(item_id, item_features)
        
        # 合并目标用户
        content_users = set(content_result["target_users"])
        popularity_users = set(popularity_result["target_users"])
        combined_users = list(content_users.union(popularity_users))
        
        # 计算综合置信度
        combined_confidence = (
            content_result["confidence"] * self.content_weight +
            popularity_result["confidence"] * self.popularity_weight
        )
        
        return {
            "strategy": "hybrid",
            "confidence": combined_confidence,
            "target_users": combined_users,
            "content_result": content_result,
            "popularity_result": popularity_result,
            "weights": {
                "content": self.content_weight,
                "popularity": self.popularity_weight
            },
            "reason": f"混合策略：{self.content_weight:.1%}内容 + {self.popularity_weight:.1%}流行度"
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取混合策略统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "strategy_type": "hybrid",
            "content_weight": self.content_weight,
            "popularity_weight": self.popularity_weight,
            "content_stats": self.content_strategy.get_statistics(),
            "popularity_stats": self.popularity_strategy.get_statistics()
        }
