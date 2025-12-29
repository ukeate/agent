"""智能标签与分类系统"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from collections import Counter, defaultdict
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = get_logger(__name__)

# 尝试下载NLTK数据
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except:
    logger.warning("NLTK data not available")
    NLTK_AVAILABLE = False

class TagCategory(Enum):
    """标签类别枚举"""
    TECHNOLOGY = "technology"  # 技术栈
    TOPIC = "topic"  # 主题
    DIFFICULTY = "difficulty"  # 难度
    TYPE = "type"  # 文档类型
    LANGUAGE = "language"  # 编程语言
    FRAMEWORK = "framework"  # 框架
    DOMAIN = "domain"  # 领域
    CUSTOM = "custom"  # 自定义

@dataclass
class DocumentTag:
    """文档标签数据类"""
    tag: str
    category: TagCategory
    confidence: float
    source: str  # 标签来源：auto, manual, rule
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DocumentClassification:
    """文档分类数据类"""
    category: str
    level: int  # 分类层级
    parent_category: Optional[str] = None
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class AutoTagger:
    """自动标签生成器
    
    基于关键词提取和主题建模自动生成标签
    """
    
    # 预定义的技术栈关键词
    TECH_KEYWORDS = {
        "python": ["python", "py", "pip", "django", "flask", "fastapi"],
        "javascript": ["javascript", "js", "node", "npm", "react", "vue", "angular"],
        "java": ["java", "spring", "maven", "gradle", "hibernate"],
        "database": ["sql", "mysql", "postgresql", "mongodb", "redis", "database"],
        "ai": ["ai", "machine learning", "ml", "deep learning", "neural", "tensorflow", "pytorch"],
        "web": ["html", "css", "web", "frontend", "backend", "api", "rest"],
        "devops": ["docker", "kubernetes", "k8s", "ci/cd", "jenkins", "aws", "cloud"],
        "mobile": ["android", "ios", "react native", "flutter", "mobile"],
    }
    
    # 难度级别关键词
    DIFFICULTY_KEYWORDS = {
        "beginner": ["basic", "introduction", "tutorial", "getting started", "simple"],
        "intermediate": ["intermediate", "practical", "implementation", "develop"],
        "advanced": ["advanced", "expert", "optimization", "architecture", "complex"],
    }
    
    # 文档类型关键词
    DOC_TYPE_KEYWORDS = {
        "tutorial": ["tutorial", "guide", "how to", "step by step"],
        "reference": ["reference", "api", "documentation", "manual"],
        "example": ["example", "demo", "sample", "showcase"],
        "article": ["article", "blog", "post", "analysis"],
        "specification": ["specification", "spec", "requirement", "design"],
    }
    
    def __init__(
        self,
        min_tag_confidence: float = 0.5,
        max_tags_per_doc: int = 20,
        enable_topic_modeling: bool = True
    ):
        """初始化自动标签器
        
        Args:
            min_tag_confidence: 最小标签置信度
            max_tags_per_doc: 每个文档最大标签数
            enable_topic_modeling: 是否启用主题建模
        """
        self.min_tag_confidence = min_tag_confidence
        self.max_tags_per_doc = max_tags_per_doc
        self.enable_topic_modeling = enable_topic_modeling
        
        # 初始化NLP工具
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stop_words = set()
            self.lemmatizer = None
        
        # 自定义规则引擎
        self.custom_rules = []
    
    async def generate_tags(
        self,
        document: Dict[str, Any],
        existing_tags: Optional[List[str]] = None
    ) -> List[DocumentTag]:
        """为文档生成标签
        
        Args:
            document: 文档数据
            existing_tags: 已有标签
            
        Returns:
            标签列表
        """
        tags = []
        existing_tags = existing_tags or []
        
        # 提取文档内容和元数据
        content = document.get("content", "")
        title = document.get("title", "")
        file_type = document.get("file_type", "")
        metadata = document.get("metadata", {})
        
        # 合并文本用于分析
        full_text = f"{title} {content[:5000]}"  # 限制内容长度
        
        # 1. 基于关键词提取
        keyword_tags = await self._extract_keyword_tags(full_text)
        tags.extend(keyword_tags)
        
        # 2. 基于技术栈检测
        tech_tags = await self._detect_technology_tags(full_text, file_type)
        tags.extend(tech_tags)
        
        # 3. 基于难度分析
        difficulty_tag = await self._analyze_difficulty(full_text)
        if difficulty_tag:
            tags.append(difficulty_tag)
        
        # 4. 基于文档类型
        type_tags = await self._detect_document_type(full_text, file_type)
        tags.extend(type_tags)
        
        # 5. 基于主题建模
        if self.enable_topic_modeling and content:
            topic_tags = await self._extract_topics(content)
            tags.extend(topic_tags)
        
        # 6. 应用自定义规则
        custom_tags = await self._apply_custom_rules(document)
        tags.extend(custom_tags)
        
        # 7. 合并已有标签
        for tag in existing_tags:
            tags.append(DocumentTag(
                tag=tag,
                category=TagCategory.CUSTOM,
                confidence=1.0,
                source="manual"
            ))
        
        # 去重和排序
        tags = self._deduplicate_and_rank_tags(tags)
        
        # 限制标签数量
        return tags[:self.max_tags_per_doc]
    
    async def classify_document(
        self,
        document: Dict[str, Any]
    ) -> List[DocumentClassification]:
        """对文档进行分类
        
        Args:
            document: 文档数据
            
        Returns:
            分类列表
        """
        classifications = []
        
        content = document.get("content", "")
        title = document.get("title", "")
        file_type = document.get("file_type", "")
        
        # 一级分类：项目模块
        if "/" in document.get("source", {}).get("original_path", ""):
            path_parts = document["source"]["original_path"].split("/")
            
            # 提取模块名
            for i, part in enumerate(path_parts):
                if part in ["src", "apps", "docs", "tests"]:
                    if i + 1 < len(path_parts):
                        module = path_parts[i + 1]
                        classifications.append(DocumentClassification(
                            category=f"module:{module}",
                            level=1,
                            confidence=0.9
                        ))
                        break
        
        # 二级分类：功能类别
        function_category = await self._detect_function_category(content, file_type)
        if function_category:
            classifications.append(function_category)
        
        # 三级分类：具体功能
        specific_function = await self._detect_specific_function(content, title)
        if specific_function:
            classifications.append(specific_function)
        
        return classifications
    
    async def _extract_keyword_tags(self, text: str) -> List[DocumentTag]:
        """提取关键词标签
        
        Args:
            text: 文本内容
            
        Returns:
            标签列表
        """
        tags = []
        
        # 使用TF-IDF提取关键词
        try:
            # 预处理文本
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return tags
            
            # TF-IDF向量化
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english' if not NLTK_AVAILABLE else None,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # 提取高分关键词
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            for keyword, score in keyword_scores[:10]:
                if score > 0.1:
                    tags.append(DocumentTag(
                        tag=keyword,
                        category=TagCategory.TOPIC,
                        confidence=float(score),
                        source="auto"
                    ))
        
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
        
        return tags
    
    async def _detect_technology_tags(
        self,
        text: str,
        file_type: str
    ) -> List[DocumentTag]:
        """检测技术栈标签
        
        Args:
            text: 文本内容
            file_type: 文件类型
            
        Returns:
            标签列表
        """
        tags = []
        text_lower = text.lower()
        
        # 基于文件类型添加语言标签
        language_map = {
            "python": "python",
            "javascript": "javascript",
            "java": "java",
            "code": "programming",
        }
        
        if file_type in language_map:
            tags.append(DocumentTag(
                tag=language_map[file_type],
                category=TagCategory.LANGUAGE,
                confidence=0.9,
                source="auto"
            ))
        
        # 检测技术关键词
        for tech, keywords in self.TECH_KEYWORDS.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                confidence = min(score / len(keywords), 1.0)
                if confidence >= self.min_tag_confidence:
                    tags.append(DocumentTag(
                        tag=tech,
                        category=TagCategory.TECHNOLOGY,
                        confidence=confidence,
                        source="auto",
                        metadata={"matched_keywords": matched_keywords}
                    ))
        
        # 检测框架
        frameworks = {
            "react": ["react", "jsx", "usestate", "useeffect"],
            "django": ["django", "models.py", "views.py", "urls.py"],
            "fastapi": ["fastapi", "pydantic", "uvicorn"],
            "tensorflow": ["tensorflow", "tf.", "keras"],
            "pytorch": ["pytorch", "torch", "cuda"],
        }
        
        for framework, keywords in frameworks.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(DocumentTag(
                    tag=framework,
                    category=TagCategory.FRAMEWORK,
                    confidence=0.7,
                    source="auto"
                ))
        
        return tags
    
    async def _analyze_difficulty(self, text: str) -> Optional[DocumentTag]:
        """分析文档难度
        
        Args:
            text: 文本内容
            
        Returns:
            难度标签或None
        """
        text_lower = text.lower()
        
        # 计算各难度级别的得分
        difficulty_scores = {}
        
        for level, keywords in self.DIFFICULTY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                difficulty_scores[level] = score
        
        # 选择得分最高的难度级别
        if difficulty_scores:
            best_level = max(difficulty_scores, key=difficulty_scores.get)
            confidence = min(difficulty_scores[best_level] / 3, 1.0)
            
            return DocumentTag(
                tag=best_level,
                category=TagCategory.DIFFICULTY,
                confidence=confidence,
                source="auto"
            )
        
        # 默认为intermediate
        return DocumentTag(
            tag="intermediate",
            category=TagCategory.DIFFICULTY,
            confidence=0.5,
            source="auto"
        )
    
    async def _detect_document_type(
        self,
        text: str,
        file_type: str
    ) -> List[DocumentTag]:
        """检测文档类型
        
        Args:
            text: 文本内容
            file_type: 文件类型
            
        Returns:
            标签列表
        """
        tags = []
        text_lower = text.lower()
        
        # 基于文件类型
        if file_type:
            tags.append(DocumentTag(
                tag=file_type,
                category=TagCategory.TYPE,
                confidence=0.9,
                source="auto"
            ))
        
        # 基于内容特征
        for doc_type, keywords in self.DOC_TYPE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(DocumentTag(
                    tag=doc_type,
                    category=TagCategory.TYPE,
                    confidence=0.7,
                    source="auto"
                ))
                break
        
        return tags
    
    async def _extract_topics(self, text: str) -> List[DocumentTag]:
        """使用主题建模提取主题
        
        Args:
            text: 文本内容
            
        Returns:
            标签列表
        """
        tags = []
        
        try:
            # 预处理文本
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return tags
            
            # 简单的主题提取（实际应使用LDA或其他主题模型）
            # 这里使用词频统计作为简化版本
            words = processed_text.split()
            word_freq = Counter(words)
            
            # 过滤停用词和短词
            filtered_words = {
                word: count for word, count in word_freq.items()
                if len(word) > 3 and word not in self.stop_words
            }
            
            # 提取高频词作为主题
            for word, count in sorted(
                filtered_words.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                if count > 2:
                    tags.append(DocumentTag(
                        tag=word,
                        category=TagCategory.TOPIC,
                        confidence=min(count / 10, 1.0),
                        source="auto"
                    ))
        
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
        
        return tags
    
    async def _apply_custom_rules(
        self,
        document: Dict[str, Any]
    ) -> List[DocumentTag]:
        """应用自定义规则
        
        Args:
            document: 文档数据
            
        Returns:
            标签列表
        """
        tags = []
        
        # 应用每个自定义规则
        for rule in self.custom_rules:
            try:
                if rule["condition"](document):
                    tags.append(DocumentTag(
                        tag=rule["tag"],
                        category=TagCategory.CUSTOM,
                        confidence=rule.get("confidence", 0.8),
                        source="rule"
                    ))
            except Exception as e:
                logger.warning(f"Custom rule failed: {e}")
        
        return tags
    
    async def _detect_function_category(
        self,
        content: str,
        file_type: str
    ) -> Optional[DocumentClassification]:
        """检测功能类别
        
        Args:
            content: 文档内容
            file_type: 文件类型
            
        Returns:
            分类或None
        """
        content_lower = content.lower()
        
        # 功能类别关键词
        categories = {
            "authentication": ["login", "auth", "password", "token", "jwt"],
            "database": ["database", "sql", "query", "table", "migration"],
            "api": ["api", "endpoint", "rest", "graphql", "request"],
            "ui": ["component", "render", "view", "template", "style"],
            "testing": ["test", "assert", "mock", "fixture", "coverage"],
            "configuration": ["config", "setting", "environment", "parameter"],
        }
        
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score >= 2:
                return DocumentClassification(
                    category=f"function:{category}",
                    level=2,
                    confidence=min(score / len(keywords), 1.0)
                )
        
        return None
    
    async def _detect_specific_function(
        self,
        content: str,
        title: str
    ) -> Optional[DocumentClassification]:
        """检测具体功能
        
        Args:
            content: 文档内容
            title: 文档标题
            
        Returns:
            分类或None
        """
        # 基于标题和内容推断具体功能
        title_lower = title.lower()
        
        # 具体功能模式
        patterns = {
            "user_management": ["user", "profile", "account"],
            "payment_processing": ["payment", "checkout", "transaction"],
            "data_analytics": ["analytics", "metrics", "dashboard"],
            "file_handling": ["file", "upload", "download", "storage"],
            "messaging": ["message", "chat", "notification", "email"],
        }
        
        for function, keywords in patterns.items():
            if any(kw in title_lower for kw in keywords):
                return DocumentClassification(
                    category=f"specific:{function}",
                    level=3,
                    confidence=0.7
                )
        
        return None
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 词形还原
        if self.lemmatizer and NLTK_AVAILABLE:
            words = text.split()
            words = [self.lemmatizer.lemmatize(w) for w in words]
            text = ' '.join(words)
        
        return text.strip()
    
    def _deduplicate_and_rank_tags(
        self,
        tags: List[DocumentTag]
    ) -> List[DocumentTag]:
        """去重和排序标签
        
        Args:
            tags: 原始标签列表
            
        Returns:
            处理后的标签列表
        """
        # 使用字典去重，保留最高置信度
        unique_tags = {}
        
        for tag in tags:
            key = (tag.tag.lower(), tag.category)
            if key not in unique_tags or tag.confidence > unique_tags[key].confidence:
                unique_tags[key] = tag
        
        # 按置信度排序
        sorted_tags = sorted(
            unique_tags.values(),
            key=lambda x: (x.confidence, x.category.value),
            reverse=True
        )
        
        # 过滤低置信度标签
        return [
            tag for tag in sorted_tags
            if tag.confidence >= self.min_tag_confidence
        ]
    
    def add_custom_rule(
        self,
        condition: callable,
        tag: str,
        confidence: float = 0.8
    ):
        """添加自定义标签规则
        
        Args:
            condition: 条件函数
            tag: 标签名
            confidence: 置信度
        """
        self.custom_rules.append({
            "condition": condition,
            "tag": tag,
            "confidence": confidence
        })
from src.core.logging import get_logger
