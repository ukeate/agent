"""文档关系图谱构建引擎"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """关系类型枚举"""
    REFERENCE = "reference"  # 引用关系
    DEPENDENCY = "dependency"  # 依赖关系
    INHERITANCE = "inheritance"  # 继承关系
    SIMILARITY = "similarity"  # 相似关系
    TOPIC = "topic"  # 主题关系
    VERSION = "version"  # 版本关系
    PARENT_CHILD = "parent_child"  # 父子关系


@dataclass
class DocumentRelationship:
    """文档关系数据类"""
    source_doc_id: str
    target_doc_id: str
    relationship_type: RelationshipType
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class DocumentCluster:
    """文档聚类数据类"""
    cluster_id: str
    documents: List[str]
    centroid: Optional[np.ndarray]
    topic: str
    keywords: List[str]
    metadata: Dict[str, Any]


class DocumentRelationshipAnalyzer:
    """文档关系分析器
    
    自动识别文档间的引用、依赖、继承关系
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 2,
        max_cluster_size: int = 50
    ):
        """初始化关系分析器
        
        Args:
            similarity_threshold: 相似度阈值
            min_cluster_size: 最小聚类大小
            max_cluster_size: 最大聚类大小
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.graph = nx.DiGraph()  # 有向图存储关系
    
    async def analyze_relationships(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """分析文档关系
        
        Args:
            documents: 文档列表
            embeddings: 文档向量嵌入
            
        Returns:
            关系分析结果
        """
        relationships = []
        
        # 1. 检测引用关系
        ref_relationships = await self._detect_references(documents)
        relationships.extend(ref_relationships)
        
        # 2. 检测相似关系
        if embeddings:
            sim_relationships = await self._detect_similarities(
                documents, embeddings
            )
            relationships.extend(sim_relationships)
        
        # 3. 检测依赖关系（代码文件）
        dep_relationships = await self._detect_dependencies(documents)
        relationships.extend(dep_relationships)
        
        # 4. 构建关系图
        self._build_relationship_graph(documents, relationships)
        
        # 5. 文档聚类
        clusters = []
        if embeddings:
            clusters = await self._cluster_documents(documents, embeddings)
        
        # 6. 生成关系摘要
        summary = self._generate_relationship_summary(
            documents, relationships, clusters
        )
        
        return {
            "relationships": relationships,
            "clusters": clusters,
            "graph_metrics": self._calculate_graph_metrics(),
            "summary": summary
        }
    
    async def _detect_references(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[DocumentRelationship]:
        """检测文档间的引用关系
        
        Args:
            documents: 文档列表
            
        Returns:
            引用关系列表
        """
        relationships = []
        
        # 构建文档标题到ID的映射
        title_to_id = {}
        filename_to_id = {}
        
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id"))
            title = doc.get("title", "")
            filename = doc.get("file_name", doc.get("source", {}).get("original_path", ""))
            
            if title:
                title_to_id[title.lower()] = doc_id
            if filename:
                # 提取文件名（不含路径）
                import os
                base_name = os.path.basename(filename)
                filename_to_id[base_name.lower()] = doc_id
        
        # 检测每个文档中的引用
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id"))
            content = doc.get("content", "")
            
            if not content:
                continue
            
            # 查找引用模式
            references = self._extract_references(content)
            
            for ref in references:
                ref_lower = ref.lower()
                target_id = None
                
                # 匹配标题
                if ref_lower in title_to_id:
                    target_id = title_to_id[ref_lower]
                # 匹配文件名
                elif ref_lower in filename_to_id:
                    target_id = filename_to_id[ref_lower]
                # 部分匹配
                else:
                    for title, tid in title_to_id.items():
                        if ref_lower in title or title in ref_lower:
                            target_id = tid
                            break
                
                if target_id and target_id != doc_id:
                    relationships.append(DocumentRelationship(
                        source_doc_id=doc_id,
                        target_doc_id=target_id,
                        relationship_type=RelationshipType.REFERENCE,
                        confidence=0.8,
                        metadata={"reference_text": ref}
                    ))
        
        return relationships
    
    async def _detect_similarities(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[np.ndarray]
    ) -> List[DocumentRelationship]:
        """检测文档间的相似关系
        
        Args:
            documents: 文档列表
            embeddings: 文档向量
            
        Returns:
            相似关系列表
        """
        relationships = []
        
        if len(embeddings) < 2:
            return relationships
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 提取高相似度对
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.similarity_threshold:
                    doc_i_id = documents[i].get("doc_id", documents[i].get("id"))
                    doc_j_id = documents[j].get("doc_id", documents[j].get("id"))
                    
                    relationships.append(DocumentRelationship(
                        source_doc_id=doc_i_id,
                        target_doc_id=doc_j_id,
                        relationship_type=RelationshipType.SIMILARITY,
                        confidence=float(similarity),
                        metadata={
                            "similarity_score": float(similarity),
                            "bidirectional": True
                        }
                    ))
        
        return relationships
    
    async def _detect_dependencies(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[DocumentRelationship]:
        """检测文档间的依赖关系（主要针对代码）
        
        Args:
            documents: 文档列表
            
        Returns:
            依赖关系列表
        """
        relationships = []
        
        # 构建模块名到文档ID的映射
        module_to_id = {}
        
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id"))
            file_type = doc.get("file_type", "")
            
            if file_type in ["code", "python", "javascript"]:
                # 从文件路径提取模块名
                source = doc.get("source", {})
                file_path = source.get("original_path", "")
                
                if file_path:
                    import os
                    # 移除扩展名
                    module_name = os.path.splitext(os.path.basename(file_path))[0]
                    module_to_id[module_name] = doc_id
                    
                    # 也存储完整路径（相对导入）
                    module_path = file_path.replace("/", ".").replace("\\", ".")
                    module_to_id[module_path] = doc_id
        
        # 检测导入语句
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id"))
            content = doc.get("content", "")
            file_type = doc.get("file_type", "")
            
            if file_type not in ["code", "python", "javascript"]:
                continue
            
            # 提取导入语句
            imports = self._extract_imports(content, file_type)
            
            for imp in imports:
                # 尝试匹配模块
                for module_name, target_id in module_to_id.items():
                    if imp in module_name or module_name in imp:
                        if target_id != doc_id:
                            relationships.append(DocumentRelationship(
                                source_doc_id=doc_id,
                                target_doc_id=target_id,
                                relationship_type=RelationshipType.DEPENDENCY,
                                confidence=0.9,
                                metadata={
                                    "import_statement": imp,
                                    "dependency_type": "import"
                                }
                            ))
                            break
        
        # 检测继承关系（针对类）
        inheritance_rels = await self._detect_inheritance(documents)
        relationships.extend(inheritance_rels)
        
        return relationships
    
    async def _detect_inheritance(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[DocumentRelationship]:
        """检测继承关系
        
        Args:
            documents: 文档列表
            
        Returns:
            继承关系列表
        """
        relationships = []
        
        # 构建类名到文档ID的映射
        class_to_doc = defaultdict(list)
        
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id"))
            content = doc.get("content", "")
            file_type = doc.get("file_type", "")
            
            if file_type in ["code", "python"]:
                # 提取类定义
                classes = self._extract_classes(content)
                for cls in classes:
                    class_to_doc[cls["name"]].append({
                        "doc_id": doc_id,
                        "class_info": cls
                    })
        
        # 检测继承关系
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id"))
            content = doc.get("content", "")
            file_type = doc.get("file_type", "")
            
            if file_type in ["code", "python"]:
                classes = self._extract_classes(content)
                
                for cls in classes:
                    if cls.get("base_classes"):
                        for base in cls["base_classes"]:
                            # 查找基类所在的文档
                            if base in class_to_doc:
                                for base_doc in class_to_doc[base]:
                                    if base_doc["doc_id"] != doc_id:
                                        relationships.append(DocumentRelationship(
                                            source_doc_id=doc_id,
                                            target_doc_id=base_doc["doc_id"],
                                            relationship_type=RelationshipType.INHERITANCE,
                                            confidence=0.95,
                                            metadata={
                                                "child_class": cls["name"],
                                                "parent_class": base
                                            }
                                        ))
        
        return relationships
    
    async def _cluster_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[np.ndarray]
    ) -> List[DocumentCluster]:
        """对文档进行聚类
        
        Args:
            documents: 文档列表
            embeddings: 文档向量
            
        Returns:
            聚类列表
        """
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        clusters = []
        
        if len(documents) < self.min_cluster_size:
            return clusters
        
        # 标准化向量
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # 确定聚类数量（使用肘部法则的简化版）
        n_clusters = min(
            int(np.sqrt(len(documents) / 2)),
            len(documents) // self.min_cluster_size,
            10  # 最多10个聚类
        )
        
        if n_clusters < 2:
            return clusters
        
        # K-means聚类
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_embeddings)
            
            # 组织聚类结果
            cluster_docs = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                doc_id = documents[idx].get("doc_id", documents[idx].get("id"))
                cluster_docs[label].append(doc_id)
            
            # 创建聚类对象
            for label, doc_ids in cluster_docs.items():
                if len(doc_ids) >= self.min_cluster_size:
                    # 计算聚类中心
                    cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
                    cluster_embeddings = [embeddings[i] for i in cluster_indices]
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # 提取主题和关键词
                    topic, keywords = await self._extract_cluster_topic(
                        [documents[i] for i in cluster_indices]
                    )
                    
                    clusters.append(DocumentCluster(
                        cluster_id=f"cluster_{label}",
                        documents=doc_ids,
                        centroid=centroid,
                        topic=topic,
                        keywords=keywords,
                        metadata={
                            "size": len(doc_ids),
                            "method": "kmeans"
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
        
        return clusters
    
    async def _extract_cluster_topic(
        self,
        documents: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """提取聚类主题和关键词
        
        Args:
            documents: 聚类中的文档
            
        Returns:
            主题和关键词列表
        """
        from collections import Counter
        import re
        
        # 合并文档内容
        combined_text = " ".join([
            doc.get("content", "")[:1000]  # 每个文档取前1000字符
            for doc in documents
        ])
        
        # 简单的关键词提取
        words = re.findall(r'\b[a-z]+\b', combined_text.lower())
        
        # 过滤停用词
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "up", "about",
            "into", "through", "during", "before", "after", "is",
            "are", "was", "were", "been", "be", "have", "has", "had"
        }
        
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # 获取最常见的词
        word_counts = Counter(words)
        keywords = [word for word, _ in word_counts.most_common(10)]
        
        # 生成主题（使用最常见的几个词）
        topic = " ".join(keywords[:3]) if keywords else "unknown"
        
        return topic, keywords
    
    def _extract_references(self, content: str) -> List[str]:
        """提取文档中的引用
        
        Args:
            content: 文档内容
            
        Returns:
            引用列表
        """
        references = []
        
        # 各种引用模式
        patterns = [
            r'\[([^\]]+)\]\([^\)]+\)',  # Markdown链接
            r'(?:see|refer to|reference|引用|参考)\s+([A-Za-z0-9_\-\.]+)',  # 引用关键词
            r'import\s+([A-Za-z0-9_\.]+)',  # 导入语句
            r'from\s+([A-Za-z0-9_\.]+)\s+import',  # Python导入
            r'require\([\'"]([^\'"]+)[\'"]\)',  # JavaScript require
            r'#include\s+[<"]([^>"]+)[>"]',  # C/C++ include
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)
        
        # 去重
        return list(set(references))
    
    def _extract_imports(self, content: str, file_type: str) -> List[str]:
        """提取导入语句
        
        Args:
            content: 代码内容
            file_type: 文件类型
            
        Returns:
            导入列表
        """
        imports = []
        
        if file_type == "python":
            # Python导入
            patterns = [
                r'import\s+([A-Za-z0-9_\.]+)',
                r'from\s+([A-Za-z0-9_\.]+)\s+import',
            ]
        elif file_type in ["javascript", "typescript"]:
            # JavaScript/TypeScript导入
            patterns = [
                r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'require\([\'"]([^\'"]+)[\'"]\)',
                r'import\([\'"]([^\'"]+)[\'"]\)',
            ]
        else:
            patterns = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)
        
        return list(set(imports))
    
    def _extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """提取类定义
        
        Args:
            content: 代码内容
            
        Returns:
            类信息列表
        """
        classes = []
        
        # Python类定义
        class_pattern = r'class\s+(\w+)(?:\(([^)]*)\))?:'
        matches = re.findall(class_pattern, content)
        
        for match in matches:
            class_name = match[0]
            base_classes = []
            
            if match[1]:
                # 解析基类
                bases = match[1].split(',')
                base_classes = [b.strip() for b in bases if b.strip()]
            
            classes.append({
                "name": class_name,
                "base_classes": base_classes
            })
        
        return classes
    
    def _build_relationship_graph(
        self,
        documents: List[Dict[str, Any]],
        relationships: List[DocumentRelationship]
    ):
        """构建关系图
        
        Args:
            documents: 文档列表
            relationships: 关系列表
        """
        # 添加节点
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("id"))
            self.graph.add_node(
                doc_id,
                title=doc.get("title", ""),
                file_type=doc.get("file_type", ""),
                metadata=doc.get("metadata", {})
            )
        
        # 添加边
        for rel in relationships:
            self.graph.add_edge(
                rel.source_doc_id,
                rel.target_doc_id,
                relationship_type=rel.relationship_type.value,
                confidence=rel.confidence,
                metadata=rel.metadata
            )
    
    def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """计算图指标
        
        Returns:
            图指标字典
        """
        if not self.graph:
            return {}
        
        metrics = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
        }
        
        # 计算中心性
        if self.graph.number_of_nodes() > 0:
            try:
                metrics["degree_centrality"] = nx.degree_centrality(self.graph)
                metrics["betweenness_centrality"] = nx.betweenness_centrality(self.graph)
                
                # 找出最重要的节点
                degree_sorted = sorted(
                    metrics["degree_centrality"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                metrics["most_connected"] = degree_sorted[:5] if degree_sorted else []
            except:
                pass
        
        return metrics
    
    def _generate_relationship_summary(
        self,
        documents: List[Dict[str, Any]],
        relationships: List[DocumentRelationship],
        clusters: List[DocumentCluster]
    ) -> Dict[str, Any]:
        """生成关系摘要
        
        Args:
            documents: 文档列表
            relationships: 关系列表
            clusters: 聚类列表
            
        Returns:
            摘要字典
        """
        # 统计各类关系
        rel_counts = defaultdict(int)
        for rel in relationships:
            rel_counts[rel.relationship_type.value] += 1
        
        # 统计文档类型
        type_counts = defaultdict(int)
        for doc in documents:
            type_counts[doc.get("file_type", "unknown")] += 1
        
        summary = {
            "total_documents": len(documents),
            "total_relationships": len(relationships),
            "relationship_types": dict(rel_counts),
            "document_types": dict(type_counts),
            "num_clusters": len(clusters),
            "avg_cluster_size": np.mean([len(c.documents) for c in clusters]) if clusters else 0,
            "graph_connected": nx.is_weakly_connected(self.graph) if self.graph else False,
        }
        
        return summary
    
    def visualize_graph(self, output_path: Optional[str] = None) -> str:
        """可视化关系图
        
        Args:
            output_path: 输出路径
            
        Returns:
            图表数据或文件路径
        """
        import json
        
        # 转换为可视化格式（D3.js兼容）
        nodes = []
        links = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            nodes.append({
                "id": node_id,
                "label": node_data.get("title", node_id)[:50],
                "type": node_data.get("file_type", "unknown"),
                "metadata": node_data.get("metadata", {})
            })
        
        for source, target in self.graph.edges():
            edge_data = self.graph.edges[source, target]
            links.append({
                "source": source,
                "target": target,
                "type": edge_data.get("relationship_type", "unknown"),
                "confidence": edge_data.get("confidence", 0.5),
                "metadata": edge_data.get("metadata", {})
            })
        
        graph_data = {
            "nodes": nodes,
            "links": links
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            return output_path
        else:
            return json.dumps(graph_data)