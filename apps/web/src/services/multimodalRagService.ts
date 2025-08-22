/**
 * 多模态RAG服务层
 * 与后端API交互，展示技术实现细节
 */

import { apiClient } from './apiClient';

// 查询分析结果
interface QueryAnalysisResult {
  queryType: 'text' | 'visual' | 'document' | 'mixed';
  confidence: number;
  requiresImageSearch: boolean;
  requiresTableSearch: boolean;
  keywords: string[];
  filters: {
    fileTypes?: string[];
    dates?: string[];
    exactMatch?: string[];
  };
  complexity: number;
  topK?: number;
  similarityThreshold?: number;
}

// 检索策略
interface RetrievalStrategyResult {
  strategy: 'text' | 'visual' | 'document' | 'hybrid';
  weights: {
    text: number;
    image: number;
    table: number;
  };
  reasoning: string[];
  algorithms: string[];
  reranking: boolean;
  diversityFactor: number;
  topK: number;
  similarityThreshold: number;
}

// 检索结果
interface RetrievalResultsData {
  texts: Array<{
    content: string;
    score: number;
    source: string;
    metadata: Record<string, any>;
  }>;
  images: Array<{
    content: string;
    score: number;
    source: string;
    metadata: Record<string, any>;
  }>;
  tables: Array<{
    content: string;
    score: number;
    source: string;
    metadata: Record<string, any>;
  }>;
  sources: string[];
  totalResults: number;
  retrievalTimeMs: number;
}

// QA响应
interface QAResponseData {
  answer: string;
  confidence: number;
  processingTime: number;
  tokensUsed: number;
  modelUsed: string;
  contextLength: number;
}

// 完整查询响应
interface MultimodalQueryResponse {
  queryAnalysis: QueryAnalysisResult;
  retrievalStrategy: RetrievalStrategyResult;
  retrievalResults: RetrievalResultsData;
  qaResponse: QAResponseData;
}

// 系统状态
interface SystemStatus {
  total_documents: number;
  text_documents: number;
  image_documents: number;
  table_documents: number;
  embedding_dimension: number;
  cache_hit_rate: number;
}

class MultimodalRagService {
  private baseUrl = '/api/v1/multimodal-rag';

  /**
   * 执行多模态查询并获取完整技术细节
   */
  async queryWithDetails(
    query: string, 
    files?: File[]
  ): Promise<MultimodalQueryResponse> {
    // 模拟API调用，返回技术展示数据
    await new Promise(resolve => setTimeout(resolve, 1500));

    // 模拟查询分析
    const queryAnalysis: QueryAnalysisResult = {
      queryType: files && files.some(f => f.type.startsWith('image/')) ? 'mixed' : 'text',
      confidence: 0.92,
      requiresImageSearch: !!files?.some(f => f.type.startsWith('image/')),
      requiresTableSearch: query.toLowerCase().includes('表格') || query.toLowerCase().includes('数据'),
      keywords: this.extractKeywords(query),
      filters: {
        fileTypes: files ? files.map(f => f.type) : undefined
      },
      complexity: 0.65,
      topK: 10,
      similarityThreshold: 0.75
    };

    // 模拟检索策略
    const retrievalStrategy: RetrievalStrategyResult = {
      strategy: queryAnalysis.queryType === 'mixed' ? 'hybrid' : 'text',
      weights: {
        text: queryAnalysis.queryType === 'text' ? 1.0 : 0.6,
        image: queryAnalysis.requiresImageSearch ? 0.3 : 0,
        table: queryAnalysis.requiresTableSearch ? 0.1 : 0
      },
      reasoning: [
        `检测到${queryAnalysis.queryType}类型查询`,
        queryAnalysis.requiresImageSearch ? '需要图像检索支持' : '纯文本检索即可',
        `关键词密度分析: ${queryAnalysis.keywords.length}个关键词`
      ],
      algorithms: [
        'Nomic-Embed-v1.5',
        'Cosine Similarity',
        'MMR Diversity',
        'BM25 Reranking'
      ],
      reranking: true,
      diversityFactor: 0.3,
      topK: queryAnalysis.topK || 10,
      similarityThreshold: queryAnalysis.similarityThreshold || 0.75
    };

    // 模拟检索结果
    const retrievalResults: RetrievalResultsData = {
      texts: [
        {
          content: 'Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。它广泛应用于数据科学、机器学习、Web开发等领域。',
          score: 0.92,
          source: 'python_guide.pdf',
          metadata: { chunk_id: 'chunk_001', page: 1, type: 'text' }
        },
        {
          content: '机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需明确编程。常见的ML框架包括TensorFlow、PyTorch等。',
          score: 0.87,
          source: 'ml_handbook.pdf',
          metadata: { chunk_id: 'chunk_042', page: 15, type: 'text' }
        }
      ],
      images: queryAnalysis.requiresImageSearch ? [
        {
          content: 'Architecture diagram showing ML pipeline components',
          score: 0.85,
          source: 'ml_architecture.png',
          metadata: { type: 'image', description: 'ML Pipeline Architecture' }
        }
      ] : [],
      tables: queryAnalysis.requiresTableSearch ? [
        {
          content: 'Performance metrics comparison table',
          score: 0.78,
          source: 'benchmark_results.xlsx',
          metadata: { type: 'table', rows: 10, cols: 5 }
        }
      ] : [],
      sources: ['python_guide.pdf', 'ml_handbook.pdf'],
      totalResults: 2 + (queryAnalysis.requiresImageSearch ? 1 : 0) + (queryAnalysis.requiresTableSearch ? 1 : 0),
      retrievalTimeMs: 235
    };

    // 模拟QA响应
    const qaResponse: QAResponseData = {
      answer: this.generateAnswer(query, retrievalResults),
      confidence: 0.88,
      processingTime: 1250,
      tokensUsed: 512,
      modelUsed: 'GPT-4o',
      contextLength: 2048
    };

    return {
      queryAnalysis,
      retrievalStrategy,
      retrievalResults,
      qaResponse
    };
  }

  /**
   * 上传并处理文档
   */
  async uploadDocument(file: File): Promise<void> {
    const formData = new FormData();
    formData.append('file', file);

    await apiClient.post(`${this.baseUrl}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  }

  /**
   * 批量处理文档
   */
  async processBatch(files: File[]): Promise<void> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    await apiClient.post(`${this.baseUrl}/batch-process`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  }

  /**
   * 获取系统状态
   */
  async getSystemStatus(): Promise<SystemStatus> {
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      total_documents: 156,
      text_documents: 120,
      image_documents: 25,
      table_documents: 11,
      embedding_dimension: 768,
      cache_hit_rate: 72.5
    };
  }

  /**
   * 清空向量存储
   */
  async clearVectorStore(): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/clear`);
  }

  /**
   * 获取向量存储统计
   */
  async getVectorStoreStats(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/stats`);
    return response.data;
  }

  // 辅助方法

  private extractKeywords(query: string): string[] {
    // 简单的关键词提取
    const stopWords = ['的', '是', '在', '和', '了', '有', '我', '你', '他', '她', '它'];
    const words = query.split(/\s+/).filter(word => 
      word.length > 1 && !stopWords.includes(word)
    );
    return words.slice(0, 5);
  }

  private generateAnswer(query: string, results: RetrievalResultsData): string {
    // 模拟答案生成
    const hasResults = results.totalResults > 0;
    
    if (!hasResults) {
      return '抱歉，我没有找到与您查询相关的信息。请尝试使用不同的关键词或上传相关文档。';
    }

    const primaryResult = results.texts[0] || results.images[0] || results.tables[0];
    
    return `基于检索到的信息，${primaryResult.content} 

这个答案来自 ${primaryResult.source}，相似度得分为 ${(primaryResult.score * 100).toFixed(1)}%。

检索系统共找到 ${results.totalResults} 个相关结果，包括 ${results.texts.length} 个文本段落${results.images.length > 0 ? `、${results.images.length} 张图像` : ''}${results.tables.length > 0 ? `和 ${results.tables.length} 个表格` : ''}。`;
  }
}

export const multimodalRagService = new MultimodalRagService();