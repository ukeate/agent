/**
 * 多模态RAG系统页面
 * 
 * 技术展示重点：
 * 1. 文档处理管道可视化
 * 2. 查询类型分析展示
 * 3. 检索策略决策过程
 * 4. 向量存储状态监控
 * 5. 多模态结果展示
 */

import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Tabs, message, Typography, Space, Statistic } from 'antd';
import {
  FileImageOutlined,
  FileTextOutlined,
  TableOutlined,
  SearchOutlined,
  DatabaseOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import DocumentUploader from '../components/multimodal/DocumentUploader';
import QueryAnalyzer from '../components/multimodal/QueryAnalyzer';
import RetrievalStrategy from '../components/multimodal/RetrievalStrategy';
import VectorStoreStatus from '../components/multimodal/VectorStoreStatus';
import MultimodalResults from '../components/multimodal/MultimodalResults';
import QueryInterface from '../components/multimodal/QueryInterface';
import { multimodalRagService } from '../services/multimodalRagService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface SystemStats {
  totalDocuments: number;
  textChunks: number;
  images: number;
  tables: number;
  embeddingDimension: number;
  cacheHitRate: number;
}

const MultimodalRagPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [systemStats, setSystemStats] = useState<SystemStats>({
    totalDocuments: 0,
    textChunks: 0,
    images: 0,
    tables: 0,
    embeddingDimension: 768,
    cacheHitRate: 0
  });
  
  // 查询分析结果
  const [queryAnalysis, setQueryAnalysis] = useState<any>(null);
  
  // 检索策略信息
  const [retrievalStrategy, setRetrievalStrategy] = useState<any>(null);
  
  // 检索结果
  const [retrievalResults, setRetrievalResults] = useState<any>(null);
  
  // 最终答案
  const [qaResponse, setQaResponse] = useState<any>(null);

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const status = await multimodalRagService.getSystemStatus();
      setSystemStats({
        totalDocuments: status.total_documents,
        textChunks: status.text_documents,
        images: status.image_documents,
        tables: status.table_documents,
        embeddingDimension: status.embedding_dimension,
        cacheHitRate: status.cache_hit_rate || 0
      });
    } catch (error) {
      console.error('Failed to load system status:', error);
    }
  };

  const handleQuery = async (query: string, files?: File[]) => {
    setLoading(true);
    try {
      // 执行查询并获取完整的技术细节
      const response = await multimodalRagService.queryWithDetails(query, files);
      
      // 更新各个技术组件的状态
      setQueryAnalysis(response.queryAnalysis);
      setRetrievalStrategy(response.retrievalStrategy);
      setRetrievalResults(response.retrievalResults);
      setQaResponse(response.qaResponse);
      
      message.success('查询执行成功');
    } catch (error) {
      message.error('查询失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDocumentUploaded = () => {
    loadSystemStatus();
    message.success('文档已成功添加到向量存储');
  };

  return (
    <div className="p-6">
      {/* 页面标题和系统概览 */}
      <Card className="mb-4">
        <Title level={2}>
          <ThunderboltOutlined className="mr-2" />
          多模态RAG系统 - 技术实现展示
        </Title>
        <Text type="secondary">
          展示LangChain多模态RAG的完整技术栈：文档处理、向量存储、智能检索、上下文组装
        </Text>
        
        <Row gutter={16} className="mt-4">
          <Col span={4}>
            <Statistic
              title="总文档数"
              value={systemStats.totalDocuments}
              prefix={<DatabaseOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="文本块"
              value={systemStats.textChunks}
              prefix={<FileTextOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="图像"
              value={systemStats.images}
              prefix={<FileImageOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="表格"
              value={systemStats.tables}
              prefix={<TableOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="嵌入维度"
              value={systemStats.embeddingDimension}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="缓存命中率"
              value={systemStats.cacheHitRate}
              suffix="%"
            />
          </Col>
        </Row>
      </Card>

      {/* 主要功能标签页 */}
      <Tabs defaultActiveKey="query" size="large">
        <TabPane 
          tab={
            <span>
              <SearchOutlined />
              多模态查询
            </span>
          } 
          key="query"
        >
          <Row gutter={16}>
            {/* 左侧：查询输入 */}
            <Col span={8}>
              <QueryInterface 
                onQuery={handleQuery}
                loading={loading}
              />
            </Col>
            
            {/* 中间：技术过程展示 */}
            <Col span={8}>
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                {/* 查询分析展示 */}
                <QueryAnalyzer analysis={queryAnalysis} />
                
                {/* 检索策略展示 */}
                <RetrievalStrategy strategy={retrievalStrategy} />
              </Space>
            </Col>
            
            {/* 右侧：结果展示 */}
            <Col span={8}>
              <MultimodalResults 
                retrievalResults={retrievalResults}
                qaResponse={qaResponse}
              />
            </Col>
          </Row>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <DatabaseOutlined />
              文档管理
            </span>
          } 
          key="documents"
        >
          <Row gutter={16}>
            <Col span={12}>
              <DocumentUploader onUploadSuccess={handleDocumentUploaded} />
            </Col>
            <Col span={12}>
              <VectorStoreStatus stats={systemStats} />
            </Col>
          </Row>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <ThunderboltOutlined />
              系统监控
            </span>
          } 
          key="monitor"
        >
          <Row gutter={16}>
            <Col span={24}>
              <Card title="检索性能监控">
                <Row gutter={16}>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title="平均检索时间"
                        value={0}
                        suffix="ms"
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title="平均相似度分数"
                        value={0}
                        precision={2}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title="查询吞吐量"
                        value={0}
                        suffix="QPS"
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title="内存使用"
                        value={0}
                        suffix="MB"
                      />
                    </Card>
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default MultimodalRagPage;