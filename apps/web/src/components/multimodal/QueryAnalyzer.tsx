/**
 * 查询分析器可视化组件
 * 展示查询类型识别、关键词提取、过滤条件等技术细节
 */

import React from 'react';
import { Card, Tag, Descriptions, Progress, Badge, Empty, Space } from 'antd';
import {
  FileTextOutlined,
  FileImageOutlined,
  TableOutlined,
  MergeCellsOutlined,
  FilterOutlined,
  SearchOutlined
} from '@ant-design/icons';

interface QueryAnalysisData {
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

interface QueryAnalyzerProps {
  analysis: QueryAnalysisData | null;
}

const QueryAnalyzer: React.FC<QueryAnalyzerProps> = ({ analysis }) => {
  if (!analysis) {
    return (
      <Card title="查询分析器" size="small">
        <Empty description="等待查询输入..." />
      </Card>
    );
  }

  const getQueryTypeIcon = () => {
    switch (analysis.queryType) {
      case 'text':
        return <FileTextOutlined />;
      case 'visual':
        return <FileImageOutlined />;
      case 'document':
        return <TableOutlined />;
      case 'mixed':
        return <MergeCellsOutlined />;
      default:
        return <SearchOutlined />;
    }
  };

  const getQueryTypeColor = () => {
    switch (analysis.queryType) {
      case 'text':
        return 'blue';
      case 'visual':
        return 'purple';
      case 'document':
        return 'green';
      case 'mixed':
        return 'orange';
      default:
        return 'default';
    }
  };

  const getComplexityLevel = (complexity: number) => {
    if (complexity < 0.3) return { text: '简单', color: 'green' };
    if (complexity < 0.7) return { text: '中等', color: 'orange' };
    return { text: '复杂', color: 'red' };
  };

  const complexityLevel = getComplexityLevel(analysis.complexity);

  return (
    <Card 
      title={
        <span>
          <SearchOutlined className="mr-2" />
          查询分析结果
        </span>
      }
      size="small"
    >
      <Descriptions column={1} size="small">
        <Descriptions.Item label="查询类型">
          <Tag icon={getQueryTypeIcon()} color={getQueryTypeColor()}>
            {analysis.queryType.toUpperCase()}
          </Tag>
          <Progress
            percent={Math.round(analysis.confidence * 100)}
            size="small"
            style={{ width: 80, marginLeft: 8 }}
            format={percent => `${percent}%`}
          />
        </Descriptions.Item>

        <Descriptions.Item label="搜索需求">
          <Space>
            {analysis.requiresImageSearch && (
              <Badge status="processing" text="图像搜索" />
            )}
            {analysis.requiresTableSearch && (
              <Badge status="processing" text="表格搜索" />
            )}
            {!analysis.requiresImageSearch && !analysis.requiresTableSearch && (
              <Badge status="default" text="纯文本搜索" />
            )}
          </Space>
        </Descriptions.Item>

        <Descriptions.Item label="查询复杂度">
          <Tag color={complexityLevel.color}>
            {complexityLevel.text}
          </Tag>
          <Progress
            percent={Math.round(analysis.complexity * 100)}
            size="small"
            strokeColor={complexityLevel.color}
            style={{ width: 100, marginLeft: 8 }}
          />
        </Descriptions.Item>

        <Descriptions.Item label="关键词">
          {analysis.keywords.length > 0 ? (
            <Space wrap>
              {analysis.keywords.slice(0, 5).map((keyword, idx) => (
                <Tag key={idx} color="blue">
                  {keyword}
                </Tag>
              ))}
              {analysis.keywords.length > 5 && (
                <Tag>+{analysis.keywords.length - 5}</Tag>
              )}
            </Space>
          ) : (
            <span style={{ color: '#999' }}>未提取到关键词</span>
          )}
        </Descriptions.Item>

        {Object.keys(analysis.filters).length > 0 && (
          <Descriptions.Item label="过滤条件">
            <Space direction="vertical" size="small">
              {analysis.filters.fileTypes && (
                <div>
                  <FilterOutlined className="mr-1" />
                  文件类型: {analysis.filters.fileTypes.join(', ')}
                </div>
              )}
              {analysis.filters.exactMatch && (
                <div>
                  <FilterOutlined className="mr-1" />
                  精确匹配: "{analysis.filters.exactMatch.join('", "')}"
                </div>
              )}
              {analysis.filters.dates && (
                <div>
                  <FilterOutlined className="mr-1" />
                  日期范围: {analysis.filters.dates.join(' - ')}
                </div>
              )}
            </Space>
          </Descriptions.Item>
        )}

        {(analysis.topK || analysis.similarityThreshold) && (
          <Descriptions.Item label="检索参数">
            <Space>
              {analysis.topK && (
                <Tag color="cyan">Top-K: {analysis.topK}</Tag>
              )}
              {analysis.similarityThreshold && (
                <Tag color="cyan">
                  相似度阈值: {analysis.similarityThreshold}
                </Tag>
              )}
            </Space>
          </Descriptions.Item>
        )}
      </Descriptions>
    </Card>
  );
};

export default QueryAnalyzer;