/**
 * 查询分析器可视化组件
 * 展示查询类型识别、关键词提取、过滤条件等技术细节
 */

import React from 'react';
import { Card, Tag, Descriptions, Badge, Empty, Space } from 'antd';
import {
  FileTextOutlined,
  FileImageOutlined,
  TableOutlined,
  MergeCellsOutlined,
  FilterOutlined,
  SearchOutlined
} from '@ant-design/icons';

interface QueryAnalysisData {
  query_type: 'text' | 'visual' | 'document' | 'mixed';
  requires_image_search: boolean;
  requires_table_search: boolean;
  filters: Record<string, any>;
  top_k: number;
  similarity_threshold: number;
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
    switch (analysis.query_type) {
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
    switch (analysis.query_type) {
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
            {analysis.query_type.toUpperCase()}
          </Tag>
        </Descriptions.Item>

        <Descriptions.Item label="搜索需求">
          <Space>
            {analysis.requires_image_search && (
              <Badge status="processing" text="图像搜索" />
            )}
            {analysis.requires_table_search && (
              <Badge status="processing" text="表格搜索" />
            )}
            {!analysis.requires_image_search && !analysis.requires_table_search && (
              <Badge status="default" text="纯文本搜索" />
            )}
          </Space>
        </Descriptions.Item>

        {Object.keys(analysis.filters || {}).length > 0 && (
          <Descriptions.Item label="过滤条件">
            <Space direction="vertical" size="small">
              {analysis.filters.file_types && (
                <div>
                  <FilterOutlined className="mr-1" />
                  文件类型: {analysis.filters.file_types.join(', ')}
                </div>
              )}
              {analysis.filters.exact_match && (
                <div>
                  <FilterOutlined className="mr-1" />
                  精确匹配: "{analysis.filters.exact_match.join('", "')}"
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

        <Descriptions.Item label="检索参数">
          <Space>
            <Tag color="cyan">Top-K: {analysis.top_k}</Tag>
            <Tag color="cyan">相似度阈值: {analysis.similarity_threshold}</Tag>
          </Space>
        </Descriptions.Item>
      </Descriptions>
    </Card>
  );
};

export default QueryAnalyzer;
