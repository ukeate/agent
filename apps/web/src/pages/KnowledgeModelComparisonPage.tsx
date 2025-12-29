import React, { useState, useEffect } from 'react';
import { Card, Table, Progress, Typography, Tag, Button, Select, Space, message, Modal, Descriptions, Spin } from 'antd';
import { DownloadOutlined, ReloadOutlined, CompareOutlined, DatabaseOutlined } from '@ant-design/icons';
import { knowledgeModelService, KnowledgeModel, ComparisonResult } from '../services/knowledgeModelService';

import { logger } from '../utils/logger'
const { Title, Paragraph } = Typography;
const { Option } = Select;

const KnowledgeModelComparisonPage: React.FC = () => {
  const [models, setModels] = useState<KnowledgeModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedType, setSelectedType] = useState<string>('');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [showComparisonModal, setShowComparisonModal] = useState(false);

  useEffect(() => {
    loadModels();
  }, [selectedType]);

  const loadModels = async () => {
    setLoading(true);
    try {
      const data = await knowledgeModelService.listModels(selectedType);
      setModels(data);
    } catch (error) {
      logger.error('加载模型失败:', error);
      message.error('加载模型数据失败');
      setModels([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (selectedModels.length !== 2) {
      message.warning('请选择两个模型进行对比');
      return;
    }

    setLoading(true);
    try {
      const result = await knowledgeModelService.compareModels(selectedModels[0], selectedModels[1]);
      setComparisonResult(result);
      setShowComparisonModal(true);
    } catch (error) {
      logger.error('模型对比失败:', error);
      message.error('模型对比失败');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (models.length === 0) {
      message.warning('没有可导出的数据');
      return;
    }

    try {
      const blob = await knowledgeModelService.exportComparison(
        models.map(m => m.key),
        'json'
      );
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model-comparison-${new Date().toISOString()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      message.success('导出成功');
    } catch (error) {
      logger.error('导出失败:', error);
      message.error('导出失败');
    }
  };

  const columns = [
    {
      title: '选择',
      key: 'select',
      width: 60,
      render: (_, record: KnowledgeModel) => (
        <input
          type="checkbox"
          checked={selectedModels.includes(record.key)}
          onChange={(e) => {
            if (e.target.checked) {
              if (selectedModels.length < 2) {
                setSelectedModels([...selectedModels, record.key]);
              } else {
                message.warning('最多选择两个模型进行对比');
              }
            } else {
              setSelectedModels(selectedModels.filter(id => id !== record.key));
            }
          }}
        />
      )
    },
    { 
      title: '模型名称', 
      dataIndex: 'name', 
      key: 'name',
      render: (name: string, record: KnowledgeModel) => (
        <Space direction="vertical" size={0}>
          <span style={{ fontWeight: 500 }}>{name}</span>
          {record.status && (
            <Tag color={record.status === 'active' ? 'green' : 'orange'} style={{ fontSize: 10 }}>
              {record.status === 'active' ? '运行中' : '测试中'}
            </Tag>
          )}
        </Space>
      )
    },
    { 
      title: '类型', 
      dataIndex: 'type', 
      key: 'type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    { 
      title: '准确率', 
      dataIndex: 'accuracy', 
      key: 'accuracy',
      sorter: (a: KnowledgeModel, b: KnowledgeModel) => a.accuracy - b.accuracy,
      render: (accuracy: number) => (
        <Progress 
          percent={accuracy} 
          size="small" 
          format={percent => `${percent}%`}
          strokeColor={accuracy >= 95 ? '#52c41a' : accuracy >= 90 ? '#1890ff' : '#faad14'}
        />
      )
    },
    { 
      title: '速度 (tokens/s)', 
      dataIndex: 'speed', 
      key: 'speed',
      sorter: (a: KnowledgeModel, b: KnowledgeModel) => a.speed - b.speed,
      render: (speed: number) => (
        <span style={{ color: speed >= 1000 ? '#52c41a' : speed >= 500 ? '#1890ff' : '#000' }}>
          {speed.toLocaleString()}
        </span>
      )
    },
    { 
      title: '模型大小', 
      dataIndex: 'size', 
      key: 'size'
    },
    {
      title: '支持语言',
      dataIndex: 'languages',
      key: 'languages',
      render: (languages: string[]) => languages ? (
        <Space size={4}>
          {languages.slice(0, 2).map(lang => (
            <Tag key={lang} color="purple">{lang}</Tag>
          ))}
          {languages.length > 2 && <Tag>+{languages.length - 2}</Tag>}
        </Space>
      ) : '-'
    },
    {
      title: '功能特性',
      dataIndex: 'features',
      key: 'features',
      render: (features: string[]) => features ? (
        <Space size={4}>
          {features.slice(0, 2).map(feature => (
            <Tag key={feature}>{feature}</Tag>
          ))}
          {features.length > 2 && <Tag>+{features.length - 2}</Tag>}
        </Space>
      ) : '-'
    }
  ];

  const modelTypes = ['实体识别', '文本分类', '文本生成', '情感分析', '翻译', '问答系统'];

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: 8 }} />
          模型对比
        </Title>
        <Paragraph type="secondary">
          对比不同知识抽取模型的性能和准确率，选择最适合的模型
        </Paragraph>
      </div>

      <Card 
        title="模型对比表"
        extra={
          <Space>
            <Select
              placeholder="筛选模型类型"
              style={{ width: 150 }}
              allowClear
              value={selectedType}
              onChange={setSelectedType}
            >
              {modelTypes.map(type => (
                <Option key={type} value={type}>{type}</Option>
              ))}
            </Select>
            <Button
              icon={<CompareOutlined />}
              type="primary"
              disabled={selectedModels.length !== 2}
              onClick={handleCompare}
            >
              对比选中模型
            </Button>
            <Button icon={<ReloadOutlined />} onClick={loadModels}>
              刷新
            </Button>
            <Button icon={<DownloadOutlined />} onClick={handleExport}>
              导出
            </Button>
          </Space>
        }
      >
        <Spin spinning={loading}>
          <Table 
            columns={columns} 
            dataSource={models}
            rowKey="key"
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              showTotal: (total) => `共 ${total} 个模型`
            }}
          />
        </Spin>
      </Card>

      <Modal
        title="模型对比结果"
        visible={showComparisonModal}
        onCancel={() => setShowComparisonModal(false)}
        footer={[
          <Button key="close" onClick={() => setShowComparisonModal(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {comparisonResult && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="模型1">{comparisonResult.model1}</Descriptions.Item>
              <Descriptions.Item label="模型2">{comparisonResult.model2}</Descriptions.Item>
              <Descriptions.Item label="准确率差异">
                <span style={{ 
                  color: comparisonResult.metrics.accuracy_diff > 0 ? '#52c41a' : '#f5222d',
                  fontWeight: 'bold'
                }}>
                  {comparisonResult.metrics.accuracy_diff > 0 ? '+' : ''}
                  {comparisonResult.metrics.accuracy_diff.toFixed(2)}%
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="速度差异">
                <span style={{ 
                  color: comparisonResult.metrics.speed_diff > 0 ? '#52c41a' : '#f5222d',
                  fontWeight: 'bold'
                }}>
                  {comparisonResult.metrics.speed_diff > 0 ? '+' : ''}
                  {comparisonResult.metrics.speed_diff} tokens/s
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="模型大小比例" span={2}>
                {comparisonResult.metrics.size_ratio.toFixed(2)}x
              </Descriptions.Item>
              <Descriptions.Item label="功能重叠度" span={2}>
                <Progress percent={comparisonResult.metrics.feature_overlap * 100} />
              </Descriptions.Item>
              <Descriptions.Item label="推荐建议" span={2}>
                <Tag color="green">{comparisonResult.recommendation}</Tag>
              </Descriptions.Item>
            </Descriptions>
            
            {comparisonResult.details && (
              <div style={{ marginTop: 16 }}>
                <Title level={5}>详细对比</Title>
                <Space direction="vertical" style={{ width: '100%' }}>
                  {comparisonResult.details.model1_advantages && (
                    <Card size="small" title={`${comparisonResult.model1} 优势`}>
                      {comparisonResult.details.model1_advantages.map((adv: string, idx: number) => (
                        <Tag key={idx} color="blue" style={{ margin: 4 }}>{adv}</Tag>
                      ))}
                    </Card>
                  )}
                  {comparisonResult.details.model2_advantages && (
                    <Card size="small" title={`${comparisonResult.model2} 优势`}>
                      {comparisonResult.details.model2_advantages.map((adv: string, idx: number) => (
                        <Tag key={idx} color="green" style={{ margin: 4 }}>{adv}</Tag>
                      ))}
                    </Card>
                  )}
                </Space>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default KnowledgeModelComparisonPage;
