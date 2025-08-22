import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Progress, Table, Tag, Button, Space, Statistic, Alert, Timeline, Tooltip, Switch } from 'antd';
import { DatabaseOutlined, CloudDownloadOutlined, CompressOutlined, ClockCircleOutlined, DeleteOutlined, ReloadOutlined, FileZipOutlined } from '@ant-design/icons';

interface ModelMetadata {
  modelId: string;
  version: string;
  sizeBytes: number;
  checksum: string;
  createdAt: string;
  lastUsed: string;
  useCount: number;
  compressionType: string;
  quantizationLevel?: string;
  tags: string[];
  isLoaded: boolean;
  compressionRatio: number;
}

interface CacheStats {
  totalModels: number;
  totalSizeBytes: number;
  maxCacheSizeBytes: number;
  memoryLoadedModels: number;
  averageModelSize: number;
  mostUsedModel?: string;
  leastUsedModel?: string;
  cacheUsagePercent: number;
}

const ModelCacheMonitorPage: React.FC = () => {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [cacheStats, setCacheStats] = useState<CacheStats>({
    totalModels: 12,
    totalSizeBytes: 2684354560, // ~2.5GB
    maxCacheSizeBytes: 5368709120, // 5GB
    memoryLoadedModels: 3,
    averageModelSize: 223696213, // ~213MB
    mostUsedModel: 'claude-3-haiku-quantized',
    leastUsedModel: 'gpt-4-base',
    cacheUsagePercent: 50.0
  });
  const [autoCleanup, setAutoCleanup] = useState(true);
  const [compressionEnabled, setCompressionEnabled] = useState(true);

  useEffect(() => {
    generateMockData();
  }, []);

  const generateMockData = () => {
    const mockModels: ModelMetadata[] = [
      {
        modelId: 'claude-3-haiku-quantized',
        version: '1.2.0',
        sizeBytes: 167772160, // 160MB
        checksum: 'sha256:abc123def456...',
        createdAt: '2024-01-10T08:30:00Z',
        lastUsed: '2024-01-15T14:20:00Z',
        useCount: 47,
        compressionType: 'gzip',
        quantizationLevel: 'int8',
        tags: ['reasoning', 'fast', 'quantized'],
        isLoaded: true,
        compressionRatio: 0.31
      },
      {
        modelId: 'gpt-4-turbo-preview',
        version: '2.1.0',
        sizeBytes: 536870912, // 512MB
        checksum: 'sha256:def456ghi789...',
        createdAt: '2024-01-12T10:15:00Z',
        lastUsed: '2024-01-15T13:45:00Z',
        useCount: 23,
        compressionType: 'gzip',
        tags: ['multimodal', 'large', 'preview'],
        isLoaded: true,
        compressionRatio: 0.28
      },
      {
        modelId: 'llama-2-13b-chat',
        version: '1.0.0',
        sizeBytes: 805306368, // 768MB
        checksum: 'sha256:ghi789jkl012...',
        createdAt: '2024-01-08T16:45:00Z',
        lastUsed: '2024-01-14T09:30:00Z',
        useCount: 15,
        compressionType: 'gzip',
        quantizationLevel: 'fp16',
        tags: ['chat', 'open-source'],
        isLoaded: false,
        compressionRatio: 0.35
      },
      {
        modelId: 'mistral-7b-instruct',
        version: '0.2.0',
        sizeBytes: 268435456, // 256MB
        checksum: 'sha256:jkl012mno345...',
        createdAt: '2024-01-14T12:00:00Z',
        lastUsed: '2024-01-15T11:15:00Z',
        useCount: 8,
        compressionType: 'gzip',
        quantizationLevel: 'int4',
        tags: ['instruct', 'efficient', 'quantized'],
        isLoaded: true,
        compressionRatio: 0.42
      },
      {
        modelId: 'embeddings-ada-002',
        version: '1.0.0',
        sizeBytes: 134217728, // 128MB
        checksum: 'sha256:mno345pqr678...',
        createdAt: '2024-01-09T14:20:00Z',
        lastUsed: '2024-01-13T16:40:00Z',
        useCount: 31,
        compressionType: 'gzip',
        tags: ['embeddings', 'search'],
        isLoaded: false,
        compressionRatio: 0.25
      },
      {
        modelId: 'gpt-4-base',
        version: '1.0.0',
        sizeBytes: 1073741824, // 1GB
        checksum: 'sha256:pqr678stu901...',
        createdAt: '2024-01-05T09:10:00Z',
        lastUsed: '2024-01-11T08:25:00Z',
        useCount: 3,
        compressionType: 'gzip',
        tags: ['base', 'large'],
        isLoaded: false,
        compressionRatio: 0.33
      }
    ];
    
    setModels(mockModels);
    
    // 更新统计信息
    const totalSize = mockModels.reduce((sum, model) => sum + model.sizeBytes, 0);
    const loadedCount = mockModels.filter(model => model.isLoaded).length;
    const avgSize = totalSize / mockModels.length;
    const usageCounts = mockModels.map(m => m.useCount);
    const mostUsed = mockModels.find(m => m.useCount === Math.max(...usageCounts));
    const leastUsed = mockModels.find(m => m.useCount === Math.min(...usageCounts));
    
    setCacheStats({
      totalModels: mockModels.length,
      totalSizeBytes: totalSize,
      maxCacheSizeBytes: 5368709120, // 5GB
      memoryLoadedModels: loadedCount,
      averageModelSize: avgSize,
      mostUsedModel: mostUsed?.modelId,
      leastUsedModel: leastUsed?.modelId,
      cacheUsagePercent: (totalSize / 5368709120) * 100
    });
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getQuantizationColor = (level?: string) => {
    switch (level) {
      case 'int4': return 'red';
      case 'int8': return 'orange';
      case 'fp16': return 'blue';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: '模型ID',
      dataIndex: 'modelId',
      key: 'modelId',
      render: (id: string, record: ModelMetadata) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{id}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>v{record.version}</div>
        </div>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (record: ModelMetadata) => (
        <Space direction="vertical" size={2}>
          {record.isLoaded ? (
            <Tag color="green" icon={<DatabaseOutlined />}>已加载</Tag>
          ) : (
            <Tag color="default">磁盘缓存</Tag>
          )}
          {record.quantizationLevel && (
            <Tag color={getQuantizationColor(record.quantizationLevel)}>
              {record.quantizationLevel.toUpperCase()}
            </Tag>
          )}
        </Space>
      )
    },
    {
      title: '大小',
      key: 'size',
      render: (record: ModelMetadata) => (
        <div>
          <div>{formatBytes(record.sizeBytes)}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            压缩比: {(record.compressionRatio * 100).toFixed(0)}%
          </div>
        </div>
      )
    },
    {
      title: '使用统计',
      key: 'usage',
      render: (record: ModelMetadata) => (
        <div>
          <div>使用次数: <span style={{ fontWeight: 'bold' }}>{record.useCount}</span></div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            最后使用: {new Date(record.lastUsed).toLocaleDateString()}
          </div>
        </div>
      )
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <div>
          {tags.map(tag => (
            <Tag key={tag} size="small">{tag}</Tag>
          ))}
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ModelMetadata) => (
        <Space>
          {!record.isLoaded && (
            <Tooltip title="加载到内存">
              <Button size="small" icon={<CloudDownloadOutlined />} />
            </Tooltip>
          )}
          <Tooltip title="重新压缩">
            <Button size="small" icon={<CompressOutlined />} />
          </Tooltip>
          <Tooltip title="删除缓存">
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Tooltip>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1>🗄️ 本地模型缓存监控</h1>
        <p>监控和管理本地AI模型缓存，包括模型压缩、量化、预加载等机制。</p>
      </div>

      {/* 缓存统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={4}>
          <Card>
            <Statistic 
              title="缓存模型" 
              value={cacheStats.totalModels} 
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="内存加载" 
              value={cacheStats.memoryLoadedModels} 
              suffix={`/ ${cacheStats.totalModels}`}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="缓存使用率" 
              value={cacheStats.cacheUsagePercent} 
              precision={1}
              suffix="%" 
              valueStyle={{ 
                color: cacheStats.cacheUsagePercent > 80 ? '#cf1322' : '#3f8600' 
              }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="总缓存大小" 
              value={formatBytes(cacheStats.totalSizeBytes)} 
              prefix={<FileZipOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="平均模型大小" 
              value={formatBytes(cacheStats.averageModelSize)} 
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="最大缓存" 
              value={formatBytes(cacheStats.maxCacheSizeBytes)} 
            />
          </Card>
        </Col>
      </Row>

      {/* 缓存空间使用情况 */}
      <Card title="💾 缓存空间使用情况" style={{ marginBottom: '24px' }}>
        <Row gutter={16}>
          <Col span={16}>
            <div style={{ marginBottom: '16px' }}>
              <Progress 
                percent={cacheStats.cacheUsagePercent} 
                strokeColor={{
                  '0%': '#108ee9',
                  '50%': '#87d068',
                  '80%': '#ffcf3c',
                  '100%': '#ff6b6b',
                }}
                status={cacheStats.cacheUsagePercent > 90 ? 'exception' : 'active'}
              />
            </div>
            <Row gutter={16}>
              <Col span={8}>
                <Card size="small">
                  <Statistic 
                    title="已使用" 
                    value={formatBytes(cacheStats.totalSizeBytes)}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic 
                    title="剩余空间" 
                    value={formatBytes(cacheStats.maxCacheSizeBytes - cacheStats.totalSizeBytes)}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic 
                    title="总容量" 
                    value={formatBytes(cacheStats.maxCacheSizeBytes)}
                  />
                </Card>
              </Col>
            </Row>
          </Col>
          <Col span={8}>
            <div style={{ background: '#f5f5f5', padding: '16px', borderRadius: '6px' }}>
              <h4>自动管理设置</h4>
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>自动清理</span>
                  <Switch checked={autoCleanup} onChange={setAutoCleanup} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>压缩优化</span>
                  <Switch checked={compressionEnabled} onChange={setCompressionEnabled} />
                </div>
                <Button type="primary" block icon={<ReloadOutlined />}>
                  手动清理
                </Button>
              </Space>
            </div>
          </Col>
        </Row>
      </Card>

      <Row gutter={16}>
        {/* 模型列表 */}
        <Col span={16}>
          <Card title="📋 缓存模型列表">
            <Table 
              dataSource={models} 
              columns={columns}
              rowKey="modelId"
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </Col>

        {/* 缓存管理策略 */}
        <Col span={8}>
          <Card title="⚙️ 缓存管理策略" style={{ marginBottom: '16px' }}>
            <Timeline size="small">
              <Timeline.Item color="blue">
                <strong>LRU淘汰</strong><br />
                <span style={{ fontSize: '12px', color: '#666' }}>
                  最少使用的模型优先淘汰
                </span>
              </Timeline.Item>
              <Timeline.Item color="green">
                <strong>智能预加载</strong><br />
                <span style={{ fontSize: '12px', color: '#666' }}>
                  根据使用模式预测并预加载
                </span>
              </Timeline.Item>
              <Timeline.Item color="orange">
                <strong>压缩存储</strong><br />
                <span style={{ fontSize: '12px', color: '#666' }}>
                  GZIP压缩减少存储空间
                </span>
              </Timeline.Item>
              <Timeline.Item color="purple">
                <strong>增量更新</strong><br />
                <span style={{ fontSize: '12px', color: '#666' }}>
                  只下载模型变更部分
                </span>
              </Timeline.Item>
              <Timeline.Item color="red">
                <strong>校验完整性</strong><br />
                <span style={{ fontSize: '12px', color: '#666' }}>
                  SHA256校验确保数据完整
                </span>
              </Timeline.Item>
            </Timeline>
          </Card>

          <Card title="📊 使用热度分析" size="small">
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>
                最常用模型
              </div>
              <Tag color="green">{cacheStats.mostUsedModel}</Tag>
            </div>
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>
                最少用模型
              </div>
              <Tag color="red">{cacheStats.leastUsedModel}</Tag>
            </div>
            <Alert 
              message="缓存优化建议"
              description="建议清理超过30天未使用的大型模型，保留高频使用的量化版本。"
              variant="default"
              size="small"
              showIcon
            />
          </Card>
        </Col>
      </Row>

      {/* 压缩技术说明 */}
      <Card title="🗜️ 模型压缩与量化技术" style={{ marginTop: '24px' }}>
        <Row gutter={16}>
          <Col span={8}>
            <Card size="small" title="压缩算法">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div><Tag color="blue">GZIP</Tag> 通用压缩，压缩比30-40%</div>
                <div><Tag color="green">LZ4</Tag> 快速压缩，低延迟</div>
                <div><Tag color="orange">ZSTD</Tag> 高压缩比，平衡性能</div>
              </Space>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" title="量化技术">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div><Tag color="red">INT4</Tag> 4位整数，最大压缩</div>
                <div><Tag color="orange">INT8</Tag> 8位整数，平衡精度</div>
                <div><Tag color="blue">FP16</Tag> 16位浮点，高精度</div>
              </Space>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" title="优化效果">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>存储空间: <span style={{ color: '#52c41a' }}>节省70%</span></div>
                <div>加载速度: <span style={{ color: '#1890ff' }}>提升3x</span></div>
                <div>推理延迟: <span style={{ color: '#faad14' }}>降低50%</span></div>
              </Space>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default ModelCacheMonitorPage;