import React, { useEffect, useMemo, useState } from 'react';
import { Card, Upload, Button, Table, Typography, Row, Col, Statistic, Space, message } from 'antd';
import { FileImageOutlined, UploadOutlined, ReloadOutlined, CheckCircleOutlined } from '@ant-design/icons';
import type { UploadRequestOption as RcCustomRequestOptions } from 'rc-upload/lib/interface';
import { fineTuningService, Dataset } from '../services/fineTuningService';

import { logger } from '../utils/logger'
const { Title, Text } = Typography;

const formatBytes = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(size >= 10 || idx === 0 ? 0 : 1)}${units[idx]}`;
};

const FineTuningDatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    try {
      setLoading(true);
      const data = await fineTuningService.getDatasets();
      setDatasets(data.datasets || []);
    } catch (e) {
      logger.error('加载数据集失败:', e);
      message.error('加载数据集失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const rows = useMemo(
    () =>
      datasets.map(d => ({
        key: d.filename,
        ...d,
      })),
    [datasets]
  );

  const columns = [
    { title: '文件名', dataIndex: 'filename', key: 'filename' },
    { title: '路径', dataIndex: 'path', key: 'path' },
    { title: '大小', dataIndex: 'size', key: 'size', render: (v: number) => formatBytes(v) },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, record: Dataset) => (
        <Button
          size="small"
          icon={<CheckCircleOutlined />}
          onClick={async () => {
            try {
              await fineTuningService.validateDatasetFormat(record.filename);
              message.success('验证通过');
            } catch (e: any) {
              const detail = e?.response?.data?.detail;
              message.error(detail ? `验证失败: ${detail}` : '验证失败');
            }
          }}
        >
          验证
        </Button>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <FileImageOutlined style={{ marginRight: 8, color: '#52c41a' }} />
          数据集管理
        </Title>
        <Text type="secondary">上传、验证和管理训练数据集</Text>
      </div>

      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic title="数据集总数" value={datasets.length} prefix={<FileImageOutlined />} />
          </Card>
        </Col>
        <Col span={18}>
          <Card>
            <Space>
              <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
                刷新
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="上传数据集" style={{ marginBottom: 16 }}>
        <Upload
          multiple={false}
          showUploadList={false}
          customRequest={async (options: RcCustomRequestOptions) => {
            try {
              const file = options.file as File;
              await fineTuningService.uploadDataset(file, file.name.replace(/\.[^/.]+$/, ''));
              options.onSuccess?.(null as any);
              message.success('上传成功');
              await load();
            } catch (e: any) {
              options.onError?.(e);
              const detail = e?.response?.data?.detail;
              message.error(detail ? `上传失败: ${detail}` : '上传失败');
            }
          }}
        >
          <Button icon={<UploadOutlined />}>选择文件</Button>
        </Upload>
      </Card>

      <Card title="数据集列表">
        <Table columns={columns} dataSource={rows} loading={loading} />
      </Card>
    </div>
  );
};

export default FineTuningDatasetsPage;
