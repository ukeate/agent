/**
 * 数据工具面板
 * 展示向量数据的导入导出和管理功能
 */

import React, { useState } from 'react';
import { Card, Upload, Button, Select, Table, Progress, Space, Alert, Tabs, message, Row, Col } from 'antd';
import { ImportOutlined, ExportOutlined, CloudUploadOutlined, DatabaseOutlined } from '@ant-design/icons';

const DataToolsPanel: React.FC = () => {
  const [exportFormat, setExportFormat] = useState('json');
  const [importFormat, setImportFormat] = useState('csv');

  const importHistory = [
    { id: 1, file: 'vectors_001.csv', format: 'CSV', status: 'completed', records: 10000, date: '2024-01-15' },
    { id: 2, file: 'embeddings.json', format: 'JSON', status: 'processing', records: 5000, date: '2024-01-14' },
    { id: 3, file: 'data.h5', format: 'HDF5', status: 'failed', records: 0, date: '2024-01-13' }
  ];

  const exportHistory = [
    { id: 1, table: 'documents', format: 'JSON', status: 'completed', records: 8500, date: '2024-01-15' },
    { id: 2, table: 'images', format: 'Parquet', status: 'processing', records: 3200, date: '2024-01-14' }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'processing';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const importColumns = [
    { title: '文件名', dataIndex: 'file', key: 'file' },
    { title: '格式', dataIndex: 'format', key: 'format' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (status: string) => (
      <Progress 
        percent={status === 'completed' ? 100 : status === 'processing' ? 60 : 0}
        status={status === 'failed' ? 'exception' : undefined}
        size="small"
      />
    )},
    { title: '记录数', dataIndex: 'records', key: 'records' },
    { title: '日期', dataIndex: 'date', key: 'date' }
  ];

  const exportColumns = [
    { title: '表名', dataIndex: 'table', key: 'table' },
    { title: '格式', dataIndex: 'format', key: 'format' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (status: string) => (
      <Progress 
        percent={status === 'completed' ? 100 : status === 'processing' ? 45 : 0}
        size="small"
      />
    )},
    { title: '记录数', dataIndex: 'records', key: 'records' },
    { title: '日期', dataIndex: 'date', key: 'date' }
  ];

  return (
    <div>
      <Alert
        message="向量数据管理工具"
        description="支持多种格式的向量数据导入导出，包括CSV、JSON、Parquet、HDF5等格式，以及数据迁移和备份功能。"
        variant="default"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Tabs
        items={[
          {
            key: 'import',
            label: <span><ImportOutlined />数据导入</span>,
            children: (
              <Row gutter={[24, 24]}>
                <Col span={8}>
                  <Card title="导入配置" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <label>数据格式</label>
                        <Select value={importFormat} onChange={setImportFormat} style={{ width: '100%' }}>
                          <Select.Option value="csv">CSV</Select.Option>
                          <Select.Option value="json">JSON</Select.Option>
                          <Select.Option value="jsonl">JSONL</Select.Option>
                          <Select.Option value="parquet">Parquet</Select.Option>
                          <Select.Option value="hdf5">HDF5</Select.Option>
                          <Select.Option value="numpy">NumPy</Select.Option>
                        </Select>
                      </div>

                      <Upload.Dragger>
                        <p><CloudUploadOutlined style={{ fontSize: 48 }} /></p>
                        <p>点击或拖拽文件到此区域上传</p>
                        <p style={{ color: '#999' }}>支持多种向量数据格式</p>
                      </Upload.Dragger>

                      <Button type="primary" block>
                        开始导入
                      </Button>
                    </Space>
                  </Card>
                </Col>

                <Col span={16}>
                  <Card title="导入历史" size="small">
                    <Table
                      columns={importColumns}
                      dataSource={importHistory}
                      rowKey="id"
                      size="small"
                      pagination={false}
                    />
                  </Card>
                </Col>
              </Row>
            )
          },
          {
            key: 'export',
            label: <span><ExportOutlined />数据导出</span>,
            children: (
              <Row gutter={[24, 24]}>
                <Col span={8}>
                  <Card title="导出配置" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <label>源表</label>
                        <Select placeholder="选择表" style={{ width: '100%' }}>
                          <Select.Option value="documents">documents</Select.Option>
                          <Select.Option value="images">images</Select.Option>
                          <Select.Option value="audio">audio</Select.Option>
                        </Select>
                      </div>

                      <div>
                        <label>导出格式</label>
                        <Select value={exportFormat} onChange={setExportFormat} style={{ width: '100%' }}>
                          <Select.Option value="json">JSON</Select.Option>
                          <Select.Option value="csv">CSV</Select.Option>
                          <Select.Option value="parquet">Parquet</Select.Option>
                          <Select.Option value="hdf5">HDF5</Select.Option>
                        </Select>
                      </div>

                      <div>
                        <label>压缩方式</label>
                        <Select placeholder="选择压缩" style={{ width: '100%' }}>
                          <Select.Option value="none">无压缩</Select.Option>
                          <Select.Option value="gzip">GZIP</Select.Option>
                          <Select.Option value="bz2">BZ2</Select.Option>
                        </Select>
                      </div>

                      <Button type="primary" block>
                        开始导出
                      </Button>
                    </Space>
                  </Card>
                </Col>

                <Col span={16}>
                  <Card title="导出历史" size="small">
                    <Table
                      columns={exportColumns}
                      dataSource={exportHistory}
                      rowKey="id"
                      size="small"
                      pagination={false}
                    />
                  </Card>
                </Col>
              </Row>
            )
          },
          {
            key: 'migration',
            label: <span><DatabaseOutlined />数据迁移</span>,
            children: (
              <Row gutter={[24, 24]}>
                <Col span={12}>
                  <Card title="数据库迁移" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <label>源数据库</label>
                        <Select placeholder="选择源数据库" style={{ width: '100%' }}>
                          <Select.Option value="postgres1">PostgreSQL-1</Select.Option>
                          <Select.Option value="postgres2">PostgreSQL-2</Select.Option>
                        </Select>
                      </div>

                      <div>
                        <label>目标数据库</label>
                        <Select placeholder="选择目标数据库" style={{ width: '100%' }}>
                          <Select.Option value="postgres3">PostgreSQL-3</Select.Option>
                        </Select>
                      </div>

                      <Button type="primary" block>
                        开始迁移
                      </Button>
                    </Space>
                  </Card>
                </Col>

                <Col span={12}>
                  <Card title="备份恢复" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Button block>创建备份</Button>
                      <Upload>
                        <Button block>恢复备份</Button>
                      </Upload>
                      <Button block>备份列表</Button>
                    </Space>
                  </Card>
                </Col>
              </Row>
            )
          }
        ]}
      />
    </div>
  );
};

export default DataToolsPanel;