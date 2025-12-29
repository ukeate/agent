import React, { useState } from 'react';
import { Card, Table, Button, Select, Space, Alert, Statistic, Row, Col, message } from 'antd';
import { LineChartOutlined } from '@ant-design/icons';
import { pgvectorApi } from '../../services/pgvectorApi';

interface QueryPerformance {
  total_queries: number;
  average_execution_time_ms: number;
  min_execution_time_ms: number;
  max_execution_time_ms: number;
}

const TemporalVectorPanel: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h'>('24h');
  const [metrics, setMetrics] = useState<QueryPerformance | null>(null);
  const [loading, setLoading] = useState(false);

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const { query_performance } = await pgvectorApi.getPerformanceMetrics(timeRange);
      setMetrics(query_performance);
    } catch (e) {
      message.error('获取向量性能指标失败');
    } finally {
      setLoading(false);
    }
  };

  const rows = metrics
    ? [
        {
          key: 'summary',
          total: metrics.total_queries,
          avg: metrics.average_execution_time_ms,
          min: metrics.min_execution_time_ms,
          max: metrics.max_execution_time_ms,
        },
      ]
    : [];

  const columns = [
    { title: '总查询数', dataIndex: 'total', key: 'total' },
    { title: '平均耗时(ms)', dataIndex: 'avg', key: 'avg' },
    { title: '最小耗时(ms)', dataIndex: 'min', key: 'min' },
    { title: '最大耗时(ms)', dataIndex: 'max', key: 'max' },
  ];

  return (
    <div>
      <Alert
        message="向量查询性能"
        description="直接从后端性能监控中读取真实查询指标，展示指定时间窗口内的执行情况。"
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        <Col span={8}>
          <Card title="查询窗口" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Select value={timeRange} onChange={setTimeRange} options={[
                { label: '最近1小时', value: '1h' },
                { label: '最近6小时', value: '6h' },
                { label: '最近24小时', value: '24h' },
              ]} />
              <Button type="primary" block onClick={loadMetrics} loading={loading}>
                拉取性能数据
              </Button>
            </Space>
          </Card>

          <Card title="关键指标" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={12}>
                <Statistic title="总查询数" value={metrics?.total_queries ?? 0} />
              </Col>
              <Col span={12}>
                <Statistic title="平均耗时(ms)" value={metrics?.average_execution_time_ms ?? 0} />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Card title="查询统计" size="small">
            <Table columns={columns} dataSource={rows} pagination={false} size="small" />
          </Card>

          <Card title="时序图占位" size="small" style={{ marginTop: 16 }}>
            <div
              style={{
                height: 200,
                backgroundColor: '#fafafa',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px dashed #d9d9d9',
              }}
            >
              <Space direction="vertical" align="center">
                <LineChartOutlined style={{ fontSize: 32, color: '#d9d9d9' }} />
                <span>后端未提供时序点，待性能采集落库后呈现</span>
              </Space>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TemporalVectorPanel;
