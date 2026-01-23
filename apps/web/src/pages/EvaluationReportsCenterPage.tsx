import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Button, Card, Form, Input, Space, Table, Typography, message } from 'antd';
import { FileTextOutlined, ReloadOutlined } from '@ant-design/icons';

import { logger } from '../utils/logger'
const { Title, Text } = Typography;

interface EvaluationJob {
  job_id: string;
  status: string;
  created_at: string;
  completed_at?: string | null;
  progress: number;
  models_count: number;
  benchmarks_count: number;
}

const EvaluationReportsCenterPage: React.FC = () => {
  const [jobs, setJobs] = useState<EvaluationJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [reportJobId, setReportJobId] = useState<string | null>(null);
  const [form] = Form.useForm();

  const loadCompletedJobs = async () => {
    try {
      setLoading(true);
      const res = await apiFetch(buildApiUrl('/api/v1/model-evaluation/jobs?status=completed&limit=50&offset=0'));
      const data = await res.json();
      setJobs(Array.isArray(data?.jobs) ? data.jobs : []);
    } catch (e) {
      logger.error('加载评估任务失败:', e);
      message.error('加载评估任务失败');
      setJobs([]);
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async () => {
    if (!selectedJobIds.length) {
      message.error('请选择至少一个已完成任务');
      return;
    }
    try {
      setGenerating(true);
      const values = await form.validateFields();
      const res = await apiFetch(buildApiUrl('/api/v1/model-evaluation/generate-report'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evaluation_ids: selectedJobIds,
          title: values.title,
          subtitle: values.subtitle || null,
          include_charts: true,
          include_detailed_metrics: true,
          include_recommendations: true,
          output_format: 'html',
        }),
      });
      const data = await res.json();
      setReportJobId(data?.report_job_id || null);
      message.success('报告生成任务已启动');
    } catch (e) {
      logger.error('生成报告失败:', e);
      message.error('生成报告失败');
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    loadCompletedJobs();
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <Title level={2} style={{ margin: 0 }}>
            <FileTextOutlined style={{ marginRight: 12 }} />
            报告生成中心
          </Title>
          <Text type="secondary">基于已完成评估任务生成 HTML 报告</Text>
        </div>
        <Space>
          <Button onClick={loadCompletedJobs} icon={<ReloadOutlined />} loading={loading}>
            刷新
          </Button>
        </Space>
      </div>

      <Card title="选择已完成任务" style={{ marginBottom: 16 }}>
        <Table
          rowKey="job_id"
          loading={loading}
          dataSource={jobs}
          pagination={false}
          rowSelection={{
            selectedRowKeys: selectedJobIds,
            onChange: (keys) => setSelectedJobIds(keys as string[]),
          }}
          columns={[
            { title: '任务ID', dataIndex: 'job_id' },
            { title: '模型数', dataIndex: 'models_count' },
            { title: '基准测试数', dataIndex: 'benchmarks_count' },
            {
              title: '完成时间',
              dataIndex: 'completed_at',
              render: (v: any) => (v ? new Date(v).toLocaleString() : '-'),
            },
          ]}
        />
      </Card>

      <Card title="生成报告">
        <Form form={form} layout="vertical" initialValues={{ title: '模型评估报告' }}>
          <Form.Item name="title" label="标题" rules={[{ required: true, message: '请输入标题' }]}>
            <Input />
          </Form.Item>
          <Form.Item name="subtitle" label="副标题">
            <Input />
          </Form.Item>
          <Space>
            <Button type="primary" onClick={generateReport} loading={generating}>
              生成
            </Button>
            {reportJobId ? (
              <Button onClick={() => window.open(`/api/v1/model-evaluation/reports/${reportJobId}`, '_blank')}>打开报告</Button>
            ) : null}
          </Space>
          {reportJobId ? (
            <div style={{ marginTop: 12 }}>
              <Text type="secondary">报告ID: </Text>
              <Text code>{reportJobId}</Text>
              <div>
                <Text type="secondary">接口: </Text>
                <Text code>/api/v1/model-evaluation/reports/{reportJobId}</Text>
              </div>
            </div>
          ) : null}
        </Form>
      </Card>
    </div>
  );
};

export default EvaluationReportsCenterPage;
