import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Table, Tag, Button, Space, Typography, Alert, Spin } from 'antd';
import { HeartOutlined, ReloadOutlined, PlusOutlined } from '@ant-design/icons';

type Strategy = {
  strategy_id: string;
  name: string;
  type: string;
  effectiveness_score?: number;
  success_rate?: number;
  last_updated?: string;
};

type Plan = {
  plan_id: string;
  user_id: string;
  status: string;
  primary_strategy?: string;
  progress?: number;
};

type Outcome = {
  plan_id: string;
  outcome_type: string;
  improvement_score?: number;
  recorded_at?: string;
};

const InterventionStrategyManagementPage: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [plans, setPlans] = useState<Plan[]>([]);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sRes, pRes, oRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/emotional-intelligence/strategies'),
        apiFetch(buildApiUrl('/api/v1/emotional-intelligence/plans'),
        apiFetch(buildApiUrl('/api/v1/emotional-intelligence/outcomes'))
      ]);
      setStrategies((await sRes.json())?.strategies || []);
      setPlans((await pRes.json())?.plans || []);
      setOutcomes((await oRes.json())?.outcomes || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setStrategies([]);
      setPlans([]);
      setOutcomes([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <HeartOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              干预策略管理
            </Typography.Title>
          </Space>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
              刷新
            </Button>
            <Button icon={<PlusOutlined />} disabled>
              创建（请走后端接口）
            </Button>
          </Space>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="策略库">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="strategy_id"
              dataSource={strategies}
              columns={[
                { title: '名称', dataIndex: 'name' },
                { title: '类型', dataIndex: 'type', render: (v) => <Tag>{v}</Tag> },
                { title: '效果评分', dataIndex: 'effectiveness_score' },
                { title: '成功率', dataIndex: 'success_rate' },
                { title: '更新时间', dataIndex: 'last_updated' }
              ]}
              locale={{ emptyText: '暂无策略数据，先通过后端接口创建。' }}
            />
          )}
        </Card>

        <Card title="干预计划">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="plan_id"
              dataSource={plans}
              columns={[
                { title: '计划ID', dataIndex: 'plan_id' },
                { title: '用户', dataIndex: 'user_id' },
                { title: '状态', dataIndex: 'status', render: (v) => <Tag>{v}</Tag> },
                { title: '主策略', dataIndex: 'primary_strategy' },
                { title: '进度', dataIndex: 'progress' }
              ]}
              locale={{ emptyText: '暂无计划数据。' }}
            />
          )}
        </Card>

        <Card title="干预结果">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="plan_id"
              dataSource={outcomes}
              columns={[
                { title: '计划ID', dataIndex: 'plan_id' },
                { title: '结果', dataIndex: 'outcome_type' },
                { title: '改善评分', dataIndex: 'improvement_score' },
                { title: '时间', dataIndex: 'recorded_at' }
              ]}
              locale={{ emptyText: '暂无结果数据。' }}
            />
          )}
        </Card>
      </Space>
    </div>
  );
};

export default InterventionStrategyManagementPage;
