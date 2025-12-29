import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Button, Table, Alert, Tabs, Space, Typography, Spin } from 'antd';
import { ExperimentOutlined, ReloadOutlined, PlayCircleOutlined, StopOutlined } from '@ant-design/icons';

const { TabPane } = Tabs;

type Scenario = { id: string; name: string; type: string; severity: string; created_at?: string };
type Test = { id: string; scenario_id: string; status: string; started_at?: string; ended_at?: string };
type Impact = { metric: string; value: number };

const FaultTestingPage: React.FC = () => {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [tests, setTests] = useState<Test[]>([]);
  const [impacts, setImpacts] = useState<Impact[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sRes, tRes, iRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/fault-tolerance/faults'),
        apiFetch(buildApiUrl('/api/v1/fault-tolerance/testing/tests'),
        apiFetch(buildApiUrl('/api/v1/fault-tolerance/metrics'))
      ]);
      const sData = await sRes.json();
      const tData = await tRes.json();
      const iData = await iRes.json();
      setScenarios(Array.isArray(sData) ? sData : []);
      setTests(Array.isArray(tData?.tests) ? tData.tests : []);
      setImpacts(Array.isArray(iData?.impact) ? iData.impact : []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setScenarios([]);
      setTests([]);
      setImpacts([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const triggerTest = async (scenarioId: string) => {
    setLoading(true);
    try {
      await apiFetch(buildApiUrl('/api/v1/fault-tolerance/testing/inject'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: scenarioId })
      });
      await load();
    } catch (e: any) {
      setError(e?.message || '触发失败');
    } finally {
      setLoading(false);
    }
  };

  const stopTest = async (testId: string) => {
    setLoading(true);
    try {
      await apiFetch(buildApiUrl(`/api/v1/fault-tolerance/testing/stop/${testId}`), { method: 'POST' });
      await load();
    } catch (e: any) {
      setError(e?.message || '停止失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <ExperimentOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              容错故障测试
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Tabs defaultActiveKey="scenarios">
          <TabPane tab="故障场景" key="scenarios">
            {loading ? (
              <Spin />
            ) : (
              <Table
                rowKey="id"
                dataSource={scenarios}
                columns={[
                  { title: 'ID', dataIndex: 'id' },
                  { title: '名称', dataIndex: 'name' },
                  { title: '类型', dataIndex: 'type' },
                  { title: '级别', dataIndex: 'severity' },
                  {
                    title: '操作',
                    render: (_, r) => (
                      <Button
                        icon={<PlayCircleOutlined />}
                        type="link"
                        onClick={() => triggerTest(r.id)}
                      >
                        注入
                      </Button>
                    )
                  }
                ]}
                locale={{ emptyText: '暂无场景，先通过后端接口创建。' }}
              />
            )}
          </TabPane>

          <TabPane tab="测试执行" key="tests">
            {loading ? (
              <Spin />
            ) : (
              <Table
                rowKey="id"
                dataSource={tests}
                columns={[
                  { title: 'ID', dataIndex: 'id' },
                  { title: '场景', dataIndex: 'scenario_id' },
                  { title: '状态', dataIndex: 'status' },
                  { title: '开始', dataIndex: 'started_at' },
                  { title: '结束', dataIndex: 'ended_at' },
                  {
                    title: '操作',
                    render: (_, r) => (
                      <Button icon={<StopOutlined />} danger type="link" onClick={() => stopTest(r.id)}>
                        停止
                      </Button>
                    )
                  }
                ]}
                locale={{ emptyText: '暂无测试记录。' }}
              />
            )}
          </TabPane>

          <TabPane tab="影响指标" key="impact">
            {loading ? (
              <Spin />
            ) : (
              <Table
                rowKey="metric"
                dataSource={impacts}
                columns={[
                  { title: '指标', dataIndex: 'metric' },
                  { title: '值', dataIndex: 'value' }
                ]}
                locale={{ emptyText: '暂无影响数据。' }}
              />
            )}
          </TabPane>
        </Tabs>
      </Space>
    </div>
  );
};

export default FaultTestingPage;
