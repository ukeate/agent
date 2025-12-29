import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Button, Alert, Space } from 'antd';

type Task = { task_id: string; dataset?: string; status?: string; assignee?: string; progress?: number };

const DataAnnotationManagementPage: React.FC = () => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/training-data/annotation-tasks'));
      const data = await res.json();
      setTasks(data?.tasks || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setTasks([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Button onClick={load} loading={loading}>刷新</Button>
        {error && <Alert type="error" message={error} />}
        <Card title="标注任务">
          <Table
            rowKey="task_id"
            loading={loading}
            dataSource={tasks}
            locale={{ emptyText: '暂无任务，请先通过后端创建标注任务。' }}
            columns={[
              { title: '任务ID', dataIndex: 'task_id' },
              { title: '数据集', dataIndex: 'dataset' },
              { title: '指派给', dataIndex: 'assignee' },
              { title: '状态', dataIndex: 'status' },
              { title: '进度', dataIndex: 'progress' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default DataAnnotationManagementPage;
