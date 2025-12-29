import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Badge } from '../../components/ui/badge';
import { Progress } from '../../components/ui/progress';
import { Alert } from '../../components/ui/alert';

interface ExportTask {
  task_id: string;
  title: string;
  data_type: string;
  format: string;
  status: string;
  progress?: number;
  created_at?: string;
  completed_at?: string;
  file_size?: number;
  record_count?: number;
  download_url?: string;
  error_message?: string;
}

const DataExportPage: React.FC = () => {
  const [tasks, setTasks] = useState<ExportTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState('');

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/analytics/exports'));
      const data = await res.json();
      setTasks(Array.isArray(data?.tasks) ? data.tasks : []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setTasks([]);
    } finally {
      setLoading(false);
    }
  };

  const createTask = async () => {
    setLoading(true);
    setError(null);
    try {
      await apiFetch(buildApiUrl('/api/v1/analytics/exports'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle || 'export', data_type: 'events'), format: 'csv' })
      });
      setNewTitle('');
      await load();
    } catch (e: any) {
      setError(e?.message || '创建失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-lg font-semibold">数据导出任务</span>
        </div>
        <div className="flex items-center space-x-2">
          <Input value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="任务标题" />
          <Button onClick={createTask} disabled={loading}>新建</Button>
          <Button onClick={load} disabled={loading}>刷新</Button>
        </div>
      </div>

      {error && <Alert variant="destructive">{error}</Alert>}

      {loading ? (
        <div className="text-sm text-gray-500">加载中...</div>
      ) : tasks.length === 0 ? (
        <Alert>暂无导出任务，先创建一个。</Alert>
      ) : (
        <div className="grid gap-3 grid-cols-1 md:grid-cols-2">
          {tasks.map((t) => (
            <Card key={t.task_id} className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="font-semibold">{t.title || t.task_id}</div>
                <Badge>{t.status}</Badge>
              </div>
              <div className="text-xs text-gray-500">类型: {t.data_type} / {t.format}</div>
              <div className="text-xs text-gray-500">创建: {t.created_at || '-'}</div>
              {typeof t.progress === 'number' && (
                <Progress value={t.progress} className="h-2" />
              )}
              {t.download_url && (
                <a className="text-blue-600 text-sm" href={t.download_url}>下载</a>
              )}
              {t.error_message && <Alert variant="destructive">{t.error_message}</Alert>}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default DataExportPage;
