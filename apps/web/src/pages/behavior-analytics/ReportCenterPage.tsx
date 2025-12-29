import React, { useEffect, useMemo, useState } from 'react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Badge } from '../../components/ui/badge';
import { Alert } from '../../components/ui/alert';
import { behaviorAnalyticsService } from '../../services/behaviorAnalyticsService';

import { logger } from '../../utils/logger'
type ReportType = 'comprehensive' | 'summary' | 'custom';
type ReportFormat = 'json' | 'html';

interface AnalyticsReport {
  report_id: string;
  status: string;
  format: ReportFormat;
  report_type: ReportType;
  filters?: Record<string, any>;
  generated_at: string;
  summary?: Record<string, any>;
}

export const ReportCenterPage: React.FC = () => {
  const [reports, setReports] = useState<AnalyticsReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [health, setHealth] = useState<any>(null);

  const [reportType, setReportType] = useState<ReportType>('comprehensive');
  const [format, setFormat] = useState<ReportFormat>('json');
  const [includeVisualizations, setIncludeVisualizations] = useState(true);

  const [filters, setFilters] = useState({
    user_id: '',
    session_id: '',
    event_type: '',
    start_time: '',
    end_time: '',
  });

  const fetchReports = async () => {
    setLoading(true);
    try {
      const resp = await behaviorAnalyticsService.getReports({ limit: 200, offset: 0 });
      setReports((resp.reports || []) as AnalyticsReport[]);
    } catch (e) {
      logger.error('获取报告列表失败:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  const stats = useMemo(() => {
    const total = reports.length;
    const completed = reports.filter(r => r.status === 'completed').length;
    const failed = reports.filter(r => r.status === 'failed').length;
    return { total, completed, failed };
  }, [reports]);

  const formatTime = (timestamp: string) => new Date(timestamp).toLocaleString('zh-CN');

  const handleCheckHealth = async () => {
    try {
      const r = await behaviorAnalyticsService.healthCheck();
      setHealth(r);
    } catch (e) {
      setHealth(null);
      logger.error('健康检查失败:', e);
    }
  };

  const handleGenerateReport = async () => {
    setGenerating(true);
    try {
      const req: any = {
        report_type: reportType,
        format,
        include_visualizations: includeVisualizations,
      };
      const f: Record<string, any> = {};
      Object.entries(filters).forEach(([k, v]) => {
        const vv = (v || '').trim();
        if (vv) f[k] = vv;
      });
      if (Object.keys(f).length) req.filters = f;
      await behaviorAnalyticsService.generateReport(req);
      await fetchReports();
      setShowCreateForm(false);
    } catch (e) {
      logger.error('生成报告失败:', e);
    } finally {
      setGenerating(false);
    }
  };

  const handleDownloadReport = async (report: AnalyticsReport) => {
    try {
      const blob = await behaviorAnalyticsService.downloadReport(report.report_id, report.format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${report.report_id}.${report.format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      logger.error('下载报告失败:', e);
    }
  };

  const handleDeleteReport = async (reportId: string) => {
    try {
      await behaviorAnalyticsService.deleteReport(reportId);
      setReports(prev => prev.filter(r => r.report_id !== reportId));
    } catch (e) {
      logger.error('删除报告失败:', e);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">报告生成中心</h1>
          <p className="text-sm text-gray-600 mt-1">数据来源: /api/v1/analytics/reports</p>
        </div>
        <div className="flex space-x-3">
          <Button variant="outline" onClick={fetchReports}>
            🔄 刷新列表
          </Button>
          <Button variant="outline" onClick={handleCheckHealth}>
            🏥 健康检查
          </Button>
          <Button onClick={() => setShowCreateForm(v => !v)}>
            ➕ 创建报告
          </Button>
        </div>
      </div>

      {health && (
        <Alert>
          <div className="text-sm">
            <div>status: {health.status}</div>
            <div>event_store: {health.components?.event_store}</div>
          </div>
        </Alert>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{stats.total}</p>
            <p className="text-sm text-gray-600">总报告数</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">{stats.completed}</p>
            <p className="text-sm text-gray-600">已完成</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
            <p className="text-sm text-gray-600">失败</p>
          </div>
        </Card>
      </div>

      {showCreateForm && (
        <Card className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">创建新报告</h3>
            <Button variant="outline" onClick={() => setShowCreateForm(false)}>
              ✖️ 取消
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">报告类型</label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
                value={reportType}
                onChange={(e) => setReportType(e.target.value as ReportType)}
              >
                <option value="comprehensive">comprehensive</option>
                <option value="summary">summary</option>
                <option value="custom">custom</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">格式</label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
                value={format}
                onChange={(e) => setFormat(e.target.value as ReportFormat)}
              >
                <option value="json">json</option>
                <option value="html">html</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">包含可视化</label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={includeVisualizations}
                  onChange={(e) => setIncludeVisualizations(e.target.checked)}
                />
                <span className="text-sm">启用</span>
              </label>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">user_id</label>
              <Input value={filters.user_id} onChange={(e) => setFilters(s => ({ ...s, user_id: e.target.value }))} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">session_id</label>
              <Input value={filters.session_id} onChange={(e) => setFilters(s => ({ ...s, session_id: e.target.value }))} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">event_type</label>
              <Input value={filters.event_type} onChange={(e) => setFilters(s => ({ ...s, event_type: e.target.value }))} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">start_time</label>
              <Input type="date" value={filters.start_time} onChange={(e) => setFilters(s => ({ ...s, start_time: e.target.value }))} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">end_time</label>
              <Input type="date" value={filters.end_time} onChange={(e) => setFilters(s => ({ ...s, end_time: e.target.value }))} />
            </div>
          </div>

          <Button onClick={handleGenerateReport} disabled={generating}>
            {generating ? '⏳ 生成中...' : '🚀 生成报告'}
          </Button>
        </Card>
      )}

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">报告列表</h3>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2 text-gray-600">加载中...</p>
          </div>
        ) : (
          <div className="space-y-4">
            {reports.map((report) => (
              <div key={report.report_id} className="p-4 border border-gray-200 rounded-md">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <code className="text-xs">{report.report_id}</code>
                      <Badge className="bg-gray-100 text-gray-800">{report.report_type}</Badge>
                      <Badge className="bg-gray-100 text-gray-800">{report.format}</Badge>
                      <Badge className="bg-green-100 text-green-800">{report.status}</Badge>
                    </div>
                    <div className="text-sm text-gray-600">
                      生成时间: {formatTime(report.generated_at)}
                    </div>
                    {report.summary && (
                      <div className="text-sm text-gray-600 mt-2">
                        total_events: {report.summary.total_events ?? '-'} | unique_users: {report.summary.unique_users ?? '-'} | unique_sessions: {report.summary.unique_sessions ?? '-'}
                      </div>
                    )}
                  </div>
                  <div className="flex space-x-2 ml-4">
                    <Button size="sm" onClick={() => handleDownloadReport(report)}>
                      📥 下载
                    </Button>
                    <Button size="sm" variant="danger" onClick={() => handleDeleteReport(report.report_id)}>
                      🗑️ 删除
                    </Button>
                  </div>
                </div>
              </div>
            ))}

            {reports.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <span className="text-4xl mb-4 block">📄</span>
                <p>暂无报告</p>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default ReportCenterPage;
