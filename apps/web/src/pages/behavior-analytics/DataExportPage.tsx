import React, { useState, useEffect } from 'react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Badge } from '../../components/ui/badge';
import { Progress } from '../../components/ui/progress';
import { Alert } from '../../components/ui/alert';
import { behaviorAnalyticsService } from '../../services/behaviorAnalyticsService';

interface ExportTask {
  task_id: string;
  title: string;
  data_type: 'events' | 'sessions' | 'patterns' | 'anomalies' | 'reports';
  format: 'csv' | 'json' | 'xlsx' | 'parquet';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  completed_at?: string;
  file_size?: number;
  record_count?: number;
  download_url?: string;
  error_message?: string;
  parameters: Record<string, any>;
}

interface ExportConfig {
  data_type: 'events' | 'sessions' | 'patterns' | 'anomalies' | 'reports';
  format: 'csv' | 'json' | 'xlsx' | 'parquet';
  title: string;
  filters: Record<string, any>;
  fields?: string[];
  compression?: boolean;
  split_by_date?: boolean;
  max_records?: number;
}

const DATA_TYPES = [
  { value: 'events', label: '行为事件', icon: '📝' },
  { value: 'sessions', label: '用户会话', icon: '👥' },
  { value: 'patterns', label: '行为模式', icon: '🔍' },
  { value: 'anomalies', label: '异常检测结果', icon: '⚠️' },
  { value: 'reports', label: '分析报告', icon: '📊' }
];

const EXPORT_FORMATS = [
  { value: 'csv', label: 'CSV', icon: '📄', description: '通用表格格式，兼容性好' },
  { value: 'json', label: 'JSON', icon: '🔧', description: '结构化数据，适合API集成' },
  { value: 'xlsx', label: 'Excel', icon: '📊', description: 'Excel表格，支持多工作表' },
  { value: 'parquet', label: 'Parquet', icon: '💾', description: '列式存储，高性能分析' }
];

export const DataExportPage: React.FC = () => {
  const [exportTasks, setExportTasks] = useState<ExportTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [exportConfig, setExportConfig] = useState<ExportConfig>({
    data_type: 'events',
    format: 'csv',
    title: '',
    filters: {},
    compression: true,
    split_by_date: false,
    max_records: 100000
  });
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);

  // 获取导出任务列表
  const fetchExportTasks = async () => {
    setLoading(true);
    try {
      // TODO: 调用获取导出任务列表API
      const mockTasks: ExportTask[] = [
        {
          task_id: 'export-001',
          title: '用户行为事件导出',
          data_type: 'events',
          format: 'csv',
          status: 'completed',
          progress: 100,
          created_at: new Date(Date.now() - 3600000).toISOString(),
          completed_at: new Date(Date.now() - 3000000).toISOString(),
          file_size: 15728640,
          record_count: 50000,
          download_url: '/api/exports/export-001/download',
          parameters: { start_date: '2024-01-01', end_date: '2024-01-31' }
        },
        {
          task_id: 'export-002',
          title: '会话分析数据',
          data_type: 'sessions',
          format: 'xlsx',
          status: 'processing',
          progress: 67,
          created_at: new Date(Date.now() - 1800000).toISOString(),
          file_size: 8388608,
          record_count: 25000,
          parameters: { user_type: 'premium' }
        }
      ];
      setExportTasks(mockTasks);
    } catch (error) {
      console.error('获取导出任务失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExportTasks();
  }, []);

  // 预览数据
  const handlePreviewData = async () => {
    setPreviewLoading(true);
    try {
      const response = await behaviorAnalyticsService.previewExportData({
        data_type: exportConfig.data_type,
        filters: exportConfig.filters,
        limit: 10
      });
      setPreviewData(response.data || []);
    } catch (error) {
      console.error('预览数据失败:', error);
      setPreviewData([]);
    } finally {
      setPreviewLoading(false);
    }
  };

  // 创建导出任务
  const handleCreateExport = async () => {
    try {
      const response = await behaviorAnalyticsService.createExportTask(exportConfig);
      
      const newTask: ExportTask = {
        task_id: response.task_id,
        title: exportConfig.title,
        data_type: exportConfig.data_type,
        format: exportConfig.format,
        status: 'pending',
        progress: 0,
        created_at: new Date().toISOString(),
        parameters: exportConfig.filters
      };
      
      setExportTasks(prev => [newTask, ...prev]);
      setShowCreateForm(false);
      setExportConfig({
        data_type: 'events',
        format: 'csv',
        title: '',
        filters: {},
        compression: true,
        split_by_date: false,
        max_records: 100000
      });
      setPreviewData([]);
    } catch (error) {
      console.error('创建导出任务失败:', error);
    }
  };

  // 下载导出文件
  const handleDownload = async (taskId: string) => {
    try {
      await behaviorAnalyticsService.downloadExportFile(taskId);
    } catch (error) {
      console.error('下载文件失败:', error);
    }
  };

  // 取消导出任务
  const handleCancelExport = async (taskId: string) => {
    try {
      // TODO: 调用取消导出API
      setExportTasks(prev => 
        prev.map(task => 
          task.task_id === taskId 
            ? { ...task, status: 'failed' as const, error_message: '用户取消' }
            : task
        )
      );
    } catch (error) {
      console.error('取消导出失败:', error);
    }
  };

  // 删除导出任务
  const handleDeleteExport = async (taskId: string) => {
    try {
      // TODO: 调用删除导出API
      setExportTasks(prev => prev.filter(task => task.task_id !== taskId));
    } catch (error) {
      console.error('删除导出任务失败:', error);
    }
  };

  // 格式化文件大小
  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '未知';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  // 格式化时间
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN');
  };

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'bg-yellow-100 text-yellow-800',
      processing: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800'
    };
    return colors[status as keyof typeof colors] || colors.pending;
  };

  // 获取数据类型图标
  const getDataTypeIcon = (type: string) => {
    const dataType = DATA_TYPES.find(dt => dt.value === type);
    return dataType?.icon || '📄';
  };

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">数据导出工具</h1>
          <p className="text-sm text-gray-600 mt-1">
            导出各类分析数据，支持多种格式和自定义筛选条件
          </p>
        </div>
        <div className="flex space-x-3">
          <Button variant="outline" onClick={fetchExportTasks}>
            🔄 刷新任务
          </Button>
          <Button onClick={() => setShowCreateForm(true)}>
            ➕ 新建导出
          </Button>
        </div>
      </div>

      {/* 统计概览 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{exportTasks.length}</p>
            <p className="text-sm text-gray-600">总导出任务</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {exportTasks.filter(t => t.status === 'completed').length}
            </p>
            <p className="text-sm text-gray-600">已完成</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {exportTasks.filter(t => t.status === 'processing').length}
            </p>
            <p className="text-sm text-gray-600">处理中</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-orange-600">
              {formatFileSize(exportTasks.reduce((sum, task) => sum + (task.file_size || 0), 0))}
            </p>
            <p className="text-sm text-gray-600">总文件大小</p>
          </div>
        </Card>
      </div>

      {/* 创建导出表单 */}
      {showCreateForm && (
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">创建导出任务</h3>
            <Button variant="outline" onClick={() => setShowCreateForm(false)}>
              ✖️ 取消
            </Button>
          </div>

          <div className="space-y-6">
            {/* 基本配置 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">任务名称</label>
                <Input
                  placeholder="输入导出任务名称"
                  value={exportConfig.title}
                  onChange={(e) => setExportConfig(prev => ({ ...prev, title: e.target.value }))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">最大记录数</label>
                <Input
                  type="number"
                  placeholder="100000"
                  value={exportConfig.max_records}
                  onChange={(e) => setExportConfig(prev => ({ ...prev, max_records: Number(e.target.value) }))}
                />
              </div>
            </div>

            {/* 数据类型选择 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">数据类型</label>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {DATA_TYPES.map(type => (
                  <div
                    key={type.value}
                    className={`p-3 border rounded-md cursor-pointer transition-colors ${
                      exportConfig.data_type === type.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setExportConfig(prev => ({ ...prev, data_type: type.value as any }))}
                  >
                    <div className="text-center">
                      <span className="text-xl">{type.icon}</span>
                      <p className="text-sm font-medium mt-1">{type.label}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 导出格式选择 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">导出格式</label>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {EXPORT_FORMATS.map(format => (
                  <div
                    key={format.value}
                    className={`p-3 border rounded-md cursor-pointer transition-colors ${
                      exportConfig.format === format.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setExportConfig(prev => ({ ...prev, format: format.value as any }))}
                  >
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-lg">{format.icon}</span>
                      <span className="font-medium">{format.label}</span>
                    </div>
                    <p className="text-xs text-gray-600">{format.description}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 筛选条件 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">筛选条件</label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">开始日期</label>
                  <Input
                    type="date"
                    value={exportConfig.filters.start_date || ''}
                    onChange={(e) => setExportConfig(prev => ({
                      ...prev,
                      filters: { ...prev.filters, start_date: e.target.value }
                    }))}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">结束日期</label>
                  <Input
                    type="date"
                    value={exportConfig.filters.end_date || ''}
                    onChange={(e) => setExportConfig(prev => ({
                      ...prev,
                      filters: { ...prev.filters, end_date: e.target.value }
                    }))}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">用户ID</label>
                  <Input
                    placeholder="可选，筛选特定用户"
                    value={exportConfig.filters.user_id || ''}
                    onChange={(e) => setExportConfig(prev => ({
                      ...prev,
                      filters: { ...prev.filters, user_id: e.target.value || undefined }
                    }))}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">事件类型</label>
                  <Input
                    placeholder="可选，筛选事件类型"
                    value={exportConfig.filters.event_type || ''}
                    onChange={(e) => setExportConfig(prev => ({
                      ...prev,
                      filters: { ...prev.filters, event_type: e.target.value || undefined }
                    }))}
                  />
                </div>
              </div>
            </div>

            {/* 高级选项 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">高级选项</label>
              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={exportConfig.compression}
                    onChange={(e) => setExportConfig(prev => ({ ...prev, compression: e.target.checked }))}
                  />
                  <span className="text-sm">启用压缩（减小文件大小）</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={exportConfig.split_by_date}
                    onChange={(e) => setExportConfig(prev => ({ ...prev, split_by_date: e.target.checked }))}
                  />
                  <span className="text-sm">按日期分割文件</span>
                </label>
              </div>
            </div>

            {/* 数据预览 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-gray-700">数据预览</label>
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={handlePreviewData}
                  disabled={previewLoading}
                >
                  {previewLoading ? '⏳ 加载中...' : '👁️ 预览数据'}
                </Button>
              </div>
              
              {previewData.length > 0 && (
                <div className="border rounded-md">
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          {Object.keys(previewData[0] || {}).slice(0, 6).map(key => (
                            <th key={key} className="px-3 py-2 text-left font-medium text-gray-700">
                              {key}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewData.slice(0, 5).map((row, index) => (
                          <tr key={index} className="border-t">
                            {Object.values(row).slice(0, 6).map((value: any, cellIndex) => (
                              <td key={cellIndex} className="px-3 py-2 text-gray-600">
                                {typeof value === 'object' ? JSON.stringify(value).substring(0, 30) + '...' : String(value).substring(0, 30)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="px-3 py-2 bg-gray-50 text-xs text-gray-500 border-t">
                    显示前5行数据，实际导出将包含所有匹配记录
                  </div>
                </div>
              )}
            </div>

            <div className="flex space-x-3">
              <Button 
                onClick={handleCreateExport}
                disabled={!exportConfig.title || previewLoading}
              >
                🚀 开始导出
              </Button>
              <Button variant="outline" onClick={handlePreviewData}>
                👁️ 重新预览
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* 导出任务列表 */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">导出任务</h3>
        
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2 text-gray-600">加载中...</p>
          </div>
        ) : (
          <div className="space-y-4">
            {exportTasks.map((task) => (
              <div key={task.task_id} className="p-4 border border-gray-200 rounded-md">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <span className="text-xl">{getDataTypeIcon(task.data_type)}</span>
                      <div>
                        <h4 className="font-medium">{task.title}</h4>
                        <div className="flex items-center space-x-2 mt-1">
                          <Badge className={getStatusColor(task.status)}>
                            {task.status}
                          </Badge>
                          <span className="text-xs text-gray-500">
                            {task.format.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                      <div>
                        <span className="text-gray-500">创建时间:</span>
                        <p>{formatTime(task.created_at)}</p>
                      </div>
                      {task.completed_at && (
                        <div>
                          <span className="text-gray-500">完成时间:</span>
                          <p>{formatTime(task.completed_at)}</p>
                        </div>
                      )}
                      <div>
                        <span className="text-gray-500">记录数:</span>
                        <p>{task.record_count?.toLocaleString() || '-'}</p>
                      </div>
                      <div>
                        <span className="text-gray-500">文件大小:</span>
                        <p>{formatFileSize(task.file_size)}</p>
                      </div>
                    </div>

                    {task.status === 'processing' && (
                      <div className="mb-3">
                        <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                          <span>导出进度</span>
                          <span>{task.progress}%</span>
                        </div>
                        <Progress value={task.progress} max={100} className="h-2" />
                      </div>
                    )}

                    {task.error_message && (
                      <Alert variant="destructive" className="text-sm">
                        导出失败: {task.error_message}
                      </Alert>
                    )}
                  </div>
                  
                  <div className="flex space-x-2 ml-4">
                    {task.status === 'completed' && (
                      <Button 
                        size="sm"
                        onClick={() => handleDownload(task.task_id)}
                      >
                        📥 下载
                      </Button>
                    )}
                    {task.status === 'processing' && (
                      <Button 
                        size="sm"
                        variant="outline"
                        onClick={() => handleCancelExport(task.task_id)}
                      >
                        ❌ 取消
                      </Button>
                    )}
                    <Button 
                      size="sm"
                      variant="danger"
                      onClick={() => handleDeleteExport(task.task_id)}
                    >
                      🗑️ 删除
                    </Button>
                  </div>
                </div>
              </div>
            ))}
            
            {exportTasks.length === 0 && !loading && (
              <div className="text-center py-8 text-gray-500">
                <span className="text-4xl mb-4 block">📦</span>
                <p>暂无导出任务</p>
                <p className="text-sm mt-1">点击上方"新建导出"按钮开始</p>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default DataExportPage;