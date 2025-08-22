import React, { useState, useEffect } from 'react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Badge } from '../../components/ui/Badge';
import { Progress } from '../../components/ui/Progress';
import { Alert } from '../../components/ui/Alert';
import { behaviorAnalyticsService } from '../../services/behaviorAnalyticsService';

interface AnalyticsReport {
  report_id: string;
  title: string;
  description: string;
  report_type: 'pattern_analysis' | 'anomaly_detection' | 'trend_forecast' | 'user_segment' | 'performance';
  status: 'pending' | 'generating' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  progress: number;
  file_size?: number;
  download_url?: string;
  parameters: Record<string, any>;
  insights_count?: number;
  chart_count?: number;
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  type: string;
  icon: string;
  parameters: Array<{
    key: string;
    label: string;
    type: 'string' | 'number' | 'date' | 'select' | 'boolean';
    required: boolean;
    options?: string[];
    default?: any;
  }>;
}

const REPORT_TEMPLATES: ReportTemplate[] = [
  {
    id: 'user_behavior_summary',
    name: '用户行为总结报告',
    description: '生成指定时间范围内的用户行为模式分析报告',
    type: 'pattern_analysis',
    icon: '👥',
    parameters: [
      { key: 'start_date', label: '开始日期', type: 'date', required: true },
      { key: 'end_date', label: '结束日期', type: 'date', required: true },
      { key: 'user_segments', label: '用户分段', type: 'select', required: false, options: ['all', 'new_users', 'active_users', 'premium_users'] },
      { key: 'include_charts', label: '包含图表', type: 'boolean', required: false, default: true }
    ]
  },
  {
    id: 'anomaly_analysis',
    name: '异常行为分析报告',
    description: '分析检测到的异常行为模式和趋势',
    type: 'anomaly_detection',
    icon: '⚠️',
    parameters: [
      { key: 'severity_threshold', label: '严重程度阈值', type: 'select', required: true, options: ['low', 'medium', 'high', 'critical'] },
      { key: 'time_range', label: '时间范围(天)', type: 'number', required: true, default: 7 },
      { key: 'group_by_user', label: '按用户分组', type: 'boolean', required: false, default: false }
    ]
  },
  {
    id: 'trend_forecast',
    name: '趋势预测报告',
    description: '基于历史数据的用户行为趋势预测',
    type: 'trend_forecast',
    icon: '📈',
    parameters: [
      { key: 'forecast_days', label: '预测天数', type: 'number', required: true, default: 30 },
      { key: 'confidence_level', label: '置信度', type: 'select', required: true, options: ['0.80', '0.90', '0.95'], default: '0.90' },
      { key: 'include_seasonality', label: '包含季节性', type: 'boolean', required: false, default: true }
    ]
  },
  {
    id: 'user_segmentation',
    name: '用户分群分析报告',
    description: '基于行为模式的用户分群和特征分析',
    type: 'user_segment',
    icon: '🎯',
    parameters: [
      { key: 'clustering_method', label: '聚类方法', type: 'select', required: true, options: ['kmeans', 'dbscan', 'hierarchical'] },
      { key: 'num_clusters', label: '聚类数量', type: 'number', required: false, default: 5 },
      { key: 'min_events', label: '最小事件数', type: 'number', required: false, default: 10 }
    ]
  },
  {
    id: 'performance_analysis',
    name: '系统性能分析报告',
    description: '分析系统性能指标和用户体验数据',
    type: 'performance',
    icon: '⚡',
    parameters: [
      { key: 'metric_types', label: '指标类型', type: 'select', required: true, options: ['response_time', 'throughput', 'error_rate', 'all'] },
      { key: 'aggregation', label: '聚合方式', type: 'select', required: true, options: ['hourly', 'daily', 'weekly'] },
      { key: 'include_recommendations', label: '包含优化建议', type: 'boolean', required: false, default: true }
    ]
  }
];

export const ReportCenterPage: React.FC = () => {
  const [reports, setReports] = useState<AnalyticsReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate | null>(null);
  const [reportParameters, setReportParameters] = useState<Record<string, any>>({});
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);

  // 获取报告列表
  const fetchReports = async () => {
    setLoading(true);
    try {
      const response = await behaviorAnalyticsService.getReports();
      setReports(response.reports || []);
    } catch (error) {
      console.error('获取报告列表失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  // 格式化时间
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN');
  };

  // 格式化文件大小
  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '未知';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // 获取报告状态颜色
  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'bg-yellow-100 text-yellow-800',
      generating: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800'
    };
    return colors[status as keyof typeof colors] || colors.pending;
  };

  // 获取报告类型图标
  const getTypeIcon = (type: string) => {
    const icons = {
      pattern_analysis: '📊',
      anomaly_detection: '⚠️',
      trend_forecast: '📈',
      user_segment: '🎯',
      performance: '⚡'
    };
    return icons[type as keyof typeof icons] || '📄';
  };

  // 选择模板
  const handleSelectTemplate = (template: ReportTemplate) => {
    setSelectedTemplate(template);
    setShowCreateForm(true);
    
    // 设置默认参数
    const defaults: Record<string, any> = {};
    template.parameters.forEach(param => {
      if (param.default !== undefined) {
        defaults[param.key] = param.default;
      }
    });
    setReportParameters(defaults);
  };

  // 更新参数
  const updateParameter = (key: string, value: any) => {
    setReportParameters(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // 生成报告
  const handleGenerateReport = async () => {
    if (!selectedTemplate) return;

    setGeneratingReport(true);
    try {
      const reportRequest = {
        report_type: selectedTemplate.type,
        title: selectedTemplate.name,
        description: selectedTemplate.description,
        parameters: reportParameters
      };

      const response = await behaviorAnalyticsService.generateReport(reportRequest);
      
      // 添加到报告列表
      const newReport: AnalyticsReport = {
        report_id: response.report_id,
        title: selectedTemplate.name,
        description: selectedTemplate.description,
        report_type: selectedTemplate.type as any,
        status: 'pending',
        created_at: new Date().toISOString(),
        progress: 0,
        parameters: reportParameters
      };
      
      setReports(prev => [newReport, ...prev]);
      setShowCreateForm(false);
      setSelectedTemplate(null);
      setReportParameters({});
    } catch (error) {
      console.error('生成报告失败:', error);
    } finally {
      setGeneratingReport(false);
    }
  };

  // 下载报告
  const handleDownloadReport = async (reportId: string) => {
    try {
      await behaviorAnalyticsService.downloadReport(reportId);
    } catch (error) {
      console.error('下载报告失败:', error);
    }
  };

  // 删除报告
  const handleDeleteReport = async (reportId: string) => {
    try {
      // TODO: 调用删除API
      setReports(prev => prev.filter(report => report.report_id !== reportId));
    } catch (error) {
      console.error('删除报告失败:', error);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">报告生成中心</h1>
          <p className="text-sm text-gray-600 mt-1">
            创建和管理各类行为分析报告，支持自定义参数和多种输出格式
          </p>
        </div>
        <div className="flex space-x-3">
          <Button variant="outline" onClick={fetchReports}>
            🔄 刷新列表
          </Button>
          <Button onClick={() => setShowCreateForm(true)}>
            ➕ 创建报告
          </Button>
        </div>
      </div>

      {/* 统计概览 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{reports.length}</p>
            <p className="text-sm text-gray-600">总报告数</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {reports.filter(r => r.status === 'completed').length}
            </p>
            <p className="text-sm text-gray-600">已完成</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {reports.filter(r => r.status === 'generating').length}
            </p>
            <p className="text-sm text-gray-600">生成中</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-red-600">
              {reports.filter(r => r.status === 'failed').length}
            </p>
            <p className="text-sm text-gray-600">失败</p>
          </div>
        </Card>
      </div>

      {/* 创建报告表单 */}
      {showCreateForm && (
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">创建新报告</h3>
            <Button variant="outline" onClick={() => setShowCreateForm(false)}>
              ✖️ 取消
            </Button>
          </div>

          {!selectedTemplate ? (
            <div>
              <p className="text-gray-600 mb-4">选择报告模板：</p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {REPORT_TEMPLATES.map(template => (
                  <div
                    key={template.id}
                    className="p-4 border border-gray-200 rounded-md hover:border-blue-300 cursor-pointer transition-colors"
                    onClick={() => handleSelectTemplate(template)}
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      <span className="text-2xl">{template.icon}</span>
                      <h4 className="font-medium">{template.name}</h4>
                    </div>
                    <p className="text-sm text-gray-600">{template.description}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <span className="text-2xl">{selectedTemplate.icon}</span>
                <div>
                  <h4 className="font-medium">{selectedTemplate.name}</h4>
                  <p className="text-sm text-gray-600">{selectedTemplate.description}</p>
                </div>
              </div>

              <div className="space-y-4">
                {selectedTemplate.parameters.map(param => (
                  <div key={param.key}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {param.label} {param.required && <span className="text-red-500">*</span>}
                    </label>
                    
                    {param.type === 'string' && (
                      <Input
                        value={reportParameters[param.key] || ''}
                        onChange={(e) => updateParameter(param.key, e.target.value)}
                        required={param.required}
                      />
                    )}
                    
                    {param.type === 'number' && (
                      <Input
                        type="number"
                        value={reportParameters[param.key] || ''}
                        onChange={(e) => updateParameter(param.key, Number(e.target.value))}
                        required={param.required}
                      />
                    )}
                    
                    {param.type === 'date' && (
                      <Input
                        type="date"
                        value={reportParameters[param.key] || ''}
                        onChange={(e) => updateParameter(param.key, e.target.value)}
                        required={param.required}
                      />
                    )}
                    
                    {param.type === 'select' && (
                      <select
                        className="w-full px-3 py-2 border border-gray-300 rounded-md"
                        value={reportParameters[param.key] || ''}
                        onChange={(e) => updateParameter(param.key, e.target.value)}
                        required={param.required}
                      >
                        <option value="">请选择...</option>
                        {param.options?.map(option => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    )}
                    
                    {param.type === 'boolean' && (
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={reportParameters[param.key] || false}
                          onChange={(e) => updateParameter(param.key, e.target.checked)}
                        />
                        <span className="text-sm">启用</span>
                      </label>
                    )}
                  </div>
                ))}
              </div>

              <div className="flex space-x-2 mt-6">
                <Button 
                  onClick={handleGenerateReport}
                  disabled={generatingReport}
                >
                  {generatingReport ? '⏳ 生成中...' : '🚀 生成报告'}
                </Button>
                <Button 
                  variant="outline"
                  onClick={() => setSelectedTemplate(null)}
                >
                  ⬅️ 返回选择
                </Button>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* 报告列表 */}
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
                      <span className="text-xl">{getTypeIcon(report.report_type)}</span>
                      <div>
                        <h4 className="font-medium">{report.title}</h4>
                        <p className="text-sm text-gray-600">{report.description}</p>
                      </div>
                      <Badge className={getStatusColor(report.status)}>
                        {report.status}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">创建时间:</span>
                        <p>{formatTime(report.created_at)}</p>
                      </div>
                      {report.completed_at && (
                        <div>
                          <span className="text-gray-500">完成时间:</span>
                          <p>{formatTime(report.completed_at)}</p>
                        </div>
                      )}
                      <div>
                        <span className="text-gray-500">文件大小:</span>
                        <p>{formatFileSize(report.file_size)}</p>
                      </div>
                      {report.insights_count && (
                        <div>
                          <span className="text-gray-500">洞察数:</span>
                          <p>{report.insights_count} 个</p>
                        </div>
                      )}
                    </div>

                    {report.status === 'generating' && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                          <span>生成进度</span>
                          <span>{report.progress}%</span>
                        </div>
                        <Progress value={report.progress} max={100} className="h-2" />
                      </div>
                    )}
                  </div>
                  
                  <div className="flex space-x-2 ml-4">
                    {report.status === 'completed' && (
                      <Button 
                        size="sm"
                        onClick={() => handleDownloadReport(report.report_id)}
                      >
                        📥 下载
                      </Button>
                    )}
                    <Button 
                      size="sm"
                      variant="outline"
                    >
                      👁️ 预览
                    </Button>
                    <Button 
                      size="sm"
                      variant="danger"
                      onClick={() => handleDeleteReport(report.report_id)}
                    >
                      🗑️ 删除
                    </Button>
                  </div>
                </div>
              </div>
            ))}
            
            {reports.length === 0 && !loading && (
              <div className="text-center py-8 text-gray-500">
                <span className="text-4xl mb-4 block">📄</span>
                <p>暂无报告</p>
                <p className="text-sm mt-1">点击上方"创建报告"按钮开始</p>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default ReportCenterPage;