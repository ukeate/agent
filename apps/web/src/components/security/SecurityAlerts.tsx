// 安全告警管理组件

import React, { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Alert } from '../ui/alert';
import { Input } from '../ui/input';
import { securityApi } from '../../services/securityApi';

interface SecurityAlert {
  id: string;
  alert_type: 'rate_limit' | 'suspicious_request' | 'unauthorized_tool' | 'data_breach' | 'authentication_failed';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affected_resource: string;
  source_ip: string;
  user_id?: string;
  timestamp: string;
  status: 'active' | 'investigating' | 'resolved' | 'false_positive';
  auto_blocked: boolean;
  action_taken: string[];
  resolution_time?: string;
}

export const SecurityAlerts: React.FC = () => {
  const [alerts, setAlerts] = useState<SecurityAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedAlert, setSelectedAlert] = useState<SecurityAlert | null>(null);

  useEffect(() => {
    loadAlerts();
    const interval = setInterval(loadAlerts, 30000); // 每30秒刷新
    return () => clearInterval(interval);
  }, []);

  const loadAlerts = async () => {
    try {
      const data = await securityApi.getSecurityAlerts();
      setAlerts(data);
      setError(null);
    } catch (err) {
      setError('加载安全告警失败');
      console.error('Error loading alerts:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateStatus = async (alertId: string, newStatus: string) => {
    try {
      await securityApi.updateAlertStatus(alertId, newStatus);
      await loadAlerts();
      setSelectedAlert(null);
    } catch (err) {
      console.error('Error updating alert status:', err);
      setError('更新告警状态失败');
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500 text-white';
      case 'high': return 'bg-orange-500 text-white';
      case 'medium': return 'bg-yellow-500 text-white';
      case 'low': return 'bg-blue-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-red-100 text-red-800';
      case 'investigating': return 'bg-yellow-100 text-yellow-800';
      case 'resolved': return 'bg-green-100 text-green-800';
      case 'false_positive': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getAlertTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      'rate_limit': '频率限制',
      'suspicious_request': '可疑请求',
      'unauthorized_tool': '未授权工具调用',
      'data_breach': '数据泄露',
      'authentication_failed': '认证失败'
    };
    return labels[type] || type;
  };

  const filteredAlerts = alerts.filter(alert => {
    const matchesStatus = filterStatus === 'all' || alert.status === filterStatus;
    const matchesSeverity = filterSeverity === 'all' || alert.severity === filterSeverity;
    const matchesSearch = searchTerm === '' || 
      alert.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      alert.affected_resource.toLowerCase().includes(searchTerm.toLowerCase()) ||
      alert.source_ip.includes(searchTerm);
    
    return matchesStatus && matchesSeverity && matchesSearch;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">安全告警管理</h1>
        <div className="flex space-x-2">
          <Badge variant="secondary">
            总计: {alerts.length}
          </Badge>
          <Badge className="bg-red-100 text-red-800">
            活跃: {alerts.filter(a => a.status === 'active').length}
          </Badge>
          <Button onClick={loadAlerts} variant="outline">
            刷新
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <p>{error}</p>
        </Alert>
      )}

      {/* 过滤器 */}
      <Card className="p-4">
        <div className="flex space-x-4">
          <Input
            className="flex-1"
            placeholder="搜索告警..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <select
            className="px-3 py-2 border rounded-md"
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
          >
            <option value="all">所有状态</option>
            <option value="active">活跃</option>
            <option value="investigating">调查中</option>
            <option value="resolved">已解决</option>
            <option value="false_positive">误报</option>
          </select>
          <select
            className="px-3 py-2 border rounded-md"
            value={filterSeverity}
            onChange={(e) => setFilterSeverity(e.target.value)}
          >
            <option value="all">所有级别</option>
            <option value="critical">严重</option>
            <option value="high">高</option>
            <option value="medium">中</option>
            <option value="low">低</option>
          </select>
        </div>
      </Card>

      {/* 告警列表 */}
      <div className="space-y-4">
        {filteredAlerts.length === 0 ? (
          <Card className="p-6 text-center text-gray-500">
            没有符合条件的安全告警
          </Card>
        ) : (
          filteredAlerts.map((alert) => (
            <Card 
              key={alert.id} 
              className={`p-6 cursor-pointer hover:shadow-lg transition-shadow ${
                alert.status === 'active' ? 'border-red-500' : ''
              }`}
              onClick={() => setSelectedAlert(alert)}
            >
              <div className="flex justify-between items-start">
                <div className="space-y-2 flex-1">
                  <div className="flex items-center space-x-2">
                    <Badge className={getSeverityColor(alert.severity)}>
                      {alert.severity.toUpperCase()}
                    </Badge>
                    <Badge className={getStatusColor(alert.status)}>
                      {alert.status}
                    </Badge>
                    <Badge variant="outline">
                      {getAlertTypeLabel(alert.alert_type)}
                    </Badge>
                    {alert.auto_blocked && (
                      <Badge variant="destructive">自动阻止</Badge>
                    )}
                  </div>
                  
                  <h3 className="font-semibold">{alert.description}</h3>
                  
                  <div className="text-sm text-gray-600 space-y-1">
                    <p>受影响资源: {alert.affected_resource}</p>
                    <p>来源IP: {alert.source_ip}</p>
                    {alert.user_id && <p>用户ID: {alert.user_id}</p>}
                    <p>时间: {new Date(alert.timestamp).toLocaleString()}</p>
                    {alert.resolution_time && (
                      <p>解决时间: {new Date(alert.resolution_time).toLocaleString()}</p>
                    )}
                  </div>
                  
                  {alert.action_taken.length > 0 && (
                    <div className="text-sm">
                      <p className="font-medium">已采取的行动:</p>
                      <ul className="list-disc list-inside text-gray-600">
                        {alert.action_taken.map((action, idx) => (
                          <li key={idx}>{action}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                
                {alert.status === 'active' && (
                  <div className="flex flex-col space-y-2 ml-4">
                    <Button
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleUpdateStatus(alert.id, 'investigating');
                      }}
                    >
                      开始调查
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleUpdateStatus(alert.id, 'false_positive');
                      }}
                    >
                      标记误报
                    </Button>
                  </div>
                )}
                
                {alert.status === 'investigating' && (
                  <Button
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleUpdateStatus(alert.id, 'resolved');
                    }}
                  >
                    标记已解决
                  </Button>
                )}
              </div>
            </Card>
          ))
        )}
      </div>

      {/* 告警详情对话框 */}
      {selectedAlert && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setSelectedAlert(null)}
        >
          <Card 
            className="max-w-2xl w-full m-4 p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="space-y-4">
              <div className="flex justify-between items-start">
                <h2 className="text-xl font-bold">告警详情</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedAlert(null)}
                >
                  ✕
                </Button>
              </div>
              
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">告警ID</p>
                    <p className="font-mono">{selectedAlert.id}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">类型</p>
                    <p>{getAlertTypeLabel(selectedAlert.alert_type)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">严重级别</p>
                    <Badge className={getSeverityColor(selectedAlert.severity)}>
                      {selectedAlert.severity}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">状态</p>
                    <Badge className={getStatusColor(selectedAlert.status)}>
                      {selectedAlert.status}
                    </Badge>
                  </div>
                </div>
                
                <div className="border-t pt-3">
                  <p className="text-sm text-gray-500">描述</p>
                  <p>{selectedAlert.description}</p>
                </div>
                
                <div className="border-t pt-3">
                  <p className="text-sm text-gray-500">详细信息</p>
                  <div className="space-y-1 text-sm">
                    <p>受影响资源: {selectedAlert.affected_resource}</p>
                    <p>来源IP: {selectedAlert.source_ip}</p>
                    {selectedAlert.user_id && <p>用户ID: {selectedAlert.user_id}</p>}
                    <p>发生时间: {new Date(selectedAlert.timestamp).toLocaleString()}</p>
                    {selectedAlert.resolution_time && (
                      <p>解决时间: {new Date(selectedAlert.resolution_time).toLocaleString()}</p>
                    )}
                    <p>自动阻止: {selectedAlert.auto_blocked ? '是' : '否'}</p>
                  </div>
                </div>
                
                {selectedAlert.action_taken.length > 0 && (
                  <div className="border-t pt-3">
                    <p className="text-sm text-gray-500">已采取的行动</p>
                    <ul className="list-disc list-inside text-sm">
                      {selectedAlert.action_taken.map((action, idx) => (
                        <li key={idx}>{action}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              
              <div className="flex justify-end space-x-2 pt-4 border-t">
                {selectedAlert.status === 'active' && (
                  <>
                    <Button
                      onClick={() => handleUpdateStatus(selectedAlert.id, 'investigating')}
                    >
                      开始调查
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => handleUpdateStatus(selectedAlert.id, 'false_positive')}
                    >
                      标记误报
                    </Button>
                  </>
                )}
                {selectedAlert.status === 'investigating' && (
                  <Button
                    onClick={() => handleUpdateStatus(selectedAlert.id, 'resolved')}
                  >
                    标记已解决
                  </Button>
                )}
                <Button
                  variant="ghost"
                  onClick={() => setSelectedAlert(null)}
                >
                  关闭
                </Button>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};