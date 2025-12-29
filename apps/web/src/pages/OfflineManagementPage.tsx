import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/Select';
import { Input } from '../components/ui/Input';
import { 
import { logger } from '../utils/logger'
  offlineService,
  type OfflineStatusResponse,
  type OfflineOperation,
  type OfflineConflict
} from '../services/offlineService';

const OfflineManagementPage: React.FC = () => {
  const [status, setStatus] = useState<OfflineStatusResponse | null>(null);
  const [operations, setOperations] = useState<OfflineOperation[]>([]);
  const [conflicts, setConflicts] = useState<OfflineConflict[]>([]);
  const [statistics, setStatistics] = useState<any>(null);
  const [networkStatus, setNetworkStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('control');

  // 分页参数
  const [currentPage, setCurrentPage] = useState(0);
  const [itemsPerPage] = useState(20);

  // 清理参数
  const [cleanupDays, setCleanupDays] = useState(30);

  // 同步参数
  const [batchSize, setBatchSize] = useState(100);

  useEffect(() => {
    loadData();
    
    // 定期更新状态
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    await Promise.all([
      loadStatus(),
      loadOperations(),
      loadConflicts(),
      loadStatistics(),
      loadNetworkStatus()
    ]);
  };

  const loadStatus = async () => {
    try {
      const statusData = await offlineService.getOfflineStatus();
      setStatus(statusData);
    } catch (err) {
      logger.warn('加载状态失败:', err);
    }
  };

  const loadOperations = async () => {
    try {
      const operationsData = await offlineService.getOperations(itemsPerPage, currentPage * itemsPerPage);
      setOperations(operationsData);
    } catch (err) {
      logger.warn('加载操作记录失败:', err);
    }
  };

  const loadConflicts = async () => {
    try {
      const conflictsData = await offlineService.getConflicts();
      setConflicts(conflictsData);
    } catch (err) {
      logger.warn('加载冲突记录失败:', err);
    }
  };

  const loadStatistics = async () => {
    try {
      const statsData = await offlineService.getStatistics();
      setStatistics(statsData);
    } catch (err) {
      logger.warn('加载统计信息失败:', err);
    }
  };

  const loadNetworkStatus = async () => {
    try {
      const networkData = await offlineService.getNetworkStatus();
      setNetworkStatus(networkData);
    } catch (err) {
      logger.warn('加载网络状态失败:', err);
    }
  };

  const handleSetMode = async (mode: string) => {
    try {
      setLoading(true);
      setError(null);
      
      await offlineService.setOfflineMode(mode);
      setSuccess(`离线模式已设置为: ${mode}`);
      
      await loadStatus();
    } catch (err) {
      setError('设置模式失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleSync = async (force: boolean = false) => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await offlineService.manualSync({
        force,
        batch_size: batchSize
      });
      
      setSuccess(force ? '强制同步已开始' : '后台同步已启动');
      logger.log('同步结果:', result);
      
      // 刷新数据
      setTimeout(() => loadData(), 2000);
    } catch (err) {
      setError('同步失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleResolveConflict = async (conflict: OfflineConflict, strategy: string) => {
    try {
      setLoading(true);
      setError(null);

      let resolvedData;
      switch (strategy) {
        case 'client_wins':
          resolvedData = conflict.local_data;
          break;
        case 'server_wins':
          resolvedData = conflict.remote_data;
          break;
        case 'merge':
          // 简单合并策略
          resolvedData = { ...conflict.remote_data, ...conflict.local_data };
          break;
      }

      await offlineService.resolveConflict({
        conflict_id: conflict.id,
        resolution_strategy: strategy,
        resolved_data: resolvedData
      });

      setSuccess('冲突已解决');
      await loadConflicts();
    } catch (err) {
      setError('解决冲突失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleCleanup = async () => {
    if (!confirm(`确定要清理${cleanupDays}天前的数据吗？`)) return;

    try {
      setLoading(true);
      setError(null);

      const result = await offlineService.cleanupOldData(cleanupDays);
      setSuccess(`清理完成: 操作记录${result.cleaned_operations}条，冲突记录${result.cleaned_conflicts}条`);
      
      await loadData();
    } catch (err) {
      setError('清理数据失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const getModeColor = (mode: string) => {
    switch (mode) {
      case 'online': return 'bg-green-100 text-green-800';
      case 'offline': return 'bg-red-100 text-red-800';
      case 'auto': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getNetworkStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'connected': return 'bg-green-100 text-green-800';
      case 'disconnected': return 'bg-red-100 text-red-800';
      case 'weak': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN');
  };

  const getConnectionQualityText = (quality: number) => {
    if (quality >= 0.8) return '优秀';
    if (quality >= 0.6) return '良好';
    if (quality >= 0.4) return '一般';
    if (quality >= 0.2) return '较差';
    return '很差';
  };

  return (
    <div className="container mx-auto px-4 py-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">离线管理</h1>
        <Button onClick={loadData} disabled={loading}>
          刷新数据
        </Button>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="border-green-200 bg-green-50">
          <AlertDescription className="text-green-800">{success}</AlertDescription>
        </Alert>
      )}

      {/* 状态概览 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>离线状态</CardTitle>
          </CardHeader>
          <CardContent>
            {status ? (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>模式:</span>
                  <Badge className={getModeColor(status.mode)}>
                    {status.mode}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>网络:</span>
                  <Badge className={getNetworkStatusColor(status.network_status)}>
                    {status.network_status}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>连接质量:</span>
                  <span>{getConnectionQualityText(status.connection_quality)}</span>
                </div>
                <div className="flex justify-between">
                  <span>待同步:</span>
                  <span className={status.pending_operations > 0 ? 'text-orange-600 font-semibold' : ''}>
                    {status.pending_operations}
                  </span>
                </div>
                {status.sync_in_progress && (
                  <div className="text-blue-600 text-sm">同步进行中...</div>
                )}
              </div>
            ) : (
              <p>加载中...</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>冲突状态</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>未解决冲突:</span>
                <span className={conflicts.length > 0 ? 'text-red-600 font-semibold' : 'text-green-600'}>
                  {conflicts.length}
                </span>
              </div>
              <div className="flex justify-between">
                <span>状态:</span>
                <Badge className={conflicts.length > 0 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}>
                  {conflicts.length > 0 ? '需要处理' : '正常'}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>网络状态</CardTitle>
          </CardHeader>
          <CardContent>
            {networkStatus ? (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>延迟:</span>
                  <span>{networkStatus.latency || 'N/A'}ms</span>
                </div>
                <div className="flex justify-between">
                  <span>带宽:</span>
                  <span>{networkStatus.bandwidth || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>稳定性:</span>
                  <span>{networkStatus.stability || 'N/A'}%</span>
                </div>
              </div>
            ) : (
              <p>加载中...</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>统计信息</CardTitle>
          </CardHeader>
          <CardContent>
            {statistics ? (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>总操作:</span>
                  <span>{statistics.total_operations || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>已同步:</span>
                  <span>{statistics.synced_operations || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>成功率:</span>
                  <span>{statistics.success_rate || 0}%</span>
                </div>
                <div className="flex justify-between">
                  <span>平均延迟:</span>
                  <span>{statistics.avg_sync_time || 0}ms</span>
                </div>
              </div>
            ) : (
              <p>加载中...</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="control">控制面板</TabsTrigger>
          <TabsTrigger value="conflicts">冲突管理</TabsTrigger>
          <TabsTrigger value="operations">操作记录</TabsTrigger>
          <TabsTrigger value="maintenance">维护</TabsTrigger>
        </TabsList>

        {/* 控制面板 */}
        <TabsContent value="control">
          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>模式控制</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex space-x-4">
                  <Button 
                    onClick={() => handleSetMode('online')}
                    disabled={loading}
                    className={status?.mode === 'online' ? 'bg-green-600' : ''}
                  >
                    在线模式
                  </Button>
                  <Button 
                    onClick={() => handleSetMode('offline')}
                    disabled={loading}
                    className={status?.mode === 'offline' ? 'bg-red-600' : ''}
                  >
                    离线模式
                  </Button>
                  <Button 
                    onClick={() => handleSetMode('auto')}
                    disabled={loading}
                    className={status?.mode === 'auto' ? 'bg-blue-600' : ''}
                  >
                    自动模式
                  </Button>
                </div>
                <div className="text-sm text-gray-600">
                  当前模式: <Badge className={getModeColor(status?.mode || '')}>{status?.mode || 'unknown'}</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>同步控制</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-4">
                  <span>批处理大小:</span>
                  <Input 
                    type="number" 
                    name="batchSize"
                    value={batchSize} 
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    className="w-32"
                    min="1"
                    max="1000"
                  />
                </div>
                <div className="flex space-x-4">
                  <Button 
                    onClick={() => handleSync(false)}
                    disabled={loading}
                  >
                    后台同步
                  </Button>
                  <Button 
                    onClick={() => handleSync(true)}
                    disabled={loading}
                    variant="outline"
                  >
                    强制同步
                  </Button>
                </div>
                <div className="text-sm text-gray-600">
                  待同步操作: {status?.pending_operations || 0} 条
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* 冲突管理 */}
        <TabsContent value="conflicts">
          <Card>
            <CardHeader>
              <CardTitle>冲突管理 ({conflicts.length})</CardTitle>
            </CardHeader>
            <CardContent>
              {conflicts.length > 0 ? (
                <div className="space-y-4">
                  {conflicts.map((conflict) => (
                    <Card key={conflict.id} className="border-l-4 border-l-red-500">
                      <CardContent className="p-4">
                        <div className="space-y-3">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="font-semibold">
                                {conflict.table_name} - {conflict.object_id}
                              </h4>
                              <p className="text-sm text-gray-600">
                                冲突类型: {conflict.conflict_type}
                              </p>
                              <p className="text-sm text-gray-600">
                                创建时间: {formatDate(conflict.created_at)}
                              </p>
                            </div>
                            <Badge variant="destructive">
                              冲突
                            </Badge>
                          </div>

                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <h5 className="font-medium text-sm">本地数据:</h5>
                              <pre className="text-xs bg-gray-100 p-2 rounded mt-1 overflow-auto max-h-32">
                                {JSON.stringify(conflict.local_data, null, 2)}
                              </pre>
                            </div>
                            <div>
                              <h5 className="font-medium text-sm">远程数据:</h5>
                              <pre className="text-xs bg-gray-100 p-2 rounded mt-1 overflow-auto max-h-32">
                                {JSON.stringify(conflict.remote_data, null, 2)}
                              </pre>
                            </div>
                          </div>

                          <div className="flex space-x-2">
                            <Button
                              size="sm"
                              onClick={() => handleResolveConflict(conflict, 'client_wins')}
                              disabled={loading}
                            >
                              使用本地数据
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleResolveConflict(conflict, 'server_wins')}
                              disabled={loading}
                            >
                              使用远程数据
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => handleResolveConflict(conflict, 'merge')}
                              disabled={loading}
                            >
                              合并数据
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  没有未解决的冲突
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* 操作记录 */}
        <TabsContent value="operations">
          <Card>
            <CardHeader>
              <CardTitle>操作记录</CardTitle>
            </CardHeader>
            <CardContent>
              {operations.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse border border-gray-300">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="border border-gray-300 px-4 py-2 text-left">操作类型</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">表名</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">对象ID</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">时间</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">同步状态</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">重试次数</th>
                      </tr>
                    </thead>
                    <tbody>
                      {operations.map((operation) => (
                        <tr key={operation.id}>
                          <td className="border border-gray-300 px-4 py-2">
                            <Badge variant={operation.operation_type === 'DELETE' ? 'destructive' : 'default'}>
                              {operation.operation_type}
                            </Badge>
                          </td>
                          <td className="border border-gray-300 px-4 py-2">
                            {operation.table_name}
                          </td>
                          <td className="border border-gray-300 px-4 py-2 font-mono text-sm">
                            {operation.object_id}
                          </td>
                          <td className="border border-gray-300 px-4 py-2 text-sm">
                            {formatDate(operation.timestamp)}
                          </td>
                          <td className="border border-gray-300 px-4 py-2">
                            <Badge className={operation.is_synced ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'}>
                              {operation.is_synced ? '已同步' : '未同步'}
                            </Badge>
                          </td>
                          <td className="border border-gray-300 px-4 py-2">
                            {operation.retry_count > 0 ? (
                              <span className="text-orange-600">{operation.retry_count}</span>
                            ) : (
                              operation.retry_count
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  没有操作记录
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* 维护 */}
        <TabsContent value="maintenance">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>数据清理</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-4">
                  <span>清理天数:</span>
                  <Input 
                    type="number" 
                    name="cleanupDays"
                    value={cleanupDays}
                    onChange={(e) => setCleanupDays(parseInt(e.target.value))}
                    className="w-32"
                    min="1"
                    max="365"
                  />
                  <span>天前的数据</span>
                </div>
                <Button 
                  onClick={handleCleanup}
                  disabled={loading}
                  variant="outline"
                >
                  清理旧数据
                </Button>
                <div className="text-sm text-gray-600">
                  清理操作将删除指定天数前的已同步操作记录和已解决的冲突记录
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>系统信息</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <h4 className="font-medium mb-2">存储使用情况</h4>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span>操作记录:</span>
                        <span>{statistics?.total_operations || 0} 条</span>
                      </div>
                      <div className="flex justify-between">
                        <span>冲突记录:</span>
                        <span>{conflicts.length} 条</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">同步统计</h4>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span>成功率:</span>
                        <span>{statistics?.success_rate || 0}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>平均延迟:</span>
                        <span>{statistics?.avg_sync_time || 0}ms</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default OfflineManagementPage;
