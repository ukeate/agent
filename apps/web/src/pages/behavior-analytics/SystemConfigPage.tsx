import React, { useState, useEffect } from 'react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Badge } from '../../components/ui/Badge';
import { Alert } from '../../components/ui/Alert';
import { Switch } from '../../components/ui/Switch';
import { behaviorAnalyticsService } from '../../services/behaviorAnalyticsService';

interface SystemConfig {
  // 事件采集配置
  event_collection: {
    buffer_size: number;
    flush_interval: number;
    compression_enabled: boolean;
    quality_monitoring: boolean;
    max_batch_size: number;
    retention_days: number;
  };
  
  // 异常检测配置
  anomaly_detection: {
    enabled: boolean;
    sensitivity: 'low' | 'medium' | 'high';
    detection_methods: string[];
    alert_threshold: number;
    auto_acknowledge: boolean;
  };
  
  // 模式识别配置
  pattern_recognition: {
    min_pattern_length: number;
    max_pattern_length: number;
    clustering_algorithm: 'kmeans' | 'dbscan';
    cluster_count: number;
    update_frequency: number;
  };
  
  // 实时监控配置
  realtime_monitoring: {
    websocket_enabled: boolean;
    connection_timeout: number;
    heartbeat_interval: number;
    max_connections: number;
    rate_limit: number;
  };
  
  // 性能优化配置
  performance: {
    cache_enabled: boolean;
    cache_ttl: number;
    parallel_processing: boolean;
    worker_threads: number;
    memory_limit: number;
  };
  
  // 数据存储配置
  storage: {
    partitioning_enabled: boolean;
    partition_interval: 'daily' | 'weekly' | 'monthly';
    compression_algorithm: 'gzip' | 'lz4' | 'snappy';
    index_optimization: boolean;
    cleanup_enabled: boolean;
  };
}

const DEFAULT_CONFIG: SystemConfig = {
  event_collection: {
    buffer_size: 1000,
    flush_interval: 5000,
    compression_enabled: true,
    quality_monitoring: true,
    max_batch_size: 100,
    retention_days: 90
  },
  anomaly_detection: {
    enabled: true,
    sensitivity: 'medium',
    detection_methods: ['statistical', 'isolation_forest'],
    alert_threshold: 0.8,
    auto_acknowledge: false
  },
  pattern_recognition: {
    min_pattern_length: 3,
    max_pattern_length: 10,
    clustering_algorithm: 'kmeans',
    cluster_count: 5,
    update_frequency: 3600
  },
  realtime_monitoring: {
    websocket_enabled: true,
    connection_timeout: 30000,
    heartbeat_interval: 10000,
    max_connections: 1000,
    rate_limit: 100
  },
  performance: {
    cache_enabled: true,
    cache_ttl: 3600,
    parallel_processing: true,
    worker_threads: 4,
    memory_limit: 2048
  },
  storage: {
    partitioning_enabled: true,
    partition_interval: 'daily',
    compression_algorithm: 'gzip',
    index_optimization: true,
    cleanup_enabled: true
  }
};

export const SystemConfigPage: React.FC = () => {
  const [config, setConfig] = useState<SystemConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [activeTab, setActiveTab] = useState<'event_collection' | 'anomaly_detection' | 'pattern_recognition' | 'realtime_monitoring' | 'performance' | 'storage'>('event_collection');

  // 加载配置
  const loadConfig = async () => {
    setLoading(true);
    try {
      const response = await behaviorAnalyticsService.getSystemConfig();
      setConfig(response.config || DEFAULT_CONFIG);
    } catch (error) {
      console.error('加载配置失败:', error);
      setConfig(DEFAULT_CONFIG);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  // 保存配置
  const saveConfig = async () => {
    setSaving(true);
    try {
      await behaviorAnalyticsService.updateSystemConfig(config);
      setHasChanges(false);
      setLastSaved(new Date().toLocaleString('zh-CN'));
    } catch (error) {
      console.error('保存配置失败:', error);
    } finally {
      setSaving(false);
    }
  };

  // 重置配置
  const resetConfig = () => {
    setConfig(DEFAULT_CONFIG);
    setHasChanges(true);
  };

  // 更新配置项
  const updateConfig = (section: keyof SystemConfig, key: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
    setHasChanges(true);
  };

  // 配置项组件
  const ConfigSection = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="space-y-4">
      <h4 className="font-medium text-gray-900">{title}</h4>
      <div className="space-y-3">
        {children}
      </div>
    </div>
  );

  const ConfigItem = ({ 
    label, 
    description, 
    children 
  }: { 
    label: string; 
    description?: string; 
    children: React.ReactNode 
  }) => (
    <div className="flex items-center justify-between py-2">
      <div className="flex-1">
        <label className="text-sm font-medium text-gray-700">{label}</label>
        {description && (
          <p className="text-xs text-gray-500 mt-1">{description}</p>
        )}
      </div>
      <div className="ml-4">
        {children}
      </div>
    </div>
  );

  const tabs = [
    { id: 'event_collection', label: '事件采集', icon: '📝' },
    { id: 'anomaly_detection', label: '异常检测', icon: '⚠️' },
    { id: 'pattern_recognition', label: '模式识别', icon: '🔍' },
    { id: 'realtime_monitoring', label: '实时监控', icon: '⚡' },
    { id: 'performance', label: '性能优化', icon: '🚀' },
    { id: 'storage', label: '数据存储', icon: '💾' }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">系统配置管理</h1>
          <p className="text-sm text-gray-600 mt-1">
            配置行为分析系统的各项参数和策略
          </p>
        </div>
        <div className="flex items-center space-x-3">
          {lastSaved && (
            <span className="text-sm text-gray-500">
              最后保存: {lastSaved}
            </span>
          )}
          <Button variant="outline" onClick={resetConfig}>
            🔄 重置为默认
          </Button>
          <Button 
            onClick={saveConfig}
            disabled={!hasChanges || saving}
          >
            {saving ? '⏳ 保存中...' : '💾 保存配置'}
          </Button>
        </div>
      </div>

      {/* 更改提醒 */}
      {hasChanges && (
        <Alert variant="warning">
          <p>⚠️ 您有未保存的配置更改。请点击"保存配置"按钮保存更改。</p>
        </Alert>
      )}

      {/* 配置标签页 */}
      <Card className="p-6">
        <div className="flex flex-wrap border-b border-gray-200 mb-6">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium border-b-2 mr-6 ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2 text-gray-600">加载配置中...</p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* 事件采集配置 */}
            {activeTab === 'event_collection' && (
              <ConfigSection title="事件采集配置">
                <ConfigItem 
                  label="缓冲区大小" 
                  description="内存缓冲区可容纳的事件数量"
                >
                  <Input
                    type="number"
                    value={config.event_collection.buffer_size}
                    onChange={(e) => updateConfig('event_collection', 'buffer_size', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="刷新间隔 (ms)" 
                  description="将缓冲区事件写入存储的间隔时间"
                >
                  <Input
                    type="number"
                    value={config.event_collection.flush_interval}
                    onChange={(e) => updateConfig('event_collection', 'flush_interval', Number(e.target.value))}
                    className="w-24"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="数据压缩" 
                  description="启用事件数据压缩以节省存储空间"
                >
                  <Switch
                    checked={config.event_collection.compression_enabled}
                    onChange={(checked) => updateConfig('event_collection', 'compression_enabled', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="质量监控" 
                  description="启用数据质量检查和监控"
                >
                  <Switch
                    checked={config.event_collection.quality_monitoring}
                    onChange={(checked) => updateConfig('event_collection', 'quality_monitoring', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="最大批次大小" 
                  description="单次批量处理的最大事件数"
                >
                  <Input
                    type="number"
                    value={config.event_collection.max_batch_size}
                    onChange={(e) => updateConfig('event_collection', 'max_batch_size', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="数据保留天数" 
                  description="事件数据在系统中保留的天数"
                >
                  <Input
                    type="number"
                    value={config.event_collection.retention_days}
                    onChange={(e) => updateConfig('event_collection', 'retention_days', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
              </ConfigSection>
            )}

            {/* 异常检测配置 */}
            {activeTab === 'anomaly_detection' && (
              <ConfigSection title="异常检测配置">
                <ConfigItem 
                  label="启用异常检测" 
                  description="开启自动异常行为检测功能"
                >
                  <Switch
                    checked={config.anomaly_detection.enabled}
                    onChange={(checked) => updateConfig('anomaly_detection', 'enabled', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="检测敏感度" 
                  description="异常检测的敏感程度设置"
                >
                  <select
                    className="px-3 py-1 border border-gray-300 rounded text-sm"
                    value={config.anomaly_detection.sensitivity}
                    onChange={(e) => updateConfig('anomaly_detection', 'sensitivity', e.target.value)}
                  >
                    <option value="low">低</option>
                    <option value="medium">中</option>
                    <option value="high">高</option>
                  </select>
                </ConfigItem>
                
                <ConfigItem 
                  label="告警阈值" 
                  description="触发告警的异常评分阈值 (0-1)"
                >
                  <Input
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    value={config.anomaly_detection.alert_threshold}
                    onChange={(e) => updateConfig('anomaly_detection', 'alert_threshold', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="自动确认告警" 
                  description="自动确认低级别的异常告警"
                >
                  <Switch
                    checked={config.anomaly_detection.auto_acknowledge}
                    onChange={(checked) => updateConfig('anomaly_detection', 'auto_acknowledge', checked)}
                  />
                </ConfigItem>
              </ConfigSection>
            )}

            {/* 模式识别配置 */}
            {activeTab === 'pattern_recognition' && (
              <ConfigSection title="模式识别配置">
                <ConfigItem 
                  label="最小模式长度" 
                  description="识别序列模式的最小事件数量"
                >
                  <Input
                    type="number"
                    min="2"
                    value={config.pattern_recognition.min_pattern_length}
                    onChange={(e) => updateConfig('pattern_recognition', 'min_pattern_length', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="最大模式长度" 
                  description="识别序列模式的最大事件数量"
                >
                  <Input
                    type="number"
                    value={config.pattern_recognition.max_pattern_length}
                    onChange={(e) => updateConfig('pattern_recognition', 'max_pattern_length', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="聚类算法" 
                  description="用户行为聚类的算法选择"
                >
                  <select
                    className="px-3 py-1 border border-gray-300 rounded text-sm"
                    value={config.pattern_recognition.clustering_algorithm}
                    onChange={(e) => updateConfig('pattern_recognition', 'clustering_algorithm', e.target.value)}
                  >
                    <option value="kmeans">K-Means</option>
                    <option value="dbscan">DBSCAN</option>
                  </select>
                </ConfigItem>
                
                <ConfigItem 
                  label="聚类数量" 
                  description="K-Means算法的聚类数量"
                >
                  <Input
                    type="number"
                    min="2"
                    value={config.pattern_recognition.cluster_count}
                    onChange={(e) => updateConfig('pattern_recognition', 'cluster_count', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="更新频率 (秒)" 
                  description="模式识别模型更新的时间间隔"
                >
                  <Input
                    type="number"
                    value={config.pattern_recognition.update_frequency}
                    onChange={(e) => updateConfig('pattern_recognition', 'update_frequency', Number(e.target.value))}
                    className="w-24"
                  />
                </ConfigItem>
              </ConfigSection>
            )}

            {/* 实时监控配置 */}
            {activeTab === 'realtime_monitoring' && (
              <ConfigSection title="实时监控配置">
                <ConfigItem 
                  label="启用WebSocket" 
                  description="开启WebSocket实时数据推送功能"
                >
                  <Switch
                    checked={config.realtime_monitoring.websocket_enabled}
                    onChange={(checked) => updateConfig('realtime_monitoring', 'websocket_enabled', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="连接超时 (ms)" 
                  description="WebSocket连接的超时时间"
                >
                  <Input
                    type="number"
                    value={config.realtime_monitoring.connection_timeout}
                    onChange={(e) => updateConfig('realtime_monitoring', 'connection_timeout', Number(e.target.value))}
                    className="w-24"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="心跳间隔 (ms)" 
                  description="WebSocket心跳检测的时间间隔"
                >
                  <Input
                    type="number"
                    value={config.realtime_monitoring.heartbeat_interval}
                    onChange={(e) => updateConfig('realtime_monitoring', 'heartbeat_interval', Number(e.target.value))}
                    className="w-24"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="最大连接数" 
                  description="同时支持的WebSocket连接数量上限"
                >
                  <Input
                    type="number"
                    value={config.realtime_monitoring.max_connections}
                    onChange={(e) => updateConfig('realtime_monitoring', 'max_connections', Number(e.target.value))}
                    className="w-24"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="速率限制 (req/min)" 
                  description="每分钟允许的最大请求数"
                >
                  <Input
                    type="number"
                    value={config.realtime_monitoring.rate_limit}
                    onChange={(e) => updateConfig('realtime_monitoring', 'rate_limit', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
              </ConfigSection>
            )}

            {/* 性能优化配置 */}
            {activeTab === 'performance' && (
              <ConfigSection title="性能优化配置">
                <ConfigItem 
                  label="启用缓存" 
                  description="开启内存缓存提升查询性能"
                >
                  <Switch
                    checked={config.performance.cache_enabled}
                    onChange={(checked) => updateConfig('performance', 'cache_enabled', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="缓存TTL (秒)" 
                  description="缓存数据的生存时间"
                >
                  <Input
                    type="number"
                    value={config.performance.cache_ttl}
                    onChange={(e) => updateConfig('performance', 'cache_ttl', Number(e.target.value))}
                    className="w-24"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="并行处理" 
                  description="启用多线程并行数据处理"
                >
                  <Switch
                    checked={config.performance.parallel_processing}
                    onChange={(checked) => updateConfig('performance', 'parallel_processing', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="工作线程数" 
                  description="并行处理的工作线程数量"
                >
                  <Input
                    type="number"
                    min="1"
                    max="16"
                    value={config.performance.worker_threads}
                    onChange={(e) => updateConfig('performance', 'worker_threads', Number(e.target.value))}
                    className="w-20"
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="内存限制 (MB)" 
                  description="系统使用的最大内存量"
                >
                  <Input
                    type="number"
                    value={config.performance.memory_limit}
                    onChange={(e) => updateConfig('performance', 'memory_limit', Number(e.target.value))}
                    className="w-24"
                  />
                </ConfigItem>
              </ConfigSection>
            )}

            {/* 数据存储配置 */}
            {activeTab === 'storage' && (
              <ConfigSection title="数据存储配置">
                <ConfigItem 
                  label="启用分区" 
                  description="按时间自动分区表以优化查询性能"
                >
                  <Switch
                    checked={config.storage.partitioning_enabled}
                    onChange={(checked) => updateConfig('storage', 'partitioning_enabled', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="分区间隔" 
                  description="数据表分区的时间间隔"
                >
                  <select
                    className="px-3 py-1 border border-gray-300 rounded text-sm"
                    value={config.storage.partition_interval}
                    onChange={(e) => updateConfig('storage', 'partition_interval', e.target.value)}
                  >
                    <option value="daily">每日</option>
                    <option value="weekly">每周</option>
                    <option value="monthly">每月</option>
                  </select>
                </ConfigItem>
                
                <ConfigItem 
                  label="压缩算法" 
                  description="数据压缩算法选择"
                >
                  <select
                    className="px-3 py-1 border border-gray-300 rounded text-sm"
                    value={config.storage.compression_algorithm}
                    onChange={(e) => updateConfig('storage', 'compression_algorithm', e.target.value)}
                  >
                    <option value="gzip">Gzip</option>
                    <option value="lz4">LZ4</option>
                    <option value="snappy">Snappy</option>
                  </select>
                </ConfigItem>
                
                <ConfigItem 
                  label="索引优化" 
                  description="自动优化数据库索引以提升查询性能"
                >
                  <Switch
                    checked={config.storage.index_optimization}
                    onChange={(checked) => updateConfig('storage', 'index_optimization', checked)}
                  />
                </ConfigItem>
                
                <ConfigItem 
                  label="自动清理" 
                  description="自动清理过期数据和日志文件"
                >
                  <Switch
                    checked={config.storage.cleanup_enabled}
                    onChange={(checked) => updateConfig('storage', 'cleanup_enabled', checked)}
                  />
                </ConfigItem>
              </ConfigSection>
            )}
          </div>
        )}
      </Card>

      {/* 配置预览 */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">当前配置概览</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {tabs.map(tab => (
            <div key={tab.id} className="p-3 bg-gray-50 rounded-md">
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-lg">{tab.icon}</span>
                <span className="font-medium text-sm">{tab.label}</span>
              </div>
              <div className="text-xs text-gray-600">
                {Object.keys(config[tab.id as keyof SystemConfig]).length} 项配置
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default SystemConfigPage;