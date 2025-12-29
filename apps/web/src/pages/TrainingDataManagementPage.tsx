import React, { useState, useEffect } from 'react';
import { Alert, AlertDescription } from '../components/ui/alert';
import authService, { User } from '../services/authService';
import trainingDataService, { 
  DataSourceResponse, 
  DataSourceCreate, 
  DataRecord, 
  AnnotationTaskCreate,
  AnnotationTask, 
  DataVersionCreate,
  DataVersion, 
  CollectionStatistics,
  AnnotationProgress,
  VersionStatistics
} from '../services/trainingDataService';

const TrainingDataManagementPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'sources' | 'records' | 'annotation' | 'versions' | 'statistics'>('overview');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 状态数据
  const [dataSources, setDataSources] = useState<DataSourceResponse[]>([]);
  const [dataRecords, setDataRecords] = useState<DataRecord[]>([]);
  const [annotationTasks, setAnnotationTasks] = useState<AnnotationTask[]>([]);
  const [dataVersions, setDataVersions] = useState<DataVersion[]>([]);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [collectionStats, setCollectionStats] = useState<CollectionStatistics | null>(null);
  const [versionStats, setVersionStats] = useState<VersionStatistics | null>(null);
  const [queueStatus, setQueueStatus] = useState<any>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);

  // 表单状态
  const [newDataSource, setNewDataSource] = useState<DataSourceCreate>({
    source_id: '',
    source_type: 'api',
    name: '',
    description: '',
    config: {},
  });
  const [newAnnotationTask, setNewAnnotationTask] = useState<AnnotationTaskCreate>({
    name: '',
    description: '',
    task_type: '',
    data_records: [] as string[],
    annotation_schema: {},
    guidelines: '',
    assignees: [] as string[],
    created_by: '',
    deadline: undefined,
  });
  const [newDataVersion, setNewDataVersion] = useState<DataVersionCreate>({
    dataset_name: '',
    version_number: '',
    description: '',
    metadata: {},
  });

  useEffect(() => {
    loadCurrentUser();
    loadOverviewData();
  }, []);

  const loadCurrentUser = async () => {
    if (!authService.getToken()) return;
    try {
      const user = await authService.getCurrentUser();
      setCurrentUser(user);
      setNewAnnotationTask(prev => ({ ...prev, created_by: user.username }));
    } catch {
      setCurrentUser(null);
    }
  };

  const loadOverviewData = async () => {
    setLoading(true);
    try {
      const [sources, records, tasks, stats, versionStats, queue] = await Promise.all([
        trainingDataService.listDataSources().catch(() => []),
        trainingDataService.getDataRecords({ limit: 10 }).catch(() => ({ records: [], count: 0, offset: 0, limit: 10 })),
        trainingDataService.listAnnotationTasks({ limit: 10 }).catch(() => ({ tasks: [], count: 0, offset: 0, limit: 10 })),
        trainingDataService.getCollectionStatistics().catch(() => null),
        trainingDataService.getVersionStatistics().catch(() => null),
        trainingDataService.getQueueStatus().catch(() => null)
      ]);

      setDataSources(sources);
      setDataRecords(records.records);
      setAnnotationTasks(tasks.tasks);
      setCollectionStats(stats);
      setVersionStats(versionStats);
      setQueueStatus(queue);
      setError(null);
    } catch (error: any) {
      setError(`加载数据失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateDataSource = async () => {
    setLoading(true);
    try {
      if (!authService.getToken()) throw new Error('未登录，请先登录');
      await trainingDataService.createDataSource(newDataSource);
      await loadOverviewData();
      setNewDataSource({
        source_id: '',
        source_type: 'api',
        name: '',
        description: '',
        config: {},
      });
      setError(null);
    } catch (error: any) {
      setError(`创建数据源失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAnnotationTask = async () => {
    setLoading(true);
    try {
      if (!authService.getToken()) throw new Error('未登录，请先登录');
      if (!newAnnotationTask.data_records.length) throw new Error('请填写至少一个数据记录ID');
      if (!newAnnotationTask.created_by) throw new Error('创建者不能为空');
      if (!Object.keys(newAnnotationTask.annotation_schema || {}).length) throw new Error('标注模式不能为空');
      await trainingDataService.createAnnotationTask(newAnnotationTask);
      await loadOverviewData();
      setNewAnnotationTask({
        name: '',
        description: '',
        task_type: '',
        data_records: [],
        annotation_schema: {},
        guidelines: '',
        assignees: [],
        created_by: currentUser?.username || '',
        deadline: undefined,
      });
      setError(null);
    } catch (error: any) {
      setError(`创建标注任务失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateDataVersion = async () => {
    setLoading(true);
    try {
      if (!authService.getToken()) throw new Error('未登录，请先登录');
      await trainingDataService.createDataVersion(newDataVersion);
      await loadOverviewData();
      setNewDataVersion({
        dataset_name: '',
        version_number: '',
        description: '',
        metadata: {},
      });
      setError(null);
    } catch (error: any) {
      setError(`创建数据版本失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCollectData = async (sourceId: string) => {
    setLoading(true);
    try {
      if (!authService.getToken()) throw new Error('未登录，请先登录');
      await trainingDataService.collectData({ source_id: sourceId });
      setError(null);
    } catch (error: any) {
      setError(`数据收集失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">训练数据管理概览</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-700">数据源</h3>
          <p className="text-2xl font-bold text-blue-900">{dataSources.length}</p>
          <p className="text-sm text-blue-600">
            活跃: {dataSources.filter(s => s.is_active).length}
          </p>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-green-700">数据记录</h3>
          <p className="text-2xl font-bold text-green-900">{collectionStats?.total_records ?? dataRecords.length}</p>
          <p className="text-sm text-green-600">
            已处理: {collectionStats?.status_distribution?.processed ?? 0}
          </p>
        </div>
        
        <div className="bg-yellow-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-yellow-700">标注任务</h3>
          <p className="text-2xl font-bold text-yellow-900">{annotationTasks.length}</p>
          <p className="text-sm text-yellow-600">
            进行中: {annotationTasks.filter(t => t.status === 'in_progress').length}
          </p>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-purple-700">数据集版本</h3>
          <p className="text-2xl font-bold text-purple-900">{versionStats?.total_versions || 0}</p>
          <p className="text-sm text-purple-600">
            数据集: {versionStats?.total_datasets || 0}
          </p>
        </div>
      </div>

      {collectionStats && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">数据质量统计</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600">平均质量分数</p>
              <p className="text-xl font-bold">{(collectionStats.quality_stats?.average ?? 0).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">最高质量分数</p>
              <p className="text-xl font-bold">{(collectionStats.quality_stats?.maximum ?? 0).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">最低质量分数</p>
              <p className="text-xl font-bold">{(collectionStats.quality_stats?.minimum ?? 0).toFixed(2)}</p>
            </div>
          </div>
        </div>
      )}

      {queueStatus && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">处理队列状态</h3>
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
            {JSON.stringify(queueStatus, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );

  const renderDataSources = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">数据源管理</h2>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">创建新数据源</h3>
        <div className="grid grid-cols-2 gap-4">
          <input
            type="text"
            placeholder="数据源ID"
            value={newDataSource.source_id}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, source_id: e.target.value }))}
            className="border p-2 rounded"
          />
          <select
            value={newDataSource.source_type}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, source_type: e.target.value as any }))}
            className="border p-2 rounded"
          >
            <option value="api">API</option>
            <option value="file">文件</option>
            <option value="web">Web</option>
            <option value="database">数据库</option>
          </select>
          <input
            type="text"
            placeholder="数据源名称"
            value={newDataSource.name}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, name: e.target.value }))}
            className="border p-2 rounded"
          />
          <input
            type="text"
            placeholder="描述"
            value={newDataSource.description}
            onChange={(e) => setNewDataSource(prev => ({ ...prev, description: e.target.value }))}
            className="border p-2 rounded"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">配置 (JSON)</label>
          <textarea
            value={JSON.stringify(newDataSource.config, null, 2)}
            onChange={(e) => {
              try {
                const config = JSON.parse(e.target.value);
                setNewDataSource(prev => ({ ...prev, config }));
              } catch {}
            }}
            className="border p-2 rounded w-full h-32"
          />
        </div>
        <button
          onClick={handleCreateDataSource}
          disabled={loading}
          className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? '创建中...' : '创建数据源'}
        </button>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">现有数据源</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300">
            <thead>
              <tr className="bg-gray-50">
                <th className="border border-gray-300 p-2 text-left">源ID</th>
                <th className="border border-gray-300 p-2 text-left">名称</th>
                <th className="border border-gray-300 p-2 text-left">类型</th>
                <th className="border border-gray-300 p-2 text-left">状态</th>
                <th className="border border-gray-300 p-2 text-left">操作</th>
              </tr>
            </thead>
            <tbody>
              {dataSources.map((source) => (
                <tr key={source.id}>
                  <td className="border border-gray-300 p-2">{source.source_id}</td>
                  <td className="border border-gray-300 p-2">{source.name}</td>
                  <td className="border border-gray-300 p-2">{source.source_type}</td>
                  <td className="border border-gray-300 p-2">
                    <span className={`px-2 py-1 rounded text-xs ${source.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {source.is_active ? '活跃' : '非活跃'}
                    </span>
                  </td>
                  <td className="border border-gray-300 p-2">
                    <button
                      onClick={() => handleCollectData(source.source_id)}
                      disabled={loading}
                      className="bg-green-600 text-white px-2 py-1 rounded text-sm hover:bg-green-700 disabled:opacity-50"
                    >
                      收集数据
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderDataRecords = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">数据记录管理</h2>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">数据记录列表</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300">
            <thead>
              <tr className="bg-gray-50">
                <th className="border border-gray-300 p-2 text-left">记录ID</th>
                <th className="border border-gray-300 p-2 text-left">数据源</th>
                <th className="border border-gray-300 p-2 text-left">状态</th>
                <th className="border border-gray-300 p-2 text-left">质量分数</th>
                <th className="border border-gray-300 p-2 text-left">创建时间</th>
              </tr>
            </thead>
            <tbody>
              {dataRecords.map((record) => (
                <tr key={record.id}>
                  <td className="border border-gray-300 p-2 font-mono text-sm">{record.record_id || record.id}</td>
                  <td className="border border-gray-300 p-2">{record.source_id}</td>
                  <td className="border border-gray-300 p-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      record.status === 'processed' ? 'bg-green-100 text-green-800' :
                      record.status === 'failed' ? 'bg-red-100 text-red-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {record.status}
                    </span>
                  </td>
                  <td className="border border-gray-300 p-2">
                    {record.quality_score ? record.quality_score.toFixed(2) : 'N/A'}
                  </td>
                  <td className="border border-gray-300 p-2">
                    {new Date(record.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderAnnotation = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">标注任务管理</h2>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">创建新标注任务</h3>
        <div className="grid grid-cols-2 gap-4">
          <input
            type="text"
            placeholder="任务名称"
            value={newAnnotationTask.name}
            onChange={(e) => setNewAnnotationTask(prev => ({ ...prev, name: e.target.value }))}
            className="border p-2 rounded"
          />
          <input
            type="text"
            placeholder="任务类型"
            value={newAnnotationTask.task_type}
            onChange={(e) => setNewAnnotationTask(prev => ({ ...prev, task_type: e.target.value }))}
            className="border p-2 rounded"
          />
          <input
            type="text"
            placeholder="创建者ID"
            value={newAnnotationTask.created_by}
            onChange={(e) => setNewAnnotationTask(prev => ({ ...prev, created_by: e.target.value }))}
            className="border p-2 rounded"
          />
          <input
            type="datetime-local"
            value={newAnnotationTask.deadline || ''}
            onChange={(e) => setNewAnnotationTask(prev => ({ ...prev, deadline: e.target.value }))}
            className="border p-2 rounded"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">数据记录ID（每行一个）</label>
          <textarea
            value={(newAnnotationTask.data_records || []).join('\n')}
            onChange={(e) => {
              const ids = e.target.value
                .split('\n')
                .map(s => s.trim())
                .filter(Boolean);
              setNewAnnotationTask(prev => ({ ...prev, data_records: ids }));
            }}
            className="border p-2 rounded w-full h-24 font-mono text-sm"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">标注模式 (JSON)</label>
          <textarea
            value={JSON.stringify(newAnnotationTask.annotation_schema, null, 2)}
            onChange={(e) => {
              try {
                const schema = JSON.parse(e.target.value);
                setNewAnnotationTask(prev => ({ ...prev, annotation_schema: schema }));
              } catch {}
            }}
            className="border p-2 rounded w-full h-32 font-mono text-sm"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">分配的标注员（每行一个）</label>
          <textarea
            value={(newAnnotationTask.assignees || []).join('\n')}
            onChange={(e) => {
              const ids = e.target.value
                .split('\n')
                .map(s => s.trim())
                .filter(Boolean);
              setNewAnnotationTask(prev => ({ ...prev, assignees: ids }));
            }}
            className="border p-2 rounded w-full h-24 font-mono text-sm"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">任务描述</label>
          <textarea
            value={newAnnotationTask.description}
            onChange={(e) => setNewAnnotationTask(prev => ({ ...prev, description: e.target.value }))}
            className="border p-2 rounded w-full h-20"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">标注指南</label>
          <textarea
            value={newAnnotationTask.guidelines}
            onChange={(e) => setNewAnnotationTask(prev => ({ ...prev, guidelines: e.target.value }))}
            className="border p-2 rounded w-full h-20"
          />
        </div>
        <button
          onClick={handleCreateAnnotationTask}
          disabled={loading}
          className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? '创建中...' : '创建标注任务'}
        </button>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">标注任务列表</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300">
            <thead>
              <tr className="bg-gray-50">
                <th className="border border-gray-300 p-2 text-left">任务名称</th>
                <th className="border border-gray-300 p-2 text-left">类型</th>
                <th className="border border-gray-300 p-2 text-left">状态</th>
                <th className="border border-gray-300 p-2 text-left">创建者</th>
                <th className="border border-gray-300 p-2 text-left">创建时间</th>
              </tr>
            </thead>
            <tbody>
              {annotationTasks.map((task) => (
                <tr key={task.id}>
                  <td className="border border-gray-300 p-2">{task.name}</td>
                  <td className="border border-gray-300 p-2">{task.task_type}</td>
                  <td className="border border-gray-300 p-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      task.status === 'completed' ? 'bg-green-100 text-green-800' :
                      task.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                      task.status === 'failed' ? 'bg-red-100 text-red-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {task.status}
                    </span>
                  </td>
                  <td className="border border-gray-300 p-2">{task.created_by}</td>
                  <td className="border border-gray-300 p-2">
                    {new Date(task.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderVersions = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">版本管理</h2>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">创建新版本</h3>
        <div className="grid grid-cols-2 gap-4">
          <input
            type="text"
            placeholder="数据集名称"
            value={newDataVersion.dataset_name}
            onChange={(e) => setNewDataVersion(prev => ({ ...prev, dataset_name: e.target.value }))}
            className="border p-2 rounded"
          />
          <input
            type="text"
            placeholder="版本号"
            value={newDataVersion.version_number}
            onChange={(e) => setNewDataVersion(prev => ({ ...prev, version_number: e.target.value }))}
            className="border p-2 rounded"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">版本描述</label>
          <textarea
            value={newDataVersion.description}
            onChange={(e) => setNewDataVersion(prev => ({ ...prev, description: e.target.value }))}
            className="border p-2 rounded w-full h-20"
          />
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">元数据 (JSON)</label>
          <textarea
            value={JSON.stringify(newDataVersion.metadata, null, 2)}
            onChange={(e) => {
              try {
                const metadata = JSON.parse(e.target.value);
                setNewDataVersion(prev => ({ ...prev, metadata }));
              } catch {}
            }}
            className="border p-2 rounded w-full h-32"
          />
        </div>
        <button
          onClick={handleCreateDataVersion}
          disabled={loading}
          className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? '创建中...' : '创建版本'}
        </button>
      </div>

      {versionStats && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">版本统计</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2">版本分布</h4>
              <div className="space-y-2">
                {versionStats.version_distribution.map((dist, index) => (
                  <div key={index} className="flex justify-between">
                    <span className="text-sm">{dist.dataset_name}</span>
                    <span className="text-sm font-mono">{dist.version_count} 版本</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-medium mb-2">存储使用情况</h4>
              <p className="text-sm">总大小: {versionStats.storage_usage.total_size} MB</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderStatistics = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">统计分析</h2>
      
      {collectionStats && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">收集统计</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">{collectionStats.total_records}</p>
              <p className="text-sm text-gray-600">总记录数</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">{collectionStats.status_distribution?.raw ?? 0}</p>
              <p className="text-sm text-gray-600">原始记录</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-yellow-600">{collectionStats.status_distribution?.processed ?? 0}</p>
              <p className="text-sm text-gray-600">已处理记录</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-red-600">{collectionStats.status_distribution?.error ?? 0}</p>
              <p className="text-sm text-gray-600">错误记录</p>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="font-medium mb-4">数据源统计</h4>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-300 p-2 text-left">数据源ID</th>
                    <th className="border border-gray-300 p-2 text-left">总记录</th>
                    <th className="border border-gray-300 p-2 text-left">平均质量</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(collectionStats.source_distribution || {}).map(([sourceId, stat]) => (
                    <tr key={sourceId}>
                      <td className="border border-gray-300 p-2">{sourceId}</td>
                      <td className="border border-gray-300 p-2">{stat.record_count}</td>
                      <td className="border border-gray-300 p-2">{(stat.average_quality ?? 0).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {versionStats && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">版本统计</h3>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2">数据集概况</h4>
              <p className="text-sm">总数据集: {versionStats.total_datasets}</p>
              <p className="text-sm">总版本数: {versionStats.total_versions}</p>
              <p className="text-sm">总记录数: {versionStats.total_records}</p>
            </div>
            <div>
              <h4 className="font-medium mb-2">存储统计</h4>
              <p className="text-sm">总存储: {(versionStats.total_size_bytes / (1024 * 1024)).toFixed(2)} MB</p>
              <p className="text-sm">平均版本大小: {(versionStats.average_size_per_version / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">训练数据管理系统</h1>

        {error && (
          <Alert className="mb-6">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="mb-6">
          <nav className="flex space-x-4">
            {[
              { key: 'overview', label: '系统概览' },
              { key: 'sources', label: '数据源管理' },
              { key: 'records', label: '数据记录' },
              { key: 'annotation', label: '标注任务' },
              { key: 'versions', label: '版本管理' },
              { key: 'statistics', label: '统计分析' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`px-4 py-2 rounded-lg ${
                  activeTab === tab.key
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="bg-gray-50 p-6 rounded-lg">
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'sources' && renderDataSources()}
          {activeTab === 'records' && renderDataRecords()}
          {activeTab === 'annotation' && renderAnnotation()}
          {activeTab === 'versions' && renderVersions()}
          {activeTab === 'statistics' && renderStatistics()}
        </div>
      </div>
    </div>
  );
};

export default TrainingDataManagementPage;
