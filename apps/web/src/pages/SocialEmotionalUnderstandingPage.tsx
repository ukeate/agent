import React, { useState, useEffect, useRef } from 'react';
import { Alert } from '../components/ui/alert';
import { 
import { logger } from '../utils/logger'
  socialEmotionalService,
  AnalysisRequest,
  GroupEmotionResponse,
  RelationshipAnalysisResponse,
  SocialDecisionResponse,
  ComprehensiveAnalysisResponse,
  SystemAnalytics,
  ParticipantData,
  SocialEnvironmentData
} from '../services/socialEmotionalService';

const SocialEmotionalUnderstandingPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis' | 'relationships' | 'decisions' | 'realtime' | 'analytics'>('overview');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 状态数据
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [systemAnalytics, setSystemAnalytics] = useState<SystemAnalytics | null>(null);
  const [analysisResults, setAnalysisResults] = useState<ComprehensiveAnalysisResponse | null>(null);
  const [realtimeData, setRealtimeData] = useState<any[]>([]);
  
  // 表单状态
  const [sessionId, setSessionId] = useState('');
  const [participants, setParticipants] = useState<ParticipantData[]>([]);
  const [socialEnvironment, setSocialEnvironment] = useState<SocialEnvironmentData>({
    scenario: 'formal_meeting',
    participants_count: 2,
    formality_level: 0.5,
    emotional_intensity: 0.5,
    time_pressure: 0.5,
    cultural_context: ''
  });

  // WebSocket 连接
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    loadSystemData();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const loadSystemData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthData, analyticsData] = await Promise.all([
        socialEmotionalService.getHealthCheck(),
        socialEmotionalService.getSystemAnalytics()
      ]);
      
      setSystemHealth(healthData);
      setSystemAnalytics(analyticsData);
    } catch (err: any) {
      setError(`加载系统数据失败: ${err.response?.data?.detail || err.message}`);
    }
    setLoading(false);
  };

  const handleAnalysis = async (analysisType: 'group_emotion' | 'relationships' | 'decisions' | 'comprehensive') => {
    if (!sessionId || participants.length === 0) {
      setError('请输入会话ID并添加至少一个参与者');
      return;
    }
    if (participants.length < 2 && (analysisType === 'group_emotion' || analysisType === 'relationships' || analysisType === 'comprehensive')) {
      setError('群体/关系/综合分析需要至少两个参与者');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const request: AnalysisRequest = {
        session_id: sessionId,
        participants,
        social_environment: {
          ...socialEnvironment,
          participants_count: participants.length
        },
        analysis_types: [analysisType],
        real_time: false
      };

      let result;
      switch (analysisType) {
        case 'group_emotion':
          result = await socialEmotionalService.analyzeGroupEmotion(request);
          break;
        case 'relationships':
          result = await socialEmotionalService.analyzeRelationships(request);
          break;
        case 'decisions':
          result = await socialEmotionalService.generateSocialDecisions(request);
          break;
        case 'comprehensive':
          result = await socialEmotionalService.comprehensiveAnalysis(request);
          setAnalysisResults(result);
          break;
      }
      
      logger.log(`${analysisType} 分析结果:`, result);
    } catch (err: any) {
      setError(`分析失败: ${err.response?.data?.detail || err.message}`);
    }
    setLoading(false);
  };

  const handleStartRealtime = () => {
    if (!sessionId) {
      setError('请先输入会话ID');
      return;
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    setRealtimeData([]);
    wsRef.current = socialEmotionalService.createWebSocketConnection(
      sessionId,
      (data) => {
        setRealtimeData(prev => [...prev, { ...data, timestamp: new Date().toLocaleTimeString() }]);
      },
      (error) => {
        setError('实时连接出现错误');
      },
      (event) => {
        logger.log('WebSocket连接已关闭');
      }
    );
  };

  const addParticipant = () => {
    const newParticipant: ParticipantData = {
      participant_id: `participant_${participants.length + 1}`,
      name: '',
      emotion_data: {
        emotions: {},
        intensity: 0,
        confidence: 0,
        context: socialEnvironment.scenario
      },
      cultural_indicators: {}
    };
    setParticipants([...participants, newParticipant]);
  };

  const updateParticipant = (index: number, field: keyof ParticipantData, value: any) => {
    const updated = [...participants];
    updated[index] = { ...updated[index], [field]: value };
    setParticipants(updated);
  };

  const removeParticipant = (index: number) => {
    setParticipants(participants.filter((_, i) => i !== index));
  };
  
  const updateEmotionData = (index: number, field: 'intensity' | 'confidence', value: number) => {
    const updated = [...participants];
    updated[index] = {
      ...updated[index],
      emotion_data: {
        ...updated[index].emotion_data,
        [field]: value
      }
    };
    setParticipants(updated);
  };

  const updateEmotionMap = (index: number, value: string) => {
    try {
      const parsed = value ? JSON.parse(value) : {};
      const updated = [...participants];
      updated[index] = {
        ...updated[index],
        emotion_data: {
          ...updated[index].emotion_data,
          emotions: parsed
        }
      };
      setParticipants(updated);
      setError(null);
    } catch {
      setError('情感数据JSON格式错误');
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">社交情感理解系统概览</h2>

      {systemHealth && (
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-green-800 mb-2">系统状态</h3>
          <p className="text-green-700">状态: {systemHealth.status}</p>
          <p className="text-green-700">时间: {new Date(systemHealth.timestamp).toLocaleString()}</p>
        </div>
      )}

      {systemAnalytics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-800">总会话数</h3>
            <p className="text-2xl font-bold text-blue-900">{systemAnalytics.total_sessions}</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-green-800">分析次数</h3>
            <p className="text-2xl font-bold text-green-900">{systemAnalytics.total_analyses_performed}</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-purple-800">平均置信度</h3>
            <p className="text-2xl font-bold text-purple-900">{(systemAnalytics.avg_confidence_score * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-orange-800">成功率</h3>
            <p className="text-2xl font-bold text-orange-900">{(systemAnalytics.system_performance.success_rate * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}

      {systemAnalytics && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">常见场景分析</h3>
          <div className="space-y-2">
            {systemAnalytics.most_common_scenarios.map((scenario, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="capitalize">{scenario.scenario}</span>
                <span className="font-bold">{scenario.frequency}次</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderAnalysis = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">社交情感分析</h2>

      {/* 会话配置 */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">会话配置</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2">会话ID</label>
            <input
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="输入会话ID"
              className="w-full p-2 border rounded"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">场景类型</label>
            <select
              value={socialEnvironment.scenario}
              onChange={(e) => setSocialEnvironment({...socialEnvironment, scenario: e.target.value})}
              className="w-full p-2 border rounded"
            >
              <option value="formal_meeting">商务会议</option>
              <option value="casual_conversation">日常对话</option>
              <option value="conflict_resolution">冲突解决</option>
              <option value="team_brainstorming">团队协作</option>
              <option value="presentation">演示汇报</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2">正式程度 (0-1)</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={socialEnvironment.formality_level}
              onChange={(e) => setSocialEnvironment({...socialEnvironment, formality_level: parseFloat(e.target.value)})}
              className="w-full p-2 border rounded"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">情感强度 (0-1)</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={socialEnvironment.emotional_intensity}
              onChange={(e) => setSocialEnvironment({...socialEnvironment, emotional_intensity: parseFloat(e.target.value)})}
              className="w-full p-2 border rounded"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">时间压力 (0-1)</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={socialEnvironment.time_pressure}
              onChange={(e) => setSocialEnvironment({...socialEnvironment, time_pressure: parseFloat(e.target.value)})}
              className="w-full p-2 border rounded"
            />
          </div>
        </div>

      </div>

      {/* 参与者管理 */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">参与者管理</h3>
          <button
            onClick={addParticipant}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            添加参与者
          </button>
        </div>

        {participants.map((participant, index) => (
          <div key={index} className="border p-4 rounded mb-4">
            <div className="flex justify-between items-center mb-2">
              <h4 className="font-medium">参与者 {index + 1}</h4>
              <button
                onClick={() => removeParticipant(index)}
                className="px-2 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
              >
                删除
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <input
                type="text"
                placeholder="参与者姓名"
                value={participant.name}
                onChange={(e) => updateParticipant(index, 'name', e.target.value)}
                className="p-2 border rounded"
              />
              <input
                type="text"
                placeholder="参与者ID"
                value={participant.participant_id}
                onChange={(e) => updateParticipant(index, 'participant_id', e.target.value)}
                className="p-2 border rounded"
              />
            </div>
            <div className="mt-2 text-sm text-gray-600">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  placeholder="情感强度(0-1)"
                  value={participant.emotion_data.intensity}
                  onChange={(e) => updateEmotionData(index, 'intensity', Number(e.target.value))}
                  className="p-2 border rounded"
                />
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  placeholder="置信度(0-1)"
                  value={participant.emotion_data.confidence}
                  onChange={(e) => updateEmotionData(index, 'confidence', Number(e.target.value))}
                  className="p-2 border rounded"
                />
                <input
                  type="text"
                  placeholder="情感上下文"
                  value={participant.emotion_data.context || ''}
                  onChange={(e) => updateParticipant(index, 'emotion_data', {
                    ...participant.emotion_data,
                    context: e.target.value
                  })}
                  className="p-2 border rounded"
                />
              </div>
              <textarea
                placeholder='情感数据(JSON)，例如 {"joy":0.6,"anxiety":0.2}'
                value={JSON.stringify(participant.emotion_data.emotions)}
                onChange={(e) => updateEmotionMap(index, e.target.value)}
                className="mt-2 p-2 border rounded w-full"
                rows={2}
              />
            </div>
          </div>
        ))}
      </div>

      {/* 分析操作 */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">分析操作</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button
            onClick={() => handleAnalysis('group_emotion')}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            群体情感分析
          </button>
          <button
            onClick={() => handleAnalysis('relationships')}
            disabled={loading}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
          >
            关系分析
          </button>
          <button
            onClick={() => handleAnalysis('decisions')}
            disabled={loading}
            className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 disabled:opacity-50"
          >
            决策建议
          </button>
          <button
            onClick={() => handleAnalysis('comprehensive')}
            disabled={loading}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            综合分析
          </button>
        </div>
      </div>

      {/* 分析结果 */}
      {analysisResults && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">分析结果</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div>
              <p className="text-sm text-gray-600">关系健康度</p>
              <p className="text-xl font-bold">{analysisResults.analysis_summary.relationship_health.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">社交凝聚力</p>
              <p className="text-xl font-bold">{analysisResults.analysis_summary.social_cohesion.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">文化和谐度</p>
              <p className="text-xl font-bold">{analysisResults.analysis_summary.cultural_harmony.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">沟通效果</p>
              <p className="text-xl font-bold">{analysisResults.analysis_summary.communication_effectiveness.toFixed(2)}</p>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">优先建议</h4>
            <div className="space-y-2">
              {analysisResults.priority_recommendations.slice(0, 3).map((rec, index) => (
                <div key={index} className={`p-2 rounded ${
                  rec.priority === 'high' ? 'bg-red-50 border-l-4 border-red-500' :
                  rec.priority === 'medium' ? 'bg-yellow-50 border-l-4 border-yellow-500' :
                  'bg-green-50 border-l-4 border-green-500'
                }`}>
                  <p className="font-medium">{rec.category}</p>
                  <p className="text-sm">{rec.recommendation}</p>
                  <p className="text-xs text-gray-500">预期影响: {(rec.expected_impact * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderRealtime = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">实时分析</h2>
      
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">实时连接控制</h3>
        <div className="flex gap-4 items-center">
          <input
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            placeholder="输入会话ID"
            className="flex-1 p-2 border rounded"
          />
          <button
            onClick={handleStartRealtime}
            disabled={!sessionId}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            开始实时分析
          </button>
          <button
            onClick={() => {
              if (wsRef.current) {
                wsRef.current.close();
                setRealtimeData([]);
              }
            }}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            停止连接
          </button>
        </div>
      </div>

      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">实时数据流</h3>
        <div className="h-64 overflow-y-auto border rounded p-2 bg-gray-50">
          {realtimeData.length === 0 ? (
            <p className="text-gray-500 text-center">暂无实时数据</p>
          ) : (
            <div className="space-y-2">
              {realtimeData.map((data, index) => (
                <div key={index} className="text-sm">
                  <span className="text-gray-500">[{data.timestamp}]</span>
                  <span className="ml-2">{JSON.stringify(data, null, 2)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">社交情感理解系统</h1>
        <p className="text-gray-600">分析和理解社交情感互动，提供智能决策建议</p>
      </div>

      {error && (
        <Alert type="error" className="mb-6">
          {error}
        </Alert>
      )}

      <div className="mb-6">
        <nav className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
          {[
            { key: 'overview', label: '系统概览' },
            { key: 'analysis', label: '情感分析' },
            { key: 'relationships', label: '关系分析' },
            { key: 'decisions', label: '决策建议' },
            { key: 'realtime', label: '实时监控' },
            { key: 'analytics', label: '数据分析' }
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.key
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {loading && (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      )}

      <div className="bg-gray-50 rounded-lg p-6">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'analysis' && renderAnalysis()}
        {activeTab === 'realtime' && renderRealtime()}
        {(activeTab === 'relationships' || activeTab === 'decisions' || activeTab === 'analytics') && (
          <div className="text-center py-8 text-gray-500">
            <h3 className="text-lg font-medium mb-2">功能开发中</h3>
            <p>此功能正在开发中，敬请期待</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SocialEmotionalUnderstandingPage;
