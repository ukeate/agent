import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, CheckCircle, Clock, Zap } from 'lucide-react';

interface ContextAPIDemo {
  user_id: string;
  session_id: string;
  conversation_id?: string;
  workflow_id?: string;
  max_iterations: number;
  timeout_seconds: number;
  enable_checkpoints: boolean;
}

interface NodeCacheStats {
  hit_count: number;
  miss_count: number;
  total_requests: number;
  hit_rate: string;
}

interface WorkflowExecution {
  id: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  durability: 'exit' | 'async' | 'sync';
  start_time: string;
  end_time?: string;
  messages: Array<{
    role: string;
    content: string;
    timestamp: string;
    metadata?: any;
  }>;
  context_used: string;
  caching_enabled: boolean;
  hooks_applied: string[];
}

const LangGraph065Page: React.FC = () => {
  const [contextDemo, setContextDemo] = useState<ContextAPIDemo>({
    user_id: 'demo_user_123',
    session_id: '550e8400-e29b-41d4-a716-446655440000',
    conversation_id: '',
    workflow_id: '',
    max_iterations: 10,
    timeout_seconds: 300,
    enable_checkpoints: true
  });

  const [durability, setDurability] = useState<'exit' | 'async' | 'sync'>('async');
  const [cachingEnabled, setCachingEnabled] = useState(true);
  const [hooksEnabled, setHooksEnabled] = useState(true);
  const [workflowInput, setWorkflowInput] = useState('演示LangGraph 0.6.5的新特性');
  
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [cacheStats, setCacheStats] = useState<NodeCacheStats | null>(null);
  const [loading, setLoading] = useState(false);

  // 模拟数据更新
  useEffect(() => {
    // 模拟缓存统计
    setCacheStats({
      hit_count: 15,
      miss_count: 8,
      total_requests: 23,
      hit_rate: '65.22%'
    });
  }, []);

  const executeWorkflow = async () => {
    setLoading(true);
    
    try {
      // 模拟工作流执行
      const executionId = `exec_${Date.now()}`;
      const startTime = new Date().toISOString();
      
      setExecution({
        id: executionId,
        status: 'running',
        durability,
        start_time: startTime,
        messages: [
          {
            role: 'system',
            content: 'LangGraph 0.6.5工作流开始执行',
            timestamp: startTime
          }
        ],
        context_used: contextDemo.user_id ? 'new_context_api' : 'legacy_config',
        caching_enabled: cachingEnabled,
        hooks_applied: hooksEnabled ? ['MessageCompressionHook', 'GuardrailsHook'] : []
      });

      // 模拟API调用
      const response = await fetch('/api/v1/langgraph/execute-demo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: workflowInput,
          context: contextDemo,
          durability,
          caching_enabled: cachingEnabled,
          hooks_enabled: hooksEnabled
        })
      });

      if (response.ok) {
        const result = await response.json();
        setExecution(prev => prev ? {
          ...prev,
          status: 'completed',
          end_time: new Date().toISOString(),
          messages: [
            ...prev.messages,
            ...result.messages
          ]
        } : null);
      } else {
        setExecution(prev => prev ? {
          ...prev,
          status: 'failed',
          end_time: new Date().toISOString()
        } : null);
      }
    } catch (error) {
      console.error('工作流执行失败:', error);
      setExecution(prev => prev ? {
        ...prev,
        status: 'failed',
        end_time: new Date().toISOString()
      } : null);
    } finally {
      setLoading(false);
    }
  };

  const clearCache = async () => {
    try {
      await fetch('/api/v1/langgraph/cache/clear', {
        method: 'POST'
      });
      setCacheStats({
        hit_count: 0,
        miss_count: 0,
        total_requests: 0,
        hit_rate: '0.00%'
      });
    } catch (error) {
      console.error('清空缓存失败:', error);
    }
  };

  const StatusIcon = ({ status }: { status: string }) => {
    switch (status) {
      case 'running':
        return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">LangGraph 0.6.5 核心特性演示</h1>
        <p className="text-gray-600">
          体验LangGraph 0.6.5的新Context API、durability控制、Node Caching和Pre/Post Model Hooks
        </p>
      </div>

      <Tabs defaultValue="context-api" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="context-api">Context API</TabsTrigger>
          <TabsTrigger value="durability">Durability控制</TabsTrigger>
          <TabsTrigger value="caching">Node Caching</TabsTrigger>
          <TabsTrigger value="hooks">Model Hooks</TabsTrigger>
        </TabsList>

        <TabsContent value="context-api">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                新Context API演示
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="user_id">用户ID</Label>
                  <Input
                    id="user_id"
                    value={contextDemo.user_id}
                    onChange={(e) => setContextDemo(prev => ({ ...prev, user_id: e.target.value }))}
                    placeholder="demo_user_123"
                  />
                </div>
                <div>
                  <Label htmlFor="session_id">会话ID (UUID)</Label>
                  <Input
                    id="session_id"
                    value={contextDemo.session_id}
                    onChange={(e) => setContextDemo(prev => ({ ...prev, session_id: e.target.value }))}
                    placeholder="550e8400-e29b-41d4-a716-446655440000"
                  />
                </div>
                <div>
                  <Label htmlFor="conversation_id">对话ID (可选)</Label>
                  <Input
                    id="conversation_id"
                    value={contextDemo.conversation_id}
                    onChange={(e) => setContextDemo(prev => ({ ...prev, conversation_id: e.target.value }))}
                    placeholder="留空则自动生成"
                  />
                </div>
                <div>
                  <Label htmlFor="workflow_id">工作流ID (可选)</Label>
                  <Input
                    id="workflow_id"
                    value={contextDemo.workflow_id}
                    onChange={(e) => setContextDemo(prev => ({ ...prev, workflow_id: e.target.value }))}
                    placeholder="留空则自动生成"
                  />
                </div>
                <div>
                  <Label htmlFor="max_iterations">最大迭代次数</Label>
                  <Input
                    id="max_iterations"
                    type="number"
                    value={contextDemo.max_iterations}
                    onChange={(e) => setContextDemo(prev => ({ ...prev, max_iterations: parseInt(e.target.value) }))}
                  />
                </div>
                <div>
                  <Label htmlFor="timeout_seconds">超时时间(秒)</Label>
                  <Input
                    id="timeout_seconds"
                    type="number"
                    value={contextDemo.timeout_seconds}
                    onChange={(e) => setContextDemo(prev => ({ ...prev, timeout_seconds: parseInt(e.target.value) }))}
                  />
                </div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Context Schema预览:</h4>
                <pre className="text-sm text-gray-700 overflow-x-auto">
                  {JSON.stringify(contextDemo, null, 2)}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="durability">
          <Card>
            <CardHeader>
              <CardTitle>Durability持久化控制</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                {(['exit', 'async', 'sync'] as const).map((mode) => (
                  <div
                    key={mode}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      durability === mode 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setDurability(mode)}
                  >
                    <div className="font-medium mb-2 capitalize">{mode}</div>
                    <div className="text-sm text-gray-600">
                      {mode === 'exit' && '仅在图完成时保存，性能最佳'}
                      {mode === 'async' && '异步保存，平衡性能和持久性'}
                      {mode === 'sync' && '同步保存，最高持久性保证'}
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant={durability === 'exit' ? 'secondary' : durability === 'async' ? 'default' : 'destructive'}>
                  当前模式: {durability}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="caching">
          <Card>
            <CardHeader>
              <CardTitle>Node Caching节点缓存</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="caching"
                    checked={cachingEnabled}
                    onChange={(e) => setCachingEnabled(e.target.checked)}
                    className="h-4 w-4"
                  />
                  <Label htmlFor="caching">启用节点缓存</Label>
                </div>
                <Button variant="outline" onClick={clearCache}>
                  清空缓存
                </Button>
              </div>

              {cacheStats && (
                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">{cacheStats.hit_count}</div>
                    <div className="text-sm text-green-600">缓存命中</div>
                  </div>
                  <div className="bg-red-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-red-600">{cacheStats.miss_count}</div>
                    <div className="text-sm text-red-600">缓存未命中</div>
                  </div>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{cacheStats.total_requests}</div>
                    <div className="text-sm text-blue-600">总请求数</div>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">{cacheStats.hit_rate}</div>
                    <div className="text-sm text-purple-600">命中率</div>
                  </div>
                </div>
              )}

              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">缓存策略说明:</h4>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li>• <strong>Fast Cache:</strong> TTL 60秒，适合频繁变化的节点</li>
                  <li>• <strong>Standard Cache:</strong> TTL 300秒，适合一般节点</li>
                  <li>• <strong>Long Cache:</strong> TTL 3600秒，适合计算密集型节点</li>
                  <li>• <strong>Context Aware:</strong> 包含上下文哈希的缓存</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hooks">
          <Card>
            <CardHeader>
              <CardTitle>Pre/Post Model Hooks</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="hooks"
                  checked={hooksEnabled}
                  onChange={(e) => setHooksEnabled(e.target.checked)}
                  className="h-4 w-4"
                />
                <Label htmlFor="hooks">启用Model Hooks</Label>
              </div>

              {hooksEnabled && (
                <div className="space-y-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">Pre Model Hooks (预处理)</h4>
                    <ul className="text-sm text-blue-800 space-y-1">
                      <li>• <strong>MessageCompressionHook:</strong> 压缩消息历史，减少token使用</li>
                      <li>• <strong>ContextEnrichmentHook:</strong> 丰富上下文信息</li>
                    </ul>
                  </div>

                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-medium text-green-900 mb-2">Post Model Hooks (后处理)</h4>
                    <ul className="text-sm text-green-800 space-y-1">
                      <li>• <strong>GuardrailsHook:</strong> 内容安全检查和防护栏</li>
                      <li>• <strong>ResponseQualityHook:</strong> 响应质量检查</li>
                    </ul>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 工作流执行区域 */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>工作流执行演示</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="workflow_input">工作流输入</Label>
            <Textarea
              id="workflow_input"
              value={workflowInput}
              onChange={(e) => setWorkflowInput(e.target.value)}
              placeholder="输入要处理的内容..."
              rows={3}
            />
          </div>

          <div className="flex gap-4">
            <Button 
              onClick={executeWorkflow} 
              disabled={loading}
              className="flex items-center gap-2"
            >
              {loading ? <Clock className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
              执行工作流
            </Button>
          </div>

          {execution && (
            <div className="border rounded-lg p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <StatusIcon status={execution.status} />
                  <span className="font-medium">执行ID: {execution.id}</span>
                  <Badge variant="outline">{execution.status}</Badge>
                </div>
                <div className="text-sm text-gray-500">
                  开始时间: {new Date(execution.start_time).toLocaleString()}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label className="text-sm font-medium">持久化模式</Label>
                  <div className="text-sm">{execution.durability}</div>
                </div>
                <div>
                  <Label className="text-sm font-medium">Context API</Label>
                  <div className="text-sm">{execution.context_used}</div>
                </div>
                <div>
                  <Label className="text-sm font-medium">缓存状态</Label>
                  <div className="text-sm">{execution.caching_enabled ? '已启用' : '已禁用'}</div>
                </div>
              </div>

              {execution.hooks_applied.length > 0 && (
                <div>
                  <Label className="text-sm font-medium">应用的Hooks</Label>
                  <div className="flex gap-2 mt-1">
                    {execution.hooks_applied.map(hook => (
                      <Badge key={hook} variant="secondary" className="text-xs">
                        {hook}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <Label className="text-sm font-medium">执行日志</Label>
                <div className="bg-gray-50 p-3 rounded max-h-64 overflow-y-auto">
                  {execution.messages.map((msg, idx) => (
                    <div key={idx} className="text-sm mb-2">
                      <span className="text-gray-500">[{new Date(msg.timestamp).toLocaleTimeString()}]</span>
                      <span className="font-medium ml-2">{msg.role}:</span>
                      <span className="ml-2">{msg.content}</span>
                      {msg.metadata && (
                        <div className="ml-4 text-xs text-gray-600 mt-1">
                          {JSON.stringify(msg.metadata, null, 2)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default LangGraph065Page;