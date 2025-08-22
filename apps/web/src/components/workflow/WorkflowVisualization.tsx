import React, { useCallback, useEffect, useState } from 'react';
import { Card, Spin, Alert, Tooltip, message } from 'antd';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  ConnectionMode,
  Background,
  Controls,
  MiniMap,
  Panel,
  NodeTypes,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import NodeDetailPanel from './NodeDetailPanel';
import WorkflowDebugPanel from './WorkflowDebugPanel';
import { workflowWebSocketService } from '../../services/workflowWebSocketService';

// 定义工作流状态类型
interface WorkflowState {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  type: 'start' | 'process' | 'decision' | 'end';
}

// 自定义节点组件
const WorkflowNode: React.FC<{ data: WorkflowState }> = ({ data }) => {
  const getNodeColor = (status: string) => {
    switch (status) {
      case 'running': return '#1890ff';
      case 'completed': return '#52c41a';
      case 'failed': return '#ff4d4f';
      case 'paused': return '#faad14';
      default: return '#d9d9d9';
    }
  };

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'start': return '▶️';
      case 'process': return '⚙️';
      case 'decision': return '🔄';
      case 'end': return '🏁';
      default: return '📦';
    }
  };

  return (
    <Tooltip title={`${data.name} - ${data.status}`}>
      <div
        style={{
          padding: '10px',
          borderRadius: '8px',
          background: getNodeColor(data.status),
          color: 'white',
          border: '2px solid #fff',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          minWidth: '120px',
          textAlign: 'center',
        }}
      >
        {/* 入口handle，除了开始节点都有 */}
        {data.type !== 'start' && (
          <Handle
            type="target"
            position={Position.Left}
            style={{ background: '#fff', border: '2px solid #555' }}
          />
        )}
        
        <div style={{ fontSize: '18px', marginBottom: '4px' }}>
          {getNodeIcon(data.type)}
        </div>
        <div style={{ fontWeight: 'bold', fontSize: '12px' }}>
          {data.name}
        </div>
        <div style={{ fontSize: '10px', opacity: 0.9 }}>
          {data.status}
        </div>
        
        {/* 出口handle，除了结束节点和条件判断节点都有普通的 */}
        {data.type !== 'end' && data.type !== 'decision' && (
          <Handle
            type="source"
            position={Position.Right}
            style={{ background: '#fff', border: '2px solid #555' }}
          />
        )}
        
        {/* 条件判断节点有两个特殊的出口handle */}
        {data.type === 'decision' && (
          <>
            <Handle
              type="source"
              position={Position.Right}
              id="high"
              style={{ 
                background: '#52c41a', 
                border: '2px solid #fff',
                top: '30%'
              }}
            />
            <Handle
              type="source"
              position={Position.Right}
              id="low"
              style={{ 
                background: '#faad14', 
                border: '2px solid #fff',
                top: '70%'
              }}
            />
          </>
        )}
      </div>
    </Tooltip>
  );
};

// 自定义节点类型
const nodeTypes: NodeTypes = {
  workflowNode: WorkflowNode,
};

interface WorkflowVisualizationProps {
  workflowId: string;
  onNodeClick?: (nodeId: string, nodeData?: WorkflowState) => void;
  demoMode?: boolean; // 新增：演示模式标识
}

export const WorkflowVisualization: React.FC<WorkflowVisualizationProps> = ({
  workflowId,
  onNodeClick = () => {},
  demoMode = false
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [workflowData, setWorkflowData] = useState<any>(null);
  const [selectedNode, setSelectedNode] = useState<WorkflowState | null>(null);
  const [detailPanelVisible, setDetailPanelVisible] = useState(false);
  const [debugPanelVisible, setDebugPanelVisible] = useState(false);
  const [stateHistory, setStateHistory] = useState<any[]>([]);
  const [executionSteps, setExecutionSteps] = useState<any[]>([]);

  // 模拟从API获取工作流数据
  const fetchWorkflowData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // 定义通用的节点和边结构
      let nodeStatusMap: { [key: string]: string } = {
        'start': 'pending',
        'process': 'pending', 
        'decision': 'pending',
        'path_a': 'pending',
        'path_b': 'pending',
        'end': 'pending'
      };

      // 演示模式：直接使用模拟数据，不调用API
      if (demoMode) {
        // 模拟演示数据
        const demoData = {
          id: workflowId,
          name: '演示工作流',
          status: 'running',
          created_at: '2025-01-01T10:00:00Z'
        };
        setWorkflowData(demoData);
        
        // 演示模式下的模拟状态
        nodeStatusMap = {
          'start': 'completed',
          'process': 'completed', 
          'decision': 'completed',
          'path_a': 'pending',
          'path_b': 'completed',
          'end': 'completed'
        };
      } else {
        // 生产模式：调用实际API
        const response = await fetch(`http://localhost:8000/api/v1/workflows/${workflowId}`);
        if (!response || !response.ok) {
          throw new Error('Failed to fetch workflow data');
        }
        
        const data = await response.json();
        setWorkflowData(data);
        
        // 根据实际工作流状态更新节点状态
        const isCompleted = data.status === 'completed';
        const executionLog = data.current_state?.context?.execution_log || [];
        
        // 根据执行日志更新节点状态
        executionLog.forEach((log: any) => {
          if (log.node === 'start') nodeStatusMap['start'] = 'completed';
          if (log.node === 'process') nodeStatusMap['process'] = 'completed';
          if (log.node === 'decision') nodeStatusMap['decision'] = 'completed';
          if (log.node === 'path_a') nodeStatusMap['path_a'] = 'completed';
          if (log.node === 'path_b') nodeStatusMap['path_b'] = 'completed';
          if (log.node === 'end') nodeStatusMap['end'] = 'completed';
        });
        
        // 如果工作流正在运行，更新当前节点状态
        if (data.status === 'running') {
          const currentNode = data.current_state?.metadata?.current_node;
          if (currentNode && nodeStatusMap[currentNode] === 'pending') {
            nodeStatusMap[currentNode] = 'running';
          }
        }
      }
      
      // 生成条件分支工作流节点（两种模式通用）
      const sampleNodes: Node[] = [
        {
          id: 'start',
          type: 'workflowNode',
          position: { x: 100, y: 100 },
          data: { id: 'start', name: '开始', status: nodeStatusMap['start'], type: 'start' },
        },
        {
          id: 'process',
          type: 'workflowNode',
          position: { x: 300, y: 100 },
          data: { id: 'process', name: '数据处理', status: nodeStatusMap['process'], type: 'process' },
        },
        {
          id: 'decision',
          type: 'workflowNode',
          position: { x: 500, y: 100 },
          data: { id: 'decision', name: '条件判断', status: nodeStatusMap['decision'], type: 'decision' },
        },
        {
          id: 'path_a',
          type: 'workflowNode',
          position: { x: 700, y: 50 },
          data: { id: 'path_a', name: '路径A', status: nodeStatusMap['path_a'], type: 'process' },
        },
        {
          id: 'path_b',
          type: 'workflowNode',
          position: { x: 700, y: 150 },
          data: { id: 'path_b', name: '路径B', status: nodeStatusMap['path_b'], type: 'process' },
        },
        {
          id: 'end',
          type: 'workflowNode',
          position: { x: 900, y: 100 },
          data: { id: 'end', name: '结束', status: nodeStatusMap['end'], type: 'end' },
        },
      ];

      // 生成条件分支工作流边（两种模式通用）
      const sampleEdges: Edge[] = [
        { 
          id: 'e1-2', 
          source: 'start', 
          target: 'process', 
          animated: true,
          style: { stroke: '#1890ff', strokeWidth: 2 }
        },
        { 
          id: 'e2-3', 
          source: 'process', 
          target: 'decision', 
          animated: true,
          style: { stroke: '#1890ff', strokeWidth: 2 }
        },
        { 
          id: 'e3-4', 
          source: 'decision', 
          sourceHandle: 'high',
          target: 'path_a', 
          label: '高质量', 
          style: { stroke: '#52c41a', strokeWidth: 2 },
          labelStyle: { fill: '#52c41a', fontWeight: 'bold' }
        },
        { 
          id: 'e3-5', 
          source: 'decision', 
          sourceHandle: 'low',
          target: 'path_b', 
          label: '低质量', 
          style: { stroke: '#faad14', strokeWidth: 2 },
          labelStyle: { fill: '#faad14', fontWeight: 'bold' }
        },
        { 
          id: 'e4-6', 
          source: 'path_a', 
          target: 'end',
          style: { stroke: '#52c41a', strokeWidth: 2 }
        },
        { 
          id: 'e5-6', 
          source: 'path_b', 
          target: 'end',
          style: { stroke: '#faad14', strokeWidth: 2 }
        },
      ];

      setNodes(sampleNodes);
      setEdges(sampleEdges);
      
      // 共同的演示数据设置（演示模式和生产模式都会用到）
      
      // 模拟状态历史数据
      const mockStateHistory = [
        {
          timestamp: '2025-01-01 10:00:00',
          nodeId: 'start',
          nodeName: '开始',
          previousStatus: 'pending',
          newStatus: 'completed',
          metadata: { duration: 100 }
        },
        {
          timestamp: '2025-01-01 10:00:05',
          nodeId: 'process1',
          nodeName: '数据处理',
          previousStatus: 'pending',
          newStatus: 'running',
          metadata: { input_size: 1024 }
        },
      ];
      
      // 生成真实的模拟数据处理结果
      const generateMockData = () => {
        const categories = ['用户行为', '交易记录', '系统日志', '设备状态', '网络流量'];
        const recordCount = Math.floor(Math.random() * 80) + 120; // 120-200条记录
        
        const records = [];
        for (let i = 0; i < recordCount; i++) {
          records.push({
            id: `record_${String(i + 1).padStart(4, '0')}`,
            category: categories[Math.floor(Math.random() * categories.length)],
            value: Math.round((Math.random() * 989.4 + 10.5) * 100) / 100,
            status: ['success', 'warning', 'info'][Math.floor(Math.random() * 3)]
          });
        }
        
        // 统计数据
        const stats = {
          total_records: recordCount,
          by_category: {},
          by_status: {},
          avg_value: Math.round((records.reduce((sum, r) => sum + r.value, 0) / recordCount) * 100) / 100,
          processing_time_ms: Math.floor(Math.random() * 300) + 200
        };
        
        // 计算各类别统计
        records.forEach(record => {
          stats.by_category[record.category] = (stats.by_category[record.category] || 0) + 1;
          stats.by_status[record.status] = (stats.by_status[record.status] || 0) + 1;
        });
        
        return { records: records.slice(0, 3), stats }; // 只返回前3条作为示例
      };
      
      const mockData = generateMockData();
      
      // 模拟执行步骤数据
      const mockExecutionSteps = [
        {
          stepId: 'step-1',
          timestamp: '2025-01-01 10:00:00',
          nodeId: 'start',
          nodeName: '开始',
          action: 'initialize',
          duration: 100,
          status: 'success' as const,
          details: { version: '1.0.0' }
        },
        {
          stepId: 'step-2',
          timestamp: '2025-01-01 10:00:05',
          nodeId: 'process1',
          nodeName: '数据处理',
          action: 'process_data',
          duration: mockData.stats.processing_time_ms,
          status: 'success' as const,
          details: {
            total_records: mockData.stats.total_records,
            by_category: mockData.stats.by_category,
            by_status: mockData.stats.by_status,
            avg_value: mockData.stats.avg_value,
            sample_records: mockData.records
          }
        },
      ];
      
      setStateHistory(mockStateHistory);
      setExecutionSteps(mockExecutionSteps);
      
    } catch (err) {
      console.error('Error fetching workflow data:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [workflowId]);

  // 处理节点点击事件
  const handleNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    console.log('Node clicked:', node);
    const nodeData = node.data as WorkflowState;
    setSelectedNode(nodeData);
    setDetailPanelVisible(true);
    onNodeClick(node.id, nodeData);
  }, [onNodeClick]);

  // 处理节点操作
  const handleNodeAction = useCallback(async (action: string, nodeId: string) => {
    try {
      console.log(`执行操作: ${action} on node: ${nodeId}`);
      
      // TODO: 实际的API调用
      // await fetch(`/api/v1/workflows/${workflowId}/nodes/${nodeId}/${action}`, {
      //   method: 'POST'
      // });
      
      message.success(`${action} 操作执行成功`);
      
      // 更新节点状态
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            let newStatus = node.data.status;
            switch (action) {
              case 'pause':
                newStatus = 'paused';
                break;
              case 'resume':
                newStatus = 'running';
                break;
              case 'stop':
                newStatus = 'failed';
                break;
              case 'retry':
                newStatus = 'running';
                break;
            }
            return {
              ...node,
              data: { ...node.data, status: newStatus }
            };
          }
          return node;
        })
      );
      
      // 更新选中节点的状态
      if (selectedNode && selectedNode.id === nodeId) {
        let updatedStatus: WorkflowState['status'];
        switch (action) {
          case 'pause':
            updatedStatus = 'paused';
            break;
          case 'resume':
            updatedStatus = 'running';
            break;
          case 'stop':
            updatedStatus = 'failed';
            break;
          case 'retry':
            updatedStatus = 'running';
            break;
          default:
            updatedStatus = selectedNode.status;
        }
        setSelectedNode(prev => prev ? { ...prev, status: updatedStatus } : null);
      }
      
    } catch (err) {
      console.error('节点操作失败:', err);
      message.error('操作失败，请重试');
    }
  }, [workflowId, selectedNode, setNodes]);

  // 关闭详情面板
  const handleCloseDetailPanel = useCallback(() => {
    setDetailPanelVisible(false);
    setSelectedNode(null);
  }, []);

  // 处理连接
  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // WebSocket更新处理
  const handleWebSocketUpdate = useCallback((message: any) => {
    console.log('收到WebSocket更新:', message);
    
    if (message.type === 'initial_status' || message.type === 'status_update') {
      const workflowData = message.data;
      if (workflowData) {
        setWorkflowData(workflowData);
        
        // TODO: 根据实际API数据更新节点状态
        // 这里应该根据workflowData.current_state来更新节点
        // 现在使用模拟数据进行演示
        
        // 模拟节点状态更新
        setNodes((nds) =>
          nds.map((node, index) => {
            // 模拟不同节点的状态变化
            let newStatus = node.data.status;
            if (workflowData.status === 'running' && index === 1) {
              newStatus = 'running';
            } else if (workflowData.status === 'completed' && index <= 1) {
              newStatus = 'completed';
            }
            
            return {
              ...node,
              data: { ...node.data, status: newStatus }
            };
          })
        );
      }
    }
  }, [setNodes]);

  useEffect(() => {
    fetchWorkflowData();
    
    // 只在非演示模式下建立WebSocket连接
    if (!demoMode) {
      workflowWebSocketService.connect(workflowId, handleWebSocketUpdate);
    }
    
    // 清理函数
    return () => {
      if (!demoMode) {
        workflowWebSocketService.disconnect(workflowId, handleWebSocketUpdate);
      }
    };
  }, [workflowId, fetchWorkflowData, handleWebSocketUpdate, demoMode]);

  if (loading) {
    return (
      <Card title="工作流可视化" className="h-96">
        <div className="flex items-center justify-center h-full">
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="工作流可视化" className="h-96">
        <Alert
          message="加载工作流失败"
          description={error}
          variant="destructive"
          showIcon
        />
      </Card>
    );
  }

  // 模拟执行日志数据
  const mockExecutionLogs = selectedNode ? [
    {
      timestamp: '2025-01-01 10:00:00',
      message: '节点开始执行',
      level: 'info' as const
    },
    {
      timestamp: '2025-01-01 10:01:00',
      message: '数据预处理完成',
      level: 'info' as const
    },
    {
      timestamp: '2025-01-01 10:02:00',
      message: '检测到潜在问题，继续执行',
      level: 'warning' as const
    },
  ] : [];

  return (
    <>
      <Card title={`工作流可视化 - ${workflowId}`} className="h-96" bodyStyle={{ height: '320px', padding: '0' }}>
        <div className="h-full">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={handleNodeClick}
            nodeTypes={nodeTypes}
            connectionMode={ConnectionMode.Loose}
            fitView
            attributionPosition="bottom-left"
          >
            <Background />
            <Controls />
            <MiniMap 
              nodeColor={(node) => {
                const status = (node.data as WorkflowState)?.status || 'pending';
                switch (status) {
                  case 'running': return '#1890ff';
                  case 'completed': return '#52c41a';
                  case 'failed': return '#ff4d4f';
                  case 'paused': return '#faad14';
                  default: return '#d9d9d9';
                }
              }}
            />
            <Panel position="top-right">
              <div className="bg-white p-2 rounded shadow text-sm">
                <div className="mb-2">
                  <button 
                    onClick={() => setDebugPanelVisible(true)}
                    className="px-2 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600"
                  >
                    🐛 调试
                  </button>
                </div>
                <div><span className="inline-block w-3 h-3 bg-gray-400 mr-2"></span>待执行</div>
                <div><span className="inline-block w-3 h-3 bg-blue-500 mr-2"></span>执行中</div>
                <div><span className="inline-block w-3 h-3 bg-green-500 mr-2"></span>已完成</div>
                <div><span className="inline-block w-3 h-3 bg-red-500 mr-2"></span>失败</div>
                <div><span className="inline-block w-3 h-3 bg-yellow-500 mr-2"></span>暂停</div>
              </div>
            </Panel>
          </ReactFlow>
        </div>
      </Card>
      
      <NodeDetailPanel
        visible={detailPanelVisible}
        onClose={handleCloseDetailPanel}
        nodeData={selectedNode}
        executionLogs={mockExecutionLogs}
        onNodeAction={handleNodeAction}
      />
      
      <WorkflowDebugPanel
        visible={debugPanelVisible}
        onClose={() => setDebugPanelVisible(false)}
        workflowId={workflowId}
        stateHistory={stateHistory}
        executionSteps={executionSteps}
        currentState={workflowData}
      />
    </>
  );
};

export default WorkflowVisualization;