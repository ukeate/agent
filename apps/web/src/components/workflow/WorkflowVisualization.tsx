import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import { logger } from '../../utils/logger'
import React, { useCallback, useEffect, useRef, useState } from 'react';
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
}

export const WorkflowVisualization: React.FC<WorkflowVisualizationProps> = ({
  workflowId,
  onNodeClick = () => {},
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const nodesRef = useRef<Node[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [workflowData, setWorkflowData] = useState<any>(null);
  const [selectedNode, setSelectedNode] = useState<WorkflowState | null>(null);
  const [detailPanelVisible, setDetailPanelVisible] = useState(false);
  const [debugPanelVisible, setDebugPanelVisible] = useState(false);
  const [stateHistory, setStateHistory] = useState<any[]>([]);
  const [executionSteps, setExecutionSteps] = useState<any[]>([]);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);
  // 从API获取工作流数据
  const fetchWorkflowData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiFetch(buildApiUrl(`/api/v1/workflows/${workflowId}`));
      const data = await response.json();
      setWorkflowData(data);
      
      const executionLog = data.current_state?.context?.execution_log || [];
      const definition = data.definition || data.workflow_definition || null;

      const resolveNodeType = (nodeId: string, rawType?: string) => {
        const type = (rawType || '').toLowerCase();
        if (nodeId === 'start' || type.includes('start')) return 'start';
        if (nodeId === 'end' || type.includes('end') || type.includes('finish')) return 'end';
        if (type.includes('decision') || type.includes('conditional')) return 'decision';
        return 'process';
      };

      const buildGraph = () => {
        const nameMap: Record<string, string> = {};
        let nodes: Node[] = [];
        let edges: Edge[] = [];

        const buildEdges = (rawEdges: any[]) => {
          return rawEdges.map((edge, index) => {
            const source = edge.source || edge.from;
            const target = edge.target || edge.to;
            const label = edge.label || edge.condition;
            let sourceHandle: string | undefined;
            if (source === 'decision' || edge.decision) {
              const labelText = String(label || '').toLowerCase();
              if (labelText.includes('high') || labelText.includes('path_a') || labelText.includes('a')) {
                sourceHandle = 'high';
              } else if (labelText.includes('low') || labelText.includes('path_b') || labelText.includes('b')) {
                sourceHandle = 'low';
              }
            }
            return {
              id: edge.id || `e-${source}-${target}-${index}`,
              source,
              target,
              label,
              sourceHandle,
              animated: Boolean(edge.animated),
            };
          });
        };

        if (definition?.nodes && definition?.edges) {
          nodes = definition.nodes.map((node: any, index: number) => {
            const nodeId = String(node.id || node.node_id || `node_${index + 1}`);
            const nodeName = node.name || node.label || nodeId;
            nameMap[nodeId] = nodeName;
            const position = node.position || { x: 100 + index * 200, y: 100 };
            const nodeType = resolveNodeType(nodeId, node.type || node.step_type);
            return {
              id: nodeId,
              type: 'workflowNode',
              position,
              data: { id: nodeId, name: nodeName, status: 'pending', type: nodeType },
            };
          });
          edges = buildEdges(definition.edges);
        } else if (definition?.steps) {
          nodes = definition.steps.map((step: any, index: number) => {
            const nodeId = String(step.id || `step_${index + 1}`);
            const nodeName = step.name || nodeId;
            nameMap[nodeId] = nodeName;
            const nodeType = resolveNodeType(nodeId, step.step_type);
            return {
              id: nodeId,
              type: 'workflowNode',
              position: { x: 100 + index * 200, y: 100 },
              data: { id: nodeId, name: nodeName, status: 'pending', type: nodeType },
            };
          });
          const rawEdges: any[] = [];
          definition.steps.forEach((step: any) => {
            const deps = step.dependencies || [];
            deps.forEach((dep: string) => rawEdges.push({ from: dep, to: step.id }));
          });
          edges = buildEdges(rawEdges);
        }

        return { nodes, edges, nameMap };
      };

      const { nodes: baseNodes, edges: baseEdges, nameMap } = buildGraph();
      if (!baseNodes.length) {
        setNodes([]);
        setEdges([]);
        throw new Error('工作流定义为空');
      }

      const nodeStatusMap: { [key: string]: string } = {};
      baseNodes.forEach((node) => {
        nodeStatusMap[node.id] = 'pending';
      });

      executionLog.forEach((log: any) => {
        const nodeId = String(log?.node || '');
        if (!nodeId || nodeStatusMap[nodeId] === undefined) return;
        nodeStatusMap[nodeId] = log?.status === 'failed' ? 'failed' : 'completed';
      });
      
      if (data.status === 'running') {
        const currentNode = data.current_state?.metadata?.current_node;
        if (currentNode && nodeStatusMap[currentNode] === 'pending') {
          nodeStatusMap[currentNode] = 'running';
        }
      }
      
      const nextNodes = baseNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          status: nodeStatusMap[node.id] || 'pending',
        },
      }));

      setNodes(nextNodes);
      setEdges(baseEdges);

        const history: any[] = [];
        const steps: any[] = [];
        const statusMap: Record<string, string> = {};
        Object.keys(nameMap).forEach((k) => (statusMap[k] = 'pending'));
        let prevTs: number | null = null;

        for (let i = 0; i < executionLog.length; i++) {
          const log = executionLog[i];
          const nodeId = String(log?.node || '');
          if (!nodeId) continue;
          const ts = String(log?.timestamp || '');
          const nextStatus = log?.status === 'failed' ? 'failed' : 'completed';
          const prevStatus = statusMap[nodeId] || 'pending';
          statusMap[nodeId] = nextStatus;

          history.push({
            timestamp: ts,
            nodeId,
            nodeName: nameMap[nodeId] || nodeId,
            previousStatus: prevStatus,
            newStatus: nextStatus,
            metadata: log,
          });

          const t = Date.parse(ts);
          const duration = Number.isFinite(t) && prevTs !== null ? Math.max(0, t - prevTs) : 0;
          if (Number.isFinite(t)) prevTs = t;
          steps.push({
            stepId: `${workflowId}-${i + 1}`,
            timestamp: ts,
            nodeId,
            nodeName: nameMap[nodeId] || nodeId,
            action: 'execute',
            duration,
            status: log?.status === 'failed' ? 'error' : 'success',
            details: log,
          });
        }

        setStateHistory(history);
        setExecutionSteps(steps);
	      
	    } catch (err) {
	      logger.error('获取工作流数据失败:', err);
	      setError(err instanceof Error ? err.message : '未知错误');
	    } finally {
	      setLoading(false);
	    }
	  }, [workflowId]);

  // 处理节点点击事件
  const handleNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    logger.log('节点已点击:', node);
    const nodeData = node.data as WorkflowState;
    setSelectedNode(nodeData);
    setDetailPanelVisible(true);
    onNodeClick(node.id, nodeData);
  }, [onNodeClick]);

  // 处理节点操作
  const handleNodeAction = useCallback(async (action: string, nodeId: string) => {
    try {
      const control = async (ctrl: 'pause' | 'resume' | 'cancel') => {
        await apiFetch(buildApiUrl(`/api/v1/workflows/${workflowId}/control`), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: ctrl }),
        });
      };

      if (action === 'pause' || action === 'resume') {
        await control(action);
      } else if (action === 'cancel') {
        await control('cancel');
      } else if (action === 'restart') {
        await apiFetch(buildApiUrl(`/api/v1/workflows/${workflowId}/start`), { method: 'POST' });
      } else {
        throw new Error('不支持的操作');
      }

      await fetchWorkflowData();
      message.success('操作成功');
    } catch (err) {
      logger.error('操作失败:', err);
      message.error('操作失败，请重试');
    }
  }, [workflowId, fetchWorkflowData]);

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
    if (message.type === 'initial_status' || message.type === 'status_update') {
      const data = message.data;
      if (!data) return;
      setWorkflowData(data);

      const executionLog = data.current_state?.context?.execution_log || [];
      const nodeStatusMap: Record<string, string> = {};
      const nameMap: Record<string, string> = {};
      nodesRef.current.forEach((node) => {
        nodeStatusMap[node.id] = 'pending';
        const data = node.data as WorkflowState | undefined;
        nameMap[node.id] = data?.name || node.id;
      });

      executionLog.forEach((log: any) => {
        const node = String(log?.node || '');
        if (!node) return;
        if (log?.status === 'failed') nodeStatusMap[node] = 'failed';
        else nodeStatusMap[node] = 'completed';
      });

      const currentNode = data.current_state?.metadata?.current_node;
      if (data.status === 'running' && currentNode && nodeStatusMap[currentNode] === 'pending') {
        nodeStatusMap[currentNode] = 'running';
      }
      if (data.status === 'paused' && currentNode && nodeStatusMap[currentNode] === 'pending') {
        nodeStatusMap[currentNode] = 'paused';
      }

      setNodes((nds) =>
        nds.map((node) => ({
          ...node,
          data: { ...node.data, status: nodeStatusMap[node.id] || node.data.status },
        }))
      );

      const history: any[] = [];
      const steps: any[] = [];
      const statusMap: Record<string, string> = {};
      Object.keys(nameMap).forEach((k) => (statusMap[k] = 'pending'));
      let prevTs: number | null = null;

      for (let i = 0; i < executionLog.length; i++) {
        const log = executionLog[i];
        const nodeId = String(log?.node || '');
        if (!nodeId) continue;
        const ts = String(log?.timestamp || '');
        const nextStatus = log?.status === 'failed' ? 'failed' : 'completed';
        const prevStatus = statusMap[nodeId] || 'pending';
        statusMap[nodeId] = nextStatus;

        history.push({
          timestamp: ts,
          nodeId,
          nodeName: nameMap[nodeId] || nodeId,
          previousStatus: prevStatus,
          newStatus: nextStatus,
          metadata: log,
        });

        const t = Date.parse(ts);
        const duration = Number.isFinite(t) && prevTs !== null ? Math.max(0, t - prevTs) : 0;
        if (Number.isFinite(t)) prevTs = t;
        steps.push({
          stepId: `${workflowId}-${i + 1}`,
          timestamp: ts,
          nodeId,
          nodeName: nameMap[nodeId] || nodeId,
          action: 'execute',
          duration,
          status: log?.status === 'failed' ? 'error' : 'success',
          details: log,
        });
      }

      setStateHistory(history);
      setExecutionSteps(steps);
    }
  }, [setNodes, workflowId]);

  useEffect(() => {
    fetchWorkflowData();
    
    workflowWebSocketService.connect(workflowId, handleWebSocketUpdate);
    
    // 清理函数
    return () => {
      workflowWebSocketService.disconnect(workflowId, handleWebSocketUpdate);
    };
  }, [workflowId, fetchWorkflowData, handleWebSocketUpdate]);

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
	          type="error"
	          showIcon
	        />
	      </Card>
	    );
	  }

  const executionLogs = selectedNode
    ? (workflowData?.current_state?.context?.execution_log || [])
        .filter((log: any) => log?.node === selectedNode.id)
        .map((log: any) => ({
          timestamp: String(log?.timestamp || ''),
          message: log?.error ? `执行失败: ${log.error}` : `节点执行${log?.status || ''}`,
          level: (log?.status === 'failed' ? 'error' : log?.error ? 'error' : 'info') as const,
        }))
    : [];

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
        executionLogs={executionLogs}
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
