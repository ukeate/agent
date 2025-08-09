export interface WorkflowNode {
  id: string;
  name: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused' | 'skipped';
  dependencies?: string[];
  data?: any;
  error?: string;
  startedAt?: string;
  completedAt?: string;
}

export interface WorkflowEdge {
  source: string;
  target: string;
  condition?: string;
}

export interface WorkflowProgress {
  total: number;
  completed: number;
  failed: number;
  percentage: number;
}

export interface WorkflowState {
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused' | 'cancelled';
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  progress: WorkflowProgress;
  currentNodeId?: string;
  context?: Record<string, any>;
  metadata?: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}