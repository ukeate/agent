import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { WorkflowVisualization } from '@/components/workflow/WorkflowVisualization';
import { WorkflowState } from '@/types/workflow';

// Mock fetch
global.fetch = vi.fn();

const mockWorkflowState: WorkflowState = {
  workflowId: 'test-workflow-1',
  status: 'running',
  nodes: [
    {
      id: 'start',
      name: '开始节点',
      type: 'start',
      status: 'completed'
    },
    {
      id: 'process',
      name: '处理节点',
      type: 'process',
      status: 'running',
      dependencies: ['start']
    },
    {
      id: 'end',
      name: '结束节点',
      type: 'end',
      status: 'pending',
      dependencies: ['process']
    }
  ],
  edges: [
    { source: 'start', target: 'process' },
    { source: 'process', target: 'end' }
  ],
  progress: {
    total: 3,
    completed: 1,
    failed: 0,
    percentage: 33
  },
  createdAt: '2025-08-05T10:00:00Z',
  updatedAt: '2025-08-05T10:01:00Z'
};

describe('WorkflowVisualization', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockWorkflowState
    });
  });

  it('应该渲染工作流可视化组件', async () => {
    render(<WorkflowVisualization workflowId="test-workflow-1" />);
    
    await waitFor(() => {
      expect(screen.getByText('工作流可视化')).toBeInTheDocument();
      expect(screen.getByText(/工作流ID: test-workflow-1/)).toBeInTheDocument();
    });
  });

  it('应该显示工作流状态', async () => {
    render(<WorkflowVisualization workflowId="test-workflow-1" />);
    
    await waitFor(() => {
      expect(screen.getByText('running')).toBeInTheDocument();
    });
  });

  it('应该显示进度条', async () => {
    render(<WorkflowVisualization workflowId="test-workflow-1" />);
    
    await waitFor(() => {
      expect(screen.getByText('进度: 1/3 节点')).toBeInTheDocument();
    });
  });

  it('应该处理API错误', async () => {
    (global.fetch as any).mockResolvedValue({
      ok: false,
      status: 500
    });

    render(<WorkflowVisualization workflowId="test-workflow-1" />);
    
    await waitFor(() => {
      expect(screen.getByText('获取工作流状态失败')).toBeInTheDocument();
    });
  });

  it('应该定期刷新工作流状态', async () => {
    render(<WorkflowVisualization workflowId="test-workflow-1" />);
    
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/v1/workflows/test-workflow-1/status');
    });

    // 等待自动刷新
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(2);
    }, { timeout: 3000 });
  });

  it('应该处理节点点击事件', async () => {
    const onNodeClick = vi.fn();
    render(<WorkflowVisualization workflowId="test-workflow-1" onNodeClick={onNodeClick} />);
    
    await waitFor(() => {
      const canvas = screen.getByRole('img'); // Canvas元素
      fireEvent.click(canvas, { clientX: 100, clientY: 100 });
    });

    // 验证是否调用了回调
    expect(onNodeClick).toHaveBeenCalled();
  });
});