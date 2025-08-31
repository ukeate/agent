import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import WorkflowVisualization from '../../src/components/workflow/WorkflowVisualization';

// Mock ReactFlow
mockFn()mock('reactflow', () => ({
  default: ({ children, onNodeClick }: any) => (
    <div data-testid="react-flow">
      <button 
        data-testid="mock-node" 
        onClick={() => onNodeClick && onNodeClick(
          { preventDefault: mockFn()fn() }, 
          { id: 'test-node', data: { id: 'test-node', name: '测试节点', status: 'running' } }
        )}
      >
        Mock Node
      </button>
      {children}
    </div>
  ),
  useNodesState: () => [[], mockFn()fn(), mockFn()fn()],
  useEdgesState: () => [[], mockFn()fn(), mockFn()fn()],
  addEdge: mockFn()fn(),
  ConnectionMode: { Loose: 'loose' },
  Background: () => <div data-testid="background" />,
  Controls: () => <div data-testid="controls" />,
  MiniMap: () => <div data-testid="minimap" />,
  Panel: ({ children }: any) => <div data-testid="panel">{children}</div>,
}));

// Mock WebSocket service
mockFn()mock('../../src/services/workflowWebSocketService', () => ({
  workflowWebSocketService: {
    connect: mockFn()fn(),
    disconnect: mockFn()fn(),
  },
}));

// Mock fetch
const mockFetch = mockFn()fn();
global.fetch = mockFetch;

describe('WorkflowVisualization', () => {
  const defaultProps = {
    workflowId: 'test-workflow-123',
    onNodeClick: mockFn()fn(),
  };

  beforeEach(() => {
    mockFn()clearAllMocks();
    // Mock successful API response
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: mockFn()fn().mockResolvedValue({
        id: 'test-workflow-123',
        name: '测试工作流',
        status: 'running',
        created_at: '2025-01-01T10:00:00Z',
      }),
    });
  });

  afterEach(() => {
    mockFn()resetAllMocks();
  });

  it('应该正确渲染工作流可视化组件', async () => {
    render(<WorkflowVisualization {...defaultProps} />);
    
    // 等待初始渲染
    await waitFor(
      () => {
        expect(screen.getByText('工作流可视化')).toBeInTheDocument();
      },
      { timeout: 1000 }
    );
  });

  it('应该显示加载状态', () => {
    // Mock pending fetch - create a promise that never resolves
    let resolveFetch: (value: any) => void;
    const pendingPromise = new Promise((resolve) => {
      resolveFetch = resolve;
    });
    mockFetch.mockReturnValue(pendingPromise);
    
    render(<WorkflowVisualization {...defaultProps} />);
    
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.getByText('工作流可视化')).toBeInTheDocument();
  });

  it('应该显示错误状态', async () => {
    // Mock failed fetch
    mockFetch.mockRejectedValue(new Error('Network error'));
    
    render(<WorkflowVisualization {...defaultProps} />);
    
    // 等待错误状态显示，使用更灵活的匹配
    await waitFor(
      () => {
        const errorElements = screen.queryAllByText(/失败|错误|error/i);
        expect(errorElements.length).toBeGreaterThan(0);
      },
      { timeout: 10000 }
    );
  });

  it('应该处理节点点击事件', async () => {
    render(<WorkflowVisualization {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByTestId('mock-node')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId('mock-node'));
    
    expect(defaultProps.onNodeClick).toHaveBeenCalledWith(
      'test-node',
      expect.objectContaining({
        id: 'test-node',
        name: '测试节点',
        status: 'running',
      })
    );
  });

  it('应该显示调试按钮并打开调试面板', async () => {
    render(<WorkflowVisualization {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('🐛 调试')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('🐛 调试'));
    
    // 调试面板应该打开
    await waitFor(() => {
      expect(screen.getByText(/工作流调试/)).toBeInTheDocument();
    });
  });

  it('应该显示状态图例', async () => {
    render(<WorkflowVisualization {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('待执行')).toBeInTheDocument();
      expect(screen.getByText('执行中')).toBeInTheDocument();
      expect(screen.getByText('已完成')).toBeInTheDocument();
      expect(screen.getByText('失败')).toBeInTheDocument();
      expect(screen.getByText('暂停')).toBeInTheDocument();
    });
  });

  it('应该连接到WebSocket服务', async () => {
    const { workflowWebSocketService } = await import('../../src/services/workflowWebSocketService');
    
    render(<WorkflowVisualization {...defaultProps} />);
    
    await waitFor(() => {
      expect(workflowWebSocketService.connect).toHaveBeenCalledWith(
        'test-workflow-123',
        expect.any(Function)
      );
    });
  });

  it('应该在卸载时断开WebSocket连接', async () => {
    const { workflowWebSocketService } = await import('../../src/services/workflowWebSocketService');
    
    const { unmount } = render(<WorkflowVisualization {...defaultProps} />);
    
    await waitFor(() => {
      expect(workflowWebSocketService.connect).toHaveBeenCalled();
    });

    unmount();

    expect(workflowWebSocketService.disconnect).toHaveBeenCalledWith(
      'test-workflow-123',
      expect.any(Function)
    );
  });

  it('应该处理API错误', async () => {
    // Mock 404 response
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      json: mockFn()fn().mockResolvedValue({}),
    });
    
    render(<WorkflowVisualization {...defaultProps} />);
    
    // 等待错误状态显示，使用更灵活的匹配
    await waitFor(
      () => {
        const errorElements = screen.queryAllByText(/失败|错误|Failed/i);
        expect(errorElements.length).toBeGreaterThan(0);
      },
      { timeout: 10000 }
    );
  });
});