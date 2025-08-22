import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import NodeDetailPanel from '../../src/components/workflow/NodeDetailPanel';

// Mock window.getComputedStyle for JSDOM
Object.defineProperty(window, 'getComputedStyle', {
  value: () => ({
    getPropertyValue: () => '',
  }),
});

describe('NodeDetailPanel', () => {
  const mockNodeData = {
    id: 'test-node',
    name: '测试节点',
    status: 'running' as const,
    type: 'process' as const,
  };

  const mockExecutionLogs = [
    {
      timestamp: '2025-01-01 10:00:00',
      message: '节点开始执行',
      level: 'info' as const,
    },
    {
      timestamp: '2025-01-01 10:01:00',
      message: '数据处理完成',
      level: 'info' as const,
    },
    {
      timestamp: '2025-01-01 10:02:00',
      message: '警告：检测到潜在问题',
      level: 'warning' as const,
    },
  ];

  const defaultProps = {
    visible: true,
    onClose: vi.fn(),
    nodeData: mockNodeData,
    executionLogs: mockExecutionLogs,
    onNodeAction: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('应该正确渲染节点详情面板', () => {
    render(<NodeDetailPanel {...defaultProps} />);
    
    expect(screen.getByText('节点详情 - 测试节点')).toBeInTheDocument();
    expect(screen.getByText('test-node')).toBeInTheDocument();
    expect(screen.getByText('测试节点')).toBeInTheDocument();
    expect(screen.getByText('处理节点')).toBeInTheDocument();
  });

  it('应该显示正确的状态徽章', () => {
    render(<NodeDetailPanel {...defaultProps} />);
    
    expect(screen.getByText('执行中')).toBeInTheDocument();
  });

  it('应该显示执行日志', () => {
    render(<NodeDetailPanel {...defaultProps} />);
    
    expect(screen.getByText('执行日志')).toBeInTheDocument();
    expect(screen.getByText('节点开始执行')).toBeInTheDocument();
    expect(screen.getByText('数据处理完成')).toBeInTheDocument();
    expect(screen.getByText('警告：检测到潜在问题')).toBeInTheDocument();
  });

  it('应该根据节点状态显示相应的操作按钮', () => {
    // 测试运行中状态 - 应该显示暂停和停止按钮
    render(<NodeDetailPanel {...defaultProps} />);
    
    expect(screen.getByText('暂停')).toBeInTheDocument();
    expect(screen.getByText('停止')).toBeInTheDocument();
    expect(screen.queryByText('恢复')).not.toBeInTheDocument();
    expect(screen.queryByText('重试')).not.toBeInTheDocument();
  });

  it('应该为暂停状态显示恢复按钮', () => {
    const pausedNodeData = { ...mockNodeData, status: 'paused' as const };
    
    render(<NodeDetailPanel {...defaultProps} nodeData={pausedNodeData} />);
    
    expect(screen.getByRole('button', { name: /恢复/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /停止/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /暂停/ })).not.toBeInTheDocument();
  });

  it('应该为失败状态显示重试按钮', () => {
    const failedNodeData = { ...mockNodeData, status: 'failed' as const };
    
    render(<NodeDetailPanel {...defaultProps} nodeData={failedNodeData} />);
    
    expect(screen.getByText(/重.*试/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '暂停' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '恢复' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '停止' })).not.toBeInTheDocument();
  });

  it('应该处理操作按钮点击事件', () => {
    render(<NodeDetailPanel {...defaultProps} />);
    
    // 点击暂停按钮
    fireEvent.click(screen.getByText('暂停'));
    expect(defaultProps.onNodeAction).toHaveBeenCalledWith('pause', 'test-node');
    
    // 点击停止按钮
    fireEvent.click(screen.getByText('停止'));
    expect(defaultProps.onNodeAction).toHaveBeenCalledWith('stop', 'test-node');
  });

  it('应该接收关闭回调函数', () => {
    render(<NodeDetailPanel {...defaultProps} />);
    
    // 验证onClose函数被正确传递
    expect(defaultProps.onClose).toBeInstanceOf(Function);
  });

  it('应该在没有执行日志时显示占位文本', () => {
    render(<NodeDetailPanel {...defaultProps} executionLogs={[]} />);
    
    expect(screen.getByText('暂无执行日志')).toBeInTheDocument();
  });

  it('应该在没有节点数据时返回null', () => {
    const { container } = render(
      <NodeDetailPanel {...defaultProps} nodeData={null} />
    );
    
    expect(container.firstChild).toBeNull();
  });

  it('应该在面板不可见时不渲染内容', () => {
    render(<NodeDetailPanel {...defaultProps} visible={false} />);
    
    // Antd的Drawer在visible=false时会隐藏内容
    expect(screen.queryByText('节点详情 - 测试节点')).not.toBeInTheDocument();
  });

  it('应该显示不同类型节点的正确描述', () => {
    const startNodeData = { ...mockNodeData, type: 'start' as const };
    const { rerender } = render(<NodeDetailPanel {...defaultProps} nodeData={startNodeData} />);
    expect(screen.getByText('开始节点')).toBeInTheDocument();

    const decisionNodeData = { ...mockNodeData, type: 'decision' as const };
    rerender(<NodeDetailPanel {...defaultProps} nodeData={decisionNodeData} />);
    expect(screen.getByText('决策节点')).toBeInTheDocument();

    const endNodeData = { ...mockNodeData, type: 'end' as const };
    rerender(<NodeDetailPanel {...defaultProps} nodeData={endNodeData} />);
    expect(screen.getByText('结束节点')).toBeInTheDocument();
  });
});