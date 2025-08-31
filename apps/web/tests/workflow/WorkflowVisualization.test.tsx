import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { WorkflowVisualization } from '@/components/workflow/WorkflowVisualization';

describe('WorkflowVisualization', () => {

  it('应该渲染工作流可视化组件', () => {
    render(<WorkflowVisualization workflowId="test-workflow-1" />);
    
    expect(screen.getByText('Workflow Visualization')).toBeInTheDocument();
    expect(screen.getByText(/Workflow visualization for: test-workflow-1/)).toBeInTheDocument();
  });

  it('应该显示开发中提示', () => {
    render(<WorkflowVisualization workflowId="test-workflow-1" />);
    
    expect(screen.getByText('This component is under development.')).toBeInTheDocument();
  });

  it('应该处理节点点击事件', () => {
    const onNodeClick = mockFn()fn();
    render(<WorkflowVisualization workflowId="test-workflow-1" onNodeClick={onNodeClick} />);
    
    const clickButton = screen.getByText('Test Node Click');
    fireEvent.click(clickButton);

    expect(onNodeClick).toHaveBeenCalledWith('sample-node');
  });
});