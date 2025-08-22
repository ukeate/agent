import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ReasoningInput } from '../../../src/components/reasoning/ReasoningInput';
import { useReasoningStore } from '../../../src/stores/reasoningStore';

// Mock Zustand store
vi.mock('../../../src/stores/reasoningStore');

const mockExecuteReasoning = vi.fn();
const mockStreamReasoning = vi.fn();
const mockClearStreamingSteps = vi.fn();

describe('ReasoningInput组件测试', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    (useReasoningStore as any).mockReturnValue({
      executeReasoning: mockExecuteReasoning,
      streamReasoning: mockStreamReasoning,
      isExecuting: false,
      clearStreamingSteps: mockClearStreamingSteps
    });
  });

  it('正确渲染基本组件', () => {
    render(<ReasoningInput />);
    
    expect(screen.getByLabelText('推理问题')).toBeInTheDocument();
    expect(screen.getByLabelText('推理策略')).toBeInTheDocument();
    expect(screen.getByText('开始推理')).toBeInTheDocument();
  });

  it('必填字段验证工作正常', async () => {
    const user = userEvent.setup();
    render(<ReasoningInput />);
    
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(screen.getByText('请输入要推理的问题')).toBeInTheDocument();
    });
  });

  it('成功提交Zero-shot推理请求', async () => {
    const user = userEvent.setup();
    const onReasoningStart = vi.fn();
    const onReasoningComplete = vi.fn();
    
    mockExecuteReasoning.mockResolvedValue({ id: 'test-chain-123' });
    
    render(
      <ReasoningInput 
        onReasoningStart={onReasoningStart}
        onReasoningComplete={onReasoningComplete}
      />
    );
    
    // 输入问题
    const problemInput = screen.getByLabelText('推理问题');
    await user.type(problemInput, '测试推理问题');
    
    // 提交
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(mockClearStreamingSteps).toHaveBeenCalled();
      expect(mockExecuteReasoning).toHaveBeenCalledWith({
        problem: '测试推理问题',
        strategy: 'ZERO_SHOT',
        context: undefined,
        max_steps: 10,
        stream: false,
        enable_branching: true,
        examples: []
      });
      expect(onReasoningStart).toHaveBeenCalled();
      expect(onReasoningComplete).toHaveBeenCalledWith('test-chain-123');
    });
  });

  it('流式推理配置正确传递', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    // 启用流式输出
    const streamSwitch = screen.getByRole('switch', { name: '流式输出' });
    await user.click(streamSwitch);
    
    // 输入问题
    const problemInput = screen.getByLabelText('推理问题');
    await user.type(problemInput, '流式推理测试');
    
    // 提交
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(mockStreamReasoning).toHaveBeenCalledWith({
        problem: '流式推理测试',
        strategy: 'ZERO_SHOT',
        context: undefined,
        max_steps: 10,
        stream: true,
        enable_branching: true,
        examples: []
      });
    });
  });

  it('策略选择功能正常', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    // 选择Few-shot策略
    const strategySelect = screen.getByLabelText('推理策略');
    await user.click(strategySelect);
    await user.click(screen.getByText('Few-shot CoT'));
    
    // 输入问题
    const problemInput = screen.getByLabelText('推理问题');
    await user.type(problemInput, 'Few-shot测试');
    
    // 提交
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(mockExecuteReasoning).toHaveBeenCalledWith(
        expect.objectContaining({
          strategy: 'FEW_SHOT'
        })
      );
    });
  });

  it('最大步骤数配置工作正常', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    // 修改最大步骤数
    const maxStepsInput = screen.getByLabelText('最大推理步骤');
    await user.clear(maxStepsInput);
    await user.type(maxStepsInput, '15');
    
    // 输入问题
    const problemInput = screen.getByLabelText('推理问题');
    await user.type(problemInput, '步骤数测试');
    
    // 提交
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(mockExecuteReasoning).toHaveBeenCalledWith(
        expect.objectContaining({
          max_steps: 15
        })
      );
    });
  });

  it('示例问题点击功能正常', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    // 展开示例问题
    const mathPanel = screen.getByText('数学推理');
    await user.click(mathPanel);
    
    // 点击第一个示例问题
    const exampleProblem = screen.getByText(/一个水池有两个进水管/);
    await user.click(exampleProblem);
    
    // 检查问题是否填入
    const problemInput = screen.getByLabelText('推理问题');
    expect(problemInput).toHaveValue(expect.stringContaining('一个水池有两个进水管'));
  });

  it('高级设置显示/隐藏功能正常', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    // 默认高级设置不显示
    expect(screen.queryByText('置信度阈值')).not.toBeInTheDocument();
    
    // 显示高级设置
    const advancedButton = screen.getByText('显示高级设置');
    await user.click(advancedButton);
    
    expect(screen.getByText('置信度阈值')).toBeInTheDocument();
    expect(screen.getByText('超时时间（秒）')).toBeInTheDocument();
    
    // 隐藏高级设置
    const hideButton = screen.getByText('隐藏高级设置');
    await user.click(hideButton);
    
    expect(screen.queryByText('置信度阈值')).not.toBeInTheDocument();
  });

  it('执行状态正确反映在UI中', () => {
    (useReasoningStore as any).mockReturnValue({
      executeReasoning: mockExecuteReasoning,
      streamReasoning: mockStreamReasoning,
      isExecuting: true,
      clearStreamingSteps: mockClearStreamingSteps
    });
    
    render(<ReasoningInput />);
    
    const submitButton = screen.getByText('推理进行中...');
    expect(submitButton).toBeDisabled();
  });

  it('错误处理正常工作', async () => {
    const user = userEvent.setup();
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    
    mockExecuteReasoning.mockRejectedValue(new Error('推理执行失败'));
    
    render(<ReasoningInput />);
    
    // 输入问题并提交
    const problemInput = screen.getByLabelText('推理问题');
    await user.type(problemInput, '错误测试');
    
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('推理执行失败:', expect.any(Error));
    });
    
    consoleSpy.mockRestore();
  });

  it('背景信息正确传递', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    // 输入问题和背景信息
    const problemInput = screen.getByLabelText('推理问题');
    await user.type(problemInput, '测试问题');
    
    const contextInput = screen.getByLabelText('背景信息（可选）');
    await user.type(contextInput, '相关背景信息');
    
    // 提交
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(mockExecuteReasoning).toHaveBeenCalledWith(
        expect.objectContaining({
          problem: '测试问题',
          context: '相关背景信息'
        })
      );
    });
  });

  it('分支功能开关正常工作', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    // 关闭分支功能
    const branchingSwitch = screen.getByRole('switch', { name: '启用分支' });
    await user.click(branchingSwitch);
    
    // 输入问题并提交
    const problemInput = screen.getByLabelText('推理问题');
    await user.type(problemInput, '分支测试');
    
    const submitButton = screen.getByText('开始推理');
    await user.click(submitButton);
    
    await waitFor(() => {
      expect(mockExecuteReasoning).toHaveBeenCalledWith(
        expect.objectContaining({
          enable_branching: false
        })
      );
    });
  });

  it('策略说明卡片正确显示', () => {
    render(<ReasoningInput />);
    
    expect(screen.getByText('Zero-shot CoT')).toBeInTheDocument();
    expect(screen.getByText('Few-shot CoT')).toBeInTheDocument();
    expect(screen.getByText('Auto-CoT')).toBeInTheDocument();
    
    expect(screen.getByText('直接使用"让我们一步一步思考"提示')).toBeInTheDocument();
    expect(screen.getByText('提供示例来指导推理过程')).toBeInTheDocument();
    expect(screen.getByText('自动选择最佳推理策略')).toBeInTheDocument();
  });

  it('表单字符数限制正常工作', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningInput />);
    
    const problemInput = screen.getByLabelText('推理问题');
    const longText = 'a'.repeat(1001); // 超过maxLength
    
    await user.type(problemInput, longText);
    
    // Ant Design的maxLength会自动截断输入
    expect(problemInput).toHaveValue('a'.repeat(1000));
  });
});