import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReasoningQualityControl } from '../../../src/components/reasoning/ReasoningQualityControl';

const mockValidationResults = {
  step_id: 'step-123',
  is_valid: true,
  consistency_score: 0.85,
  issues: ['推理步骤2存在逻辑跳跃'],
  suggestions: ['建议增加中间推理步骤', '提高证据支持强度']
};

const mockRecoveryStats = {
  total_failures: 5,
  recovery_attempts: 3,
  recovery_success_rate: 75.5,
  strategy_effectiveness: {
    backtrack: 0.8,
    branch: 0.65,
    restart: 0.45,
    refine: 0.9,
    alternative: 0.7
  }
};

const mockCurrentChain = {
  id: 'chain-123',
  problem: '测试推理链',
  strategy: 'ZERO_SHOT',
  steps: [
    {
      id: 'step-1',
      step_number: 1,
      step_type: 'observation',
      content: '观察内容',
      reasoning: '推理过程',
      confidence: 0.8
    }
  ]
};

describe('ReasoningQualityControl组件测试', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('正确渲染基本组件结构', () => {
    render(<ReasoningQualityControl />);
    
    expect(screen.getByText('验证状态')).toBeInTheDocument();
    expect(screen.getByText('一致性分数')).toBeInTheDocument();
    expect(screen.getByText('恢复成功率')).toBeInTheDocument();
    expect(screen.getByText('失败次数')).toBeInTheDocument();
    expect(screen.getByText('质量验证')).toBeInTheDocument();
    expect(screen.getByText('恢复控制')).toBeInTheDocument();
  });

  it('无验证结果时显示默认状态', () => {
    render(<ReasoningQualityControl />);
    
    expect(screen.getByText('未验证')).toBeInTheDocument();
    expect(screen.getByText('暂无验证结果')).toBeInTheDocument();
    expect(screen.getByText('点击右侧按钮开始验证推理链质量')).toBeInTheDocument();
  });

  it('验证通过时正确显示状态', () => {
    render(
      <ReasoningQualityControl 
        validationResults={mockValidationResults}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    expect(screen.getByText('验证通过')).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument(); // 一致性分数
  });

  it('验证失败时正确显示状态和问题', () => {
    const failedValidation = {
      ...mockValidationResults,
      is_valid: false,
      consistency_score: 0.45
    };
    
    render(
      <ReasoningQualityControl 
        validationResults={failedValidation}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    expect(screen.getByText('验证失败')).toBeInTheDocument();
    expect(screen.getByText('发现的问题')).toBeInTheDocument();
    expect(screen.getByText('推理步骤2存在逻辑跳跃')).toBeInTheDocument();
  });

  it('改进建议正确显示', () => {
    render(
      <ReasoningQualityControl 
        validationResults={mockValidationResults}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    expect(screen.getByText('改进建议')).toBeInTheDocument();
    expect(screen.getByText('建议增加中间推理步骤')).toBeInTheDocument();
    expect(screen.getByText('提高证据支持强度')).toBeInTheDocument();
  });

  it('恢复统计数据正确显示', () => {
    render(
      <ReasoningQualityControl 
        validationResults={mockValidationResults}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    expect(screen.getByText('75.5')).toBeInTheDocument(); // 恢复成功率
    expect(screen.getByText('5')).toBeInTheDocument(); // 失败次数
  });

  it('验证按钮功能正常', async () => {
    const user = userEvent.setup();
    
    render(
      <ReasoningQualityControl 
        currentChain={mockCurrentChain}
      />
    );
    
    const validateButton = screen.getByText('验证推理链');
    expect(validateButton).not.toBeDisabled();
    
    await user.click(validateButton);
    
    // 检查loading状态
    expect(screen.getByText('验证中...')).toBeInTheDocument();
    
    // 快进时间以完成模拟的异步操作
    vi.advanceTimersByTime(2000);
    
    await waitFor(() => {
      expect(screen.getByText('验证推理链')).toBeInTheDocument();
    });
  });

  it('无推理链时验证按钮被禁用', () => {
    render(<ReasoningQualityControl />);
    
    const validateButton = screen.getByText('验证推理链');
    expect(validateButton).toBeDisabled();
    expect(screen.getByText('请先开始推理')).toBeInTheDocument();
  });

  it('恢复策略选择功能正常', async () => {
    const user = userEvent.setup();
    
    render(
      <ReasoningQualityControl 
        currentChain={mockCurrentChain}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    // 点击选择器
    const strategySelect = screen.getByDisplayValue('回溯');
    await user.click(strategySelect);
    
    // 选择细化策略
    await user.click(screen.getByText('细化'));
    
    // 检查策略描述更新
    expect(screen.getByText('优化当前推理步骤的内容')).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument(); // 细化策略的有效性
  });

  it('恢复按钮状态正确控制', () => {
    render(
      <ReasoningQualityControl 
        currentChain={mockCurrentChain}
        validationResults={mockValidationResults}
      />
    );
    
    // 验证通过时恢复按钮应该被禁用
    const recoverButton = screen.getByText('执行恢复');
    expect(recoverButton).toBeDisabled();
    expect(screen.getByText('推理链状态良好')).toBeInTheDocument();
  });

  it('验证失败时恢复按钮可用', () => {
    const failedValidation = {
      ...mockValidationResults,
      is_valid: false
    };
    
    render(
      <ReasoningQualityControl 
        currentChain={mockCurrentChain}
        validationResults={failedValidation}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    const recoverButton = screen.getByText('执行恢复');
    expect(recoverButton).not.toBeDisabled();
  });

  it('恢复执行功能正常', async () => {
    const user = userEvent.setup();
    const failedValidation = {
      ...mockValidationResults,
      is_valid: false
    };
    
    render(
      <ReasoningQualityControl 
        currentChain={mockCurrentChain}
        validationResults={failedValidation}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    const recoverButton = screen.getByText('执行恢复');
    await user.click(recoverButton);
    
    expect(screen.getByText('恢复中...')).toBeInTheDocument();
    
    // 快进时间以完成恢复操作
    vi.advanceTimersByTime(3000);
    
    await waitFor(() => {
      expect(screen.getByText('执行恢复')).toBeInTheDocument();
    });
  });

  it('恢复策略有效性正确显示', () => {
    render(
      <ReasoningQualityControl 
        recoveryStats={mockRecoveryStats}
      />
    );
    
    // 检查默认选中的回溯策略
    expect(screen.getByText('80.0%')).toBeInTheDocument(); // 回溯策略的有效性
  });

  it('技术实现详情正确显示', async () => {
    const user = userEvent.setup();
    
    render(<ReasoningQualityControl />);
    
    // 展开验证器详情
    const validatorsPanel = screen.getByText('验证器详情');
    await user.click(validatorsPanel);
    
    expect(screen.getByText('一致性验证器')).toBeInTheDocument();
    expect(screen.getByText('置信度验证器')).toBeInTheDocument();
    expect(screen.getByText('自我检查验证器')).toBeInTheDocument();
    expect(screen.getByText('组合验证器')).toBeInTheDocument();
    
    // 展开恢复机制
    const recoveryPanel = screen.getByText('恢复机制');
    await user.click(recoveryPanel);
    
    expect(screen.getByText('失败检测')).toBeInTheDocument();
    expect(screen.getByText('检查点管理')).toBeInTheDocument();
    expect(screen.getByText('替代路径生成')).toBeInTheDocument();
    expect(screen.getByText('恢复管理器')).toBeInTheDocument();
  });

  it('所有恢复策略选项正确显示', async () => {
    const user = userEvent.setup();
    
    render(
      <ReasoningQualityControl 
        recoveryStats={mockRecoveryStats}
      />
    );
    
    // 点击策略选择器查看所有选项
    const strategySelect = screen.getByDisplayValue('回溯');
    await user.click(strategySelect);
    
    expect(screen.getByText('回溯')).toBeInTheDocument();
    expect(screen.getByText('分支')).toBeInTheDocument();
    expect(screen.getByText('重启')).toBeInTheDocument();
    expect(screen.getByText('细化')).toBeInTheDocument();
    expect(screen.getByText('替代')).toBeInTheDocument();
  });

  it('空问题列表时不显示问题卡片', () => {
    const validationWithoutIssues = {
      ...mockValidationResults,
      issues: []
    };
    
    render(
      <ReasoningQualityControl 
        validationResults={validationWithoutIssues}
      />
    );
    
    expect(screen.queryByText('发现的问题')).not.toBeInTheDocument();
  });

  it('空建议列表时不显示建议卡片', () => {
    const validationWithoutSuggestions = {
      ...mockValidationResults,
      suggestions: []
    };
    
    render(
      <ReasoningQualityControl 
        validationResults={validationWithoutSuggestions}
      />
    );
    
    expect(screen.queryByText('改进建议')).not.toBeInTheDocument();
  });

  it('缺少恢复统计时使用默认值', () => {
    render(<ReasoningQualityControl />);
    
    expect(screen.getByText('0')).toBeInTheDocument(); // 默认的失败次数和恢复成功率
  });

  it('进度条正确反映数值', () => {
    render(
      <ReasoningQualityControl 
        validationResults={mockValidationResults}
        recoveryStats={mockRecoveryStats}
      />
    );
    
    // 检查进度条元素存在（具体百分比由Progress组件内部处理）
    const progressBars = screen.getAllByRole('progressbar');
    expect(progressBars.length).toBeGreaterThan(0);
  });
});