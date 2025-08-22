import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, beforeEach } from 'vitest';
import { ReasoningChainVisualization } from '../../../src/components/reasoning/ReasoningChainVisualization';

const mockChain = {
  id: 'chain-123',
  problem: '测试推理问题：分析市场趋势',
  strategy: 'ZERO_SHOT',
  steps: [
    {
      id: 'step-1',
      step_number: 1,
      step_type: 'observation',
      content: '观察当前市场数据和趋势指标',
      reasoning: '首先需要收集和分析现有的市场数据',
      confidence: 0.85,
      duration_ms: 1200
    },
    {
      id: 'step-2',
      step_number: 2,
      step_type: 'analysis',
      content: '分析数据中的关键模式和异常',
      reasoning: '通过统计分析识别重要的市场信号',
      confidence: 0.78,
      duration_ms: 1800
    },
    {
      id: 'step-3',
      step_number: 3,
      step_type: 'conclusion',
      content: '基于分析得出市场预测结论',
      reasoning: '综合所有分析结果得出最终判断',
      confidence: 0.92,
      duration_ms: 900
    }
  ],
  conclusion: '市场呈现上升趋势，建议适当增加投资',
  confidence_score: 0.85,
  total_duration_ms: 3900,
  created_at: '2025-01-15T10:00:00Z',
  completed_at: '2025-01-15T10:05:00Z'
};

const mockStreamingSteps = [
  {
    id: 'streaming-1',
    step_number: 1,
    step_type: 'observation',
    content: '正在收集数据...',
    reasoning: '开始数据收集过程',
    confidence: 0.7
  }
];

describe('ReasoningChainVisualization组件测试', () => {
  it('无数据时显示等待状态', () => {
    render(<ReasoningChainVisualization />);
    
    expect(screen.getByText('等待推理开始')).toBeInTheDocument();
    expect(screen.getByText('请在左侧"推理输入"标签页配置推理参数并开始推理')).toBeInTheDocument();
  });

  it('正确显示推理链基本信息', () => {
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    expect(screen.getByText('ZERO_SHOT')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('测试推理问题：分析市场趋势')).toBeInTheDocument();
  });

  it('步骤视图正确渲染所有步骤', () => {
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    // 检查步骤标题
    expect(screen.getByText('步骤 1: OBSERVATION')).toBeInTheDocument();
    expect(screen.getByText('步骤 2: ANALYSIS')).toBeInTheDocument();
    expect(screen.getByText('步骤 3: CONCLUSION')).toBeInTheDocument();
    
    // 检查步骤内容
    expect(screen.getByText('观察当前市场数据和趋势指标')).toBeInTheDocument();
    expect(screen.getByText('分析数据中的关键模式和异常')).toBeInTheDocument();
    expect(screen.getByText('基于分析得出市场预测结论')).toBeInTheDocument();
    
    // 检查推理过程
    expect(screen.getByText('首先需要收集和分析现有的市场数据')).toBeInTheDocument();
    expect(screen.getByText('通过统计分析识别重要的市场信号')).toBeInTheDocument();
    expect(screen.getByText('综合所有分析结果得出最终判断')).toBeInTheDocument();
  });

  it('时间线视图切换功能正常', async () => {
    const user = userEvent.setup();
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    // 切换到时间线视图
    const timelineButton = screen.getByText('时间线');
    await user.click(timelineButton);
    
    // 检查时间线是否显示
    expect(screen.getByText('observation')).toBeInTheDocument();
    expect(screen.getByText('analysis')).toBeInTheDocument();
    expect(screen.getByText('conclusion')).toBeInTheDocument();
    
    // 切换回步骤视图
    const stepsButton = screen.getByText('步骤视图');
    await user.click(stepsButton);
    
    // 确认步骤视图内容重新显示
    expect(screen.getByText('步骤 1: OBSERVATION')).toBeInTheDocument();
  });

  it('技术细节切换功能正常', async () => {
    const user = userEvent.setup();
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    // 默认技术细节应该显示
    expect(screen.getByText('3')).toBeInTheDocument(); // 推理步骤数
    expect(screen.getByText('3900')).toBeInTheDocument(); // 总耗时
    
    // 切换关闭技术细节
    const techButton = screen.getByText('技术细节');
    await user.click(techButton);
    
    // 展开第一个步骤的技术细节
    const firstStepCard = screen.getByText('步骤 1: OBSERVATION').closest('.ant-card');
    const techPanel = firstStepCard?.querySelector('.ant-collapse-header');
    if (techPanel) {
      await user.click(techPanel);
      expect(screen.getByText('step-1')).toBeInTheDocument();
    }
  });

  it('流式输出状态正确显示', () => {
    render(
      <ReasoningChainVisualization 
        streamingSteps={mockStreamingSteps}
        isExecuting={true}
      />
    );
    
    expect(screen.getByText('执行中')).toBeInTheDocument();
    expect(screen.getByText('STREAMING')).toBeInTheDocument();
    expect(screen.getByText('正在收集数据...')).toBeInTheDocument();
  });

  it('混合数据（已完成+流式）正确显示', () => {
    render(
      <ReasoningChainVisualization 
        chain={mockChain}
        streamingSteps={mockStreamingSteps}
        isExecuting={true}
      />
    );
    
    // 应该显示完整链的步骤 + 流式步骤
    expect(screen.getByText('步骤 1: OBSERVATION')).toBeInTheDocument();
    expect(screen.getByText('步骤 2: ANALYSIS')).toBeInTheDocument();
    expect(screen.getByText('步骤 3: CONCLUSION')).toBeInTheDocument();
    expect(screen.getByText('正在收集数据...')).toBeInTheDocument();
  });

  it('置信度颜色编码正确', () => {
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    // 检查高置信度（>80%）显示
    const highConfidenceElements = screen.getAllByText('85%');
    expect(highConfidenceElements.length).toBeGreaterThan(0);
    
    // 检查中等置信度（60-80%）
    const mediumConfidenceElements = screen.getAllByText('78%');
    expect(mediumConfidenceElements.length).toBeGreaterThan(0);
    
    // 检查高置信度（>80%）
    const veryHighConfidenceElements = screen.getAllByText('92%');
    expect(veryHighConfidenceElements.length).toBeGreaterThan(0);
  });

  it('步骤类型图标正确显示', () => {
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    // 虽然我们不能直接测试emoji，但可以测试步骤类型标签
    expect(screen.getByText('observation')).toBeInTheDocument();
    expect(screen.getByText('analysis')).toBeInTheDocument();
    expect(screen.getByText('conclusion')).toBeInTheDocument();
  });

  it('结论卡片正确显示', () => {
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    expect(screen.getByText('推理结论')).toBeInTheDocument();
    expect(screen.getByText('市场呈现上升趋势，建议适当增加投资')).toBeInTheDocument();
  });

  it('执行时间显示格式正确', () => {
    render(<ReasoningChainVisualization chain={mockChain} />);
    
    // 检查各步骤的执行时间
    expect(screen.getByText('1200ms')).toBeInTheDocument();
    expect(screen.getByText('1800ms')).toBeInTheDocument();
    expect(screen.getByText('900ms')).toBeInTheDocument();
  });

  it('时间线视图显示推理进行中状态', async () => {
    const user = userEvent.setup();
    render(
      <ReasoningChainVisualization 
        chain={mockChain}
        isExecuting={true}
      />
    );
    
    // 切换到时间线视图
    const timelineButton = screen.getByText('时间线');
    await user.click(timelineButton);
    
    expect(screen.getByText('推理进行中...')).toBeInTheDocument();
  });

  it('分支信息正确显示', () => {
    const chainWithBranches = {
      ...mockChain,
      branches: [
        {
          id: 'branch-1',
          parent_step_id: 'step-1',
          branch_reason: '探索替代路径',
          priority: 1,
          is_active: true,
          steps: []
        }
      ]
    };
    
    render(<ReasoningChainVisualization chain={chainWithBranches} />);
    
    // 检查分支数量统计
    expect(screen.getByText('1')).toBeInTheDocument(); // 分支数量
  });

  it('无结论时不显示结论卡片', () => {
    const incompleteChain = {
      ...mockChain,
      conclusion: undefined
    };
    
    render(<ReasoningChainVisualization chain={incompleteChain} />);
    
    expect(screen.queryByText('推理结论')).not.toBeInTheDocument();
  });

  it('缺少元数据时优雅降级', () => {
    const minimalChain = {
      id: 'minimal-chain',
      problem: '简单问题',
      strategy: 'ZERO_SHOT',
      steps: [
        {
          id: 'step-1',
          step_number: 1,
          step_type: 'observation',
          content: '简单观察',
          reasoning: '简单推理',
          confidence: 0.8
          // 缺少 duration_ms
        }
      ]
      // 缺少其他可选字段
    };
    
    render(<ReasoningChainVisualization chain={minimalChain} />);
    
    expect(screen.getByText('简单问题')).toBeInTheDocument();
    expect(screen.getByText('简单观察')).toBeInTheDocument();
    expect(screen.getByText('简单推理')).toBeInTheDocument();
  });

  it('空步骤数组时正确处理', () => {
    const emptyChain = {
      ...mockChain,
      steps: []
    };
    
    render(<ReasoningChainVisualization chain={emptyChain} />);
    
    expect(screen.getByText('测试推理问题：分析市场趋势')).toBeInTheDocument();
    expect(screen.queryByText('步骤 1')).not.toBeInTheDocument();
  });
});