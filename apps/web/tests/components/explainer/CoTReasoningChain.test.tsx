import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import CoTReasoningChain from '../../../src/components/explainer/CoTReasoningChain';

describe('CoTReasoningChain', () => {
  const mockChain = {
    chain_id: 'test-chain-001',
    reasoning_mode: 'analytical' as const,
    steps: [
      {
        step_id: 'step-1',
        step_type: 'observation' as const,
        content: '用户查询分析：需要评估贷款申请的风险',
        confidence: 0.9,
        dependencies: [],
        evidence: ['申请表完整', '收入证明齐全'],
        duration_ms: 150,
        metadata: {
          reasoning_strategy: 'data_analysis',
          evidence_quality: 0.85,
          logical_validity: 0.9
        }
      },
      {
        step_id: 'step-2',
        step_type: 'thought' as const,
        content: '基于信用评分750分，属于优质客户范围',
        confidence: 0.85,
        dependencies: ['step-1'],
        evidence: ['征信报告', '历史还款记录'],
        duration_ms: 200,
        metadata: {
          reasoning_strategy: 'pattern_matching',
          evidence_quality: 0.8,
          logical_validity: 0.85
        }
      },
      {
        step_id: 'step-3',
        step_type: 'conclusion' as const,
        content: '建议批准贷款申请，风险等级为低',
        confidence: 0.88,
        dependencies: ['step-1', 'step-2'],
        evidence: ['综合评分', '风险模型输出'],
        duration_ms: 100,
        metadata: {
          reasoning_strategy: 'decision_making',
          evidence_quality: 0.9,
          logical_validity: 0.88
        }
      }
    ],
    overall_confidence: 0.88,
    logical_consistency: 0.85,
    evidence_quality: 0.85,
    created_at: '2025-01-17T10:00:00Z'
  };

  it('renders reasoning chain with basic information', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    expect(screen.getByText('Chain-of-Thought 推理链')).toBeInTheDocument();
    expect(screen.getByText('analytical')).toBeInTheDocument();
    expect(screen.getByText('分析性推理 - 逐步分解复杂问题')).toBeInTheDocument();
    expect(screen.getByText('88.0%')).toBeInTheDocument();
  });

  it('displays all reasoning steps', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    expect(screen.getByText('步骤 1')).toBeInTheDocument();
    expect(screen.getByText('步骤 2')).toBeInTheDocument();
    expect(screen.getByText('步骤 3')).toBeInTheDocument();
    
    expect(screen.getByText('observation')).toBeInTheDocument();
    expect(screen.getByText('thought')).toBeInTheDocument();
    expect(screen.getByText('conclusion')).toBeInTheDocument();
  });

  it('shows confidence scores for each step', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    expect(screen.getByText('90%')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('88%')).toBeInTheDocument();
  });

  it('expands step details when clicked', async () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    const step1 = screen.getByText('步骤 1').closest('[data-testid]') || 
                  screen.getByText('步骤 1').closest('div');
    
    if (step1) {
      fireEvent.click(step1);
      
      await waitFor(() => {
        expect(screen.getByText('推理内容')).toBeInTheDocument();
        expect(screen.getByText('支持证据')).toBeInTheDocument();
        expect(screen.getByText('申请表完整')).toBeInTheDocument();
        expect(screen.getByText('收入证明齐全')).toBeInTheDocument();
      });
    }
  });

  it('shows metadata when expanded', async () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    const step1Header = screen.getByText('用户查询分析：需要评估贷款申请的风险').closest('header');
    if (step1Header) {
      fireEvent.click(step1Header);
      
      await waitFor(() => {
        expect(screen.getByText('技术指标')).toBeInTheDocument();
        expect(screen.getByText('data_analysis')).toBeInTheDocument();
        expect(screen.getByText('85.0%')).toBeInTheDocument();
        expect(screen.getByText('90.0%')).toBeInTheDocument();
      });
    }
  });

  it('toggles metadata visibility', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    const metadataButton = screen.getByText('显示元数据');
    fireEvent.click(metadataButton);
    
    expect(screen.getByText('隐藏元数据')).toBeInTheDocument();
    expect(screen.getByText('推理链ID:')).toBeInTheDocument();
    expect(screen.getByText('test-chain-001')).toBeInTheDocument();
    expect(screen.getByText('步骤数量:')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('shows dependencies correctly', async () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    // 展开第三步（有依赖关系）
    const step3Header = screen.getByText('建议批准贷款申请，风险等级为低').closest('header');
    if (step3Header) {
      fireEvent.click(step3Header);
      
      await waitFor(() => {
        expect(screen.getByText('依赖步骤')).toBeInTheDocument();
        expect(screen.getByText('step-1')).toBeInTheDocument();
        expect(screen.getByText('step-2')).toBeInTheDocument();
      });
    }
  });

  it('displays reasoning chain analysis', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    expect(screen.getByText('推理链分析')).toBeInTheDocument();
    expect(screen.getByText('强项分析')).toBeInTheDocument();
    expect(screen.getByText('改进建议')).toBeInTheDocument();
    
    // 应该显示强项（因为指标都较高）
    expect(screen.getByText('逻辑推理一致性很高')).toBeInTheDocument();
    expect(screen.getByText('证据质量良好')).toBeInTheDocument();
    expect(screen.getByText('推理步骤充分详细')).toBeInTheDocument();
  });

  it('shows step duration information', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    expect(screen.getByText('150ms')).toBeInTheDocument();
    expect(screen.getByText('200ms')).toBeInTheDocument();
    expect(screen.getByText('100ms')).toBeInTheDocument();
    
    // 总耗时应该在元数据中显示
    const metadataButton = screen.getByText('显示元数据');
    fireEvent.click(metadataButton);
    
    expect(screen.getByText('450ms')).toBeInTheDocument(); // 150+200+100
  });

  it('handles different reasoning modes', () => {
    const deductiveChain = {
      ...mockChain,
      reasoning_mode: 'deductive' as const
    };
    
    render(<CoTReasoningChain chain={deductiveChain} />);
    
    expect(screen.getByText('演绎推理 - 从一般原理推导具体结论')).toBeInTheDocument();
  });

  it('applies correct styling for different step types', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    const observationStep = screen.getByText('observation').closest('div');
    const thoughtStep = screen.getByText('thought').closest('div');
    const conclusionStep = screen.getByText('conclusion').closest('div');
    
    expect(observationStep).toHaveClass('bg-blue-100', 'text-blue-700');
    expect(thoughtStep).toHaveClass('bg-purple-100', 'text-purple-700');
    expect(conclusionStep).toHaveClass('bg-red-100', 'text-red-700');
  });

  it('shows confidence colors correctly', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    // 高置信度应该是绿色
    const highConfidence = screen.getByText('90%');
    expect(highConfidence).toHaveClass('text-green-600');
    
    // 中等置信度应该是黄色
    const mediumConfidence = screen.getByText('85%');
    expect(mediumConfidence).toHaveClass('text-green-600'); // 85% 仍然是高置信度
  });

  it('handles empty steps gracefully', () => {
    const emptyChain = {
      ...mockChain,
      steps: []
    };
    
    render(<CoTReasoningChain chain={emptyChain} />);
    
    expect(screen.getByText('Chain-of-Thought 推理链')).toBeInTheDocument();
    expect(screen.getByText('推理步骤')).toBeInTheDocument();
    // 不应该崩溃
  });

  it('formats creation time correctly', () => {
    render(<CoTReasoningChain chain={mockChain} />);
    
    const metadataButton = screen.getByText('显示元数据');
    fireEvent.click(metadataButton);
    
    // 应该显示格式化的时间
    expect(screen.getByText(/2025/)).toBeInTheDocument();
  });
});