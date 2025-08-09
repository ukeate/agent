/**
 * ExplanationViewer组件单元测试
 * 
 * 测试功能包括：
 * - 基础渲染和空状态显示
 * - 解释数据展示和格式化
 * - 标签页切换和视图模式
 * - 用户交互和事件处理
 * - 分享和导出功能
 * - 推理步骤选择和详情显示
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import ExplanationViewer from '../../../src/components/agentic-rag/ExplanationViewer';

// Mock dependencies
vi.mock('../../../src/stores/ragStore', () => ({
  useRagStore: vi.fn(() => ({
    agenticResults: null,
    explanationData: null,
    currentQuery: '',
    showExplanation: true,
    setShowExplanation: vi.fn(),
  })),
}));

// Mock message
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
    },
  };
});

describe('ExplanationViewer', () => {
  const mockAgenticResults = {
    success: true,
    query_id: 'test-query-1',
    analysis_info: {
      intent_type: 'factual',
      complexity_score: 0.6,
      entities: ['AI', '代理', '系统'],
      keywords: ['智能', '检索', '系统'],
    },
    expanded_queries: [
      {
        original_query: '如何实现多智能体协作',
        expanded_queries: ['多智能体协作实现', '代理协作机制'],
        strategy: 'semantic',
        confidence: 0.85,
      },
    ],
    results: [
      {
        id: '1',
        content: '多智能体系统实现方法',
        file_path: '/docs/multi-agent.md',
        score: 0.9,
        content_type: 'markdown',
        metadata: { source: 'documentation' },
      },
      {
        id: '2',
        content: 'class AgentCoordinator',
        file_path: '/src/coordinator.py',
        score: 0.8,
        content_type: 'code',
        metadata: { source: 'codebase' },
      },
    ],
    processing_time: 1500,
    confidence: 0.85,
    timestamp: '2024-01-01T00:00:00Z',
  };

  const mockOnShare = vi.fn();
  const mockOnExport = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    // Mock DOM methods for export functionality
    global.document.createElement = vi.fn((tagName: string) => {
      if (tagName === 'a') {
        return {
          setAttribute: vi.fn(),
          click: vi.fn(),
        } as any;
      }
      return {} as any;
    });
  });

  describe('基础渲染', () => {
    it('无数据时显示空状态', () => {
      render(<ExplanationViewer />);
      
      expect(screen.getByText('暂无解释数据')).toBeInTheDocument();
    });

    it('有数据时显示解释界面', () => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
      
      expect(screen.getByText('检索过程解释')).toBeInTheDocument();
      expect(screen.getByText('智能分析')).toBeInTheDocument();
    });

    it('compact模式下隐藏额外控件', () => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          compact={true}
        />
      );
      
      // 紧凑模式下不显示分享和导出按钮
      expect(screen.queryByRole('button', { name: /分享/ })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /导出/ })).not.toBeInTheDocument();
    });
  });

  describe('解释数据显示', () => {
    beforeEach(() => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
    });

    it('显示智能检索摘要', () => {
      expect(screen.getByText('智能检索摘要')).toBeInTheDocument();
      expect(screen.getByText(/基于对查询.*的分析/)).toBeInTheDocument();
    });

    it('显示置信度指标', () => {
      expect(screen.getByText('整体置信度')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
    });

    it('显示质量评估指标', () => {
      expect(screen.getByText('质量评估')).toBeInTheDocument();
      expect(screen.getByText('relevance')).toBeInTheDocument();
      expect(screen.getByText('accuracy')).toBeInTheDocument();
    });

    it('显示关键统计数据', () => {
      expect(screen.getByText('检索结果数')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument(); // 结果数量
      expect(screen.getByText('扩展查询数')).toBeInTheDocument();
      expect(screen.getByText('1')).toBeInTheDocument(); // 扩展查询数量
      expect(screen.getByText('处理时间')).toBeInTheDocument();
      expect(screen.getByText('1500')).toBeInTheDocument(); // 处理时间
    });
  });

  describe('标签页切换', () => {
    beforeEach(() => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
    });

    it('默认显示总览标签页', () => {
      const overviewTab = screen.getByRole('tab', { name: /总览/ });
      expect(overviewTab).toHaveAttribute('aria-selected', 'true');
    });

    it('可以切换到推理过程标签页', async () => {
      const user = userEvent.setup();
      const reasoningTab = screen.getByRole('tab', { name: /推理过程/ });
      
      await user.click(reasoningTab);
      
      expect(reasoningTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('查询理解')).toBeInTheDocument();
      expect(screen.getByText('策略选择')).toBeInTheDocument();
      expect(screen.getByText('多代理检索')).toBeInTheDocument();
      expect(screen.getByText('结果验证')).toBeInTheDocument();
    });

    it('可以切换到详细分析标签页', async () => {
      const user = userEvent.setup();
      const analysisTab = screen.getByRole('tab', { name: /详细分析/ });
      
      await user.click(analysisTab);
      
      expect(analysisTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('查询理解分析')).toBeInTheDocument();
      expect(screen.getByText('策略选择分析')).toBeInTheDocument();
      expect(screen.getByText('检索过程分析')).toBeInTheDocument();
    });
  });

  describe('推理过程交互', () => {
    beforeEach(async () => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
      
      const user = userEvent.setup();
      const reasoningTab = screen.getByRole('tab', { name: /推理过程/ });
      await user.click(reasoningTab);
    });

    it('可以选择推理步骤查看详情', async () => {
      const user = userEvent.setup();
      
      // 点击查询理解步骤
      const understandingStep = screen.getByText('查询理解');
      await user.click(understandingStep);
      
      // 验证步骤详情显示
      expect(screen.getByText('步骤详情')).toBeInTheDocument();
      expect(screen.getByText('备选方案:')).toBeInTheDocument();
    });

    it('支持不同的视图模式切换', async () => {
      const user = userEvent.setup();
      
      // 切换到时间线视图
      const timelineButton = screen.getByRole('button', { name: '时间线' });
      await user.click(timelineButton);
      
      expect(timelineButton).toHaveClass('ant-btn-primary');
      
      // 切换到树形图视图
      const treeButton = screen.getByRole('button', { name: '树形图' });
      await user.click(treeButton);
      
      expect(treeButton).toHaveClass('ant-btn-primary');
    });
  });

  describe('详细分析展开', () => {
    beforeEach(async () => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
      
      const user = userEvent.setup();
      const analysisTab = screen.getByRole('tab', { name: /详细分析/ });
      await user.click(analysisTab);
    });

    it('显示查询理解分析详情', () => {
      expect(screen.getByText('意图: factual')).toBeInTheDocument();
      expect(screen.getByText('复杂度: 60%')).toBeInTheDocument();
      expect(screen.getByText('关键概念:')).toBeInTheDocument();
      expect(screen.getByText('AI')).toBeInTheDocument();
      expect(screen.getByText('代理')).toBeInTheDocument();
      expect(screen.getByText('系统')).toBeInTheDocument();
    });

    it('显示策略选择分析详情', () => {
      expect(screen.getByText('选择的策略:')).toBeInTheDocument();
      expect(screen.getByText('选择理由')).toBeInTheDocument();
      expect(screen.getByText('权衡考量:')).toBeInTheDocument();
    });

    it('显示检索过程分析详情', () => {
      expect(screen.getByText('代理协作:')).toBeInTheDocument();
      expect(screen.getByText('semantic_retriever')).toBeInTheDocument();
      expect(screen.getByText('keyword_retriever')).toBeInTheDocument();
      expect(screen.getByText('structured_retriever')).toBeInTheDocument();
    });
  });

  describe('分享和导出功能', () => {
    beforeEach(() => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
          onShare={mockOnShare}
          onExport={mockOnExport}
        />
      );
    });

    it('点击分享按钮触发分享功能', async () => {
      const user = userEvent.setup();
      const shareButton = screen.getByRole('button', { name: /分享解释/ });
      
      await user.click(shareButton);
      
      expect(mockOnShare).toHaveBeenCalledWith(
        expect.objectContaining({
          query: '如何实现多智能体协作',
          explanation: expect.any(Object),
          timestamp: expect.any(String),
        })
      );
    });

    it('点击导出按钮触发导出功能', async () => {
      const user = userEvent.setup();
      const exportButton = screen.getByRole('button', { name: /导出解释/ });
      
      await user.click(exportButton);
      
      expect(mockOnExport).toHaveBeenCalledWith(
        expect.objectContaining({
          query: '如何实现多智能体协作',
          explanation: expect.any(Object),
          reasoning_steps: expect.any(Array),
          timestamp: expect.any(String),
        })
      );
    });
  });

  describe('显示/隐藏控制', () => {
    it('可以控制解释内容的显示和隐藏', async () => {
      const mockSetShowExplanation = vi.fn();
      
      // Mock store to return setShowExplanation function
      const { useRagStore } = await import('../../../src/stores/ragStore');
      vi.mocked(useRagStore).mockReturnValue({
        agenticResults: null,
        explanationData: null,
        currentQuery: '',
        showExplanation: true,
        setShowExplanation: mockSetShowExplanation,
      });

      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
      
      const user = userEvent.setup();
      const hideButton = screen.getByRole('button', { name: '隐藏' });
      
      await user.click(hideButton);
      
      expect(mockSetShowExplanation).toHaveBeenCalledWith(false);
    });
  });

  describe('置信度和质量指标渲染', () => {
    beforeEach(() => {
      render(
        <ExplanationViewer
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
    });

    it('根据置信度显示不同颜色的徽章', () => {
      // 高置信度显示绿色
      const badges = screen.getAllByText(/\d+%/);
      expect(badges.length).toBeGreaterThan(0);
    });

    it('显示各维度的质量评分', () => {
      const qualityScores = ['relevance', 'accuracy', 'completeness', 'timeliness', 'credibility', 'clarity'];
      
      qualityScores.forEach(score => {
        expect(screen.getByText(score)).toBeInTheDocument();
      });
    });
  });

  describe('错误处理', () => {
    it('无效数据时正常渲染空状态', () => {
      render(
        <ExplanationViewer
          agenticResults={null}
          explanationData={undefined}
        />
      );
      
      expect(screen.getByText('暂无解释数据')).toBeInTheDocument();
    });

    it('部分数据缺失时使用默认值', () => {
      const partialResults = {
        ...mockAgenticResults,
        analysis_info: undefined,
        results: undefined,
      };
      
      render(
        <ExplanationViewer
          agenticResults={partialResults}
          query="测试查询"
        />
      );
      
      expect(screen.getByText('检索过程解释')).toBeInTheDocument();
      expect(screen.getByText('检索结果数')).toBeInTheDocument();
      expect(screen.getByText('0')).toBeInTheDocument(); // 默认结果数
    });
  });

  describe('自定义类名和样式', () => {
    it('应用自定义className', () => {
      const { container } = render(
        <ExplanationViewer
          className="custom-explanation-viewer"
          agenticResults={mockAgenticResults}
        />
      );
      
      expect(container.querySelector('.custom-explanation-viewer')).toBeInTheDocument();
    });
  });
});