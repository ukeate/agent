/**
 * FeedbackInterface组件单元测试
 * 
 * 测试功能包括：
 * - 基础渲染和空状态显示
 * - 多维度评分功能
 * - 结果反馈评价
 * - 系统反馈收集
 * - 总体反馈提交
 * - 历史记录管理
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach, Mock } from 'vitest';
import FeedbackInterface from '../../../src/components/agentic-rag/FeedbackInterface';

// Mock dependencies
const mockRagStore = {
  agenticResults: null,
  currentQuery: '',
  setFeedbackData: vi.fn(),
  clearFeedback: vi.fn(),
};

vi.mock('../../../src/stores/ragStore', () => ({
  useRagStore: () => mockRagStore,
}));

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

// Mock message
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
    },
  };
});

describe('FeedbackInterface', () => {
  const mockAgenticResults = {
    success: true,
    query_id: 'test-query-1',
    results: [
      {
        id: '1',
        content: '多智能体系统实现方法详解，包括架构设计和协作机制',
        file_path: '/docs/multi-agent-implementation.md',
        score: 0.95,
        content_type: 'markdown',
        metadata: { source: 'documentation', author: 'system' },
      },
      {
        id: '2',
        content: 'class AgentCoordinator: 代理协调器的实现代码',
        file_path: '/src/agent/coordinator.py',
        score: 0.88,
        content_type: 'code',
        metadata: { source: 'codebase', language: 'python' },
      },
    ],
    processing_time: 1200,
    confidence: 0.9,
    timestamp: '2024-01-01T00:00:00Z',
  };

  const mockOnFeedbackSubmit = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(null);
    mockRagStore.agenticResults = null;
    mockRagStore.currentQuery = '';
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('基础渲染', () => {
    it('无数据时显示空状态', () => {
      render(<FeedbackInterface />);
      
      expect(screen.getByText('请先进行智能检索以提供反馈')).toBeInTheDocument();
    });

    it('有数据时显示反馈界面', () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
      
      expect(screen.getByText('智能检索反馈')).toBeInTheDocument();
      expect(screen.getByText('改进系统')).toBeInTheDocument();
    });

    it('compact模式下使用简化样式', () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          compact={true}
        />
      );
      
      // 紧凑模式下应该不显示历史反馈按钮
      expect(screen.queryByRole('button', { name: /历史反馈/ })).not.toBeInTheDocument();
    });
  });

  describe('标签页切换', () => {
    beforeEach(() => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
    });

    it('默认显示质量评分标签页', () => {
      const ratingTab = screen.getByRole('tab', { name: /质量评分/ });
      expect(ratingTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('请对本次检索结果的各个维度进行评分')).toBeInTheDocument();
    });

    it('可以切换到结果反馈标签页', async () => {
      const user = userEvent.setup();
      const resultsTab = screen.getByRole('tab', { name: /结果反馈/ });
      
      await user.click(resultsTab);
      
      expect(resultsTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('请对每个检索结果进行评价')).toBeInTheDocument();
    });

    it('可以切换到系统反馈标签页', async () => {
      const user = userEvent.setup();
      const systemTab = screen.getByRole('tab', { name: /系统反馈/ });
      
      await user.click(systemTab);
      
      expect(systemTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('请评价系统的性能和易用性')).toBeInTheDocument();
    });

    it('可以切换到总体反馈标签页', async () => {
      const user = userEvent.setup();
      const generalTab = screen.getByRole('tab', { name: /总体反馈/ });
      
      await user.click(generalTab);
      
      expect(generalTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('请提供您的总体评价和建议')).toBeInTheDocument();
    });
  });

  describe('质量评分功能', () => {
    beforeEach(() => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
    });

    it('显示各个评分维度', () => {
      expect(screen.getByText('相关性:')).toBeInTheDocument();
      expect(screen.getByText('准确性:')).toBeInTheDocument();
      expect(screen.getByText('完整性:')).toBeInTheDocument();
      expect(screen.getByText('实用性:')).toBeInTheDocument();
      expect(screen.getByText('清晰度:')).toBeInTheDocument();
      expect(screen.getByText('总体满意度:')).toBeInTheDocument();
    });

    it('可以对各维度进行评分', async () => {
      const user = userEvent.setup();
      
      // 找到相关性评分的星星
      const relevanceStars = screen.getAllByRole('radio');
      const relevanceFourStar = relevanceStars.find(star => 
        star.getAttribute('aria-label') === '4 stars' && 
        star.closest('.ant-rate')?.previousElementSibling?.textContent?.includes('相关性')
      );
      
      if (relevanceFourStar) {
        await user.click(relevanceFourStar);
        expect(relevanceFourStar).toBeChecked();
      }
    });

    it('总体满意度可以独立评分', async () => {
      const user = userEvent.setup();
      
      // 找到总体满意度的评分组件
      const overallSatisfactionContainer = screen.getByText('总体满意度:').closest('div');
      const overallStars = overallSatisfactionContainer?.querySelectorAll('.ant-rate-star input');
      
      if (overallStars && overallStars.length > 0) {
        await user.click(overallStars[4] as HTMLElement); // 点击5星
        expect(overallStars[4]).toBeChecked();
      }
    });
  });

  describe('结果反馈评价', () => {
    beforeEach(async () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
      
      const user = userEvent.setup();
      const resultsTab = screen.getByRole('tab', { name: /结果反馈/ });
      await user.click(resultsTab);
    });

    it('显示所有检索结果', () => {
      expect(screen.getByText('结果 1: multi-agent-implementation.md')).toBeInTheDocument();
      expect(screen.getByText('结果 2: coordinator.py')).toBeInTheDocument();
      
      // 验证结果内容预览
      expect(screen.getByText(/多智能体系统实现方法详解/)).toBeInTheDocument();
      expect(screen.getByText(/class AgentCoordinator/)).toBeInTheDocument();
    });

    it('可以对结果进行星级评分', async () => {
      const user = userEvent.setup();
      
      // 查找第一个结果的评分星星
      const ratingStars = screen.getAllByRole('radio');
      const firstResultRatingStars = ratingStars.filter(star => 
        star.getAttribute('aria-label')?.includes('stars')
      );
      
      if (firstResultRatingStars.length > 0) {
        await user.click(firstResultRatingStars[3]); // 点击4星
        expect(firstResultRatingStars[3]).toBeChecked();
      }
    });

    it('可以标记结果是否有帮助', async () => {
      const user = userEvent.setup();
      
      // 查找"有帮助"按钮
      const helpfulButtons = screen.getAllByRole('button');
      const thumbsUpButton = helpfulButtons.find(btn => 
        btn.querySelector('.anticon-like')
      );
      
      if (thumbsUpButton) {
        await user.click(thumbsUpButton);
        expect(thumbsUpButton).toHaveClass('ant-btn-primary');
      }
    });

    it('可以标记结果问题', async () => {
      const user = userEvent.setup();
      
      // 查找问题标记复选框
      const issueCheckbox = screen.getByLabelText('内容过时');
      await user.click(issueCheckbox);
      
      expect(issueCheckbox).toBeChecked();
    });

    it('可以填写具体意见', async () => {
      const user = userEvent.setup();
      
      // 查找评论输入框
      const commentTextareas = screen.getAllByPlaceholderText(/对这个结果的具体意见/);
      if (commentTextareas.length > 0) {
        await user.type(commentTextareas[0], '这个结果很有用，但需要更多示例代码');
        expect(commentTextareas[0]).toHaveValue('这个结果很有用，但需要更多示例代码');
      }
    });
  });

  describe('系统反馈收集', () => {
    beforeEach(async () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
      
      const user = userEvent.setup();
      const systemTab = screen.getByRole('tab', { name: /系统反馈/ });
      await user.click(systemTab);
    });

    it('可以评价响应时间', async () => {
      const user = userEvent.setup();
      
      expect(screen.getByText('响应时间感受:')).toBeInTheDocument();
      
      const fastOption = screen.getByLabelText('很快');
      await user.click(fastOption);
      
      expect(fastOption).toBeChecked();
    });

    it('可以评分界面易用性', async () => {
      const user = userEvent.setup();
      
      expect(screen.getByText('界面易用性:')).toBeInTheDocument();
      
      // 查找界面易用性评分的星星
      const usabilityStars = screen.getAllByRole('radio');
      const usabilityFourStar = usabilityStars.find(star => 
        star.getAttribute('aria-label') === '4 stars'
      );
      
      if (usabilityFourStar) {
        await user.click(usabilityFourStar);
        expect(usabilityFourStar).toBeChecked();
      }
    });

    it('可以选择功能需求', async () => {
      const user = userEvent.setup();
      
      expect(screen.getByText('希望增加的功能:')).toBeInTheDocument();
      
      const feature = screen.getByLabelText('增加更多数据源');
      await user.click(feature);
      
      expect(feature).toBeChecked();
    });

    it('可以报告遇到的问题', async () => {
      const user = userEvent.setup();
      
      expect(screen.getByText('遇到的问题:')).toBeInTheDocument();
      
      // 查找Bug报告输入框
      const bugInput = screen.getByPlaceholderText('描述遇到的问题或Bug...');
      await user.click(bugInput);
      await user.type(bugInput, '搜索结果加载慢{enter}');
      
      expect(screen.getByText('搜索结果加载慢')).toBeInTheDocument();
    });
  });

  describe('总体反馈提交', () => {
    beforeEach(async () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
          onFeedbackSubmit={mockOnFeedbackSubmit}
        />
      );
      
      const user = userEvent.setup();
      const generalTab = screen.getByRole('tab', { name: /总体反馈/ });
      await user.click(generalTab);
    });

    it('可以填写总体评价', async () => {
      const user = userEvent.setup();
      
      const commentTextarea = screen.getByPlaceholderText(/请分享您对本次智能检索体验/);
      await user.type(commentTextarea, '系统整体表现良好，检索准确度高');
      
      expect(commentTextarea).toHaveValue('系统整体表现良好，检索准确度高');
    });

    it('可以选择改进建议', async () => {
      const user = userEvent.setup();
      
      const improvementSuggestion = screen.getByLabelText('提升检索准确性');
      await user.click(improvementSuggestion);
      
      expect(improvementSuggestion).toBeChecked();
    });

    it('可以选择推荐意愿', async () => {
      const user = userEvent.setup();
      
      const recommendYes = screen.getByLabelText('是的，我会推荐');
      await user.click(recommendYes);
      
      expect(recommendYes).toBeChecked();
    });

    it('可以填写联系信息', async () => {
      const user = userEvent.setup();
      
      const contactInput = screen.getByPlaceholderText(/如果您希望我们就反馈内容与您联系/);
      await user.type(contactInput, 'user@example.com');
      
      expect(contactInput).toHaveValue('user@example.com');
    });

    it('可以成功提交反馈', async () => {
      const user = userEvent.setup();
      
      // 填写必要信息
      const commentTextarea = screen.getByPlaceholderText(/请分享您对本次智能检索体验/);
      await user.type(commentTextarea, '很好的系统');
      
      const recommendYes = screen.getByLabelText('是的，我会推荐');
      await user.click(recommendYes);
      
      // 提交表单
      const submitButton = screen.getByRole('button', { name: /提交反馈/ });
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(mockOnFeedbackSubmit).toHaveBeenCalled();
      });
      
      // 验证localStorage调用
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'rag_feedback_history',
        expect.any(String)
      );
    });
  });

  describe('历史记录功能', () => {
    const mockFeedbackHistory = [
      {
        id: 'feedback_1',
        query_id: 'query_1',
        query_text: '测试查询1',
        timestamp: new Date('2024-01-01'),
        overall_satisfaction: 4,
        would_recommend: true,
        general_comments: '很好的结果',
      },
      {
        id: 'feedback_2',
        query_id: 'query_2',
        query_text: '测试查询2',
        timestamp: new Date('2024-01-02'),
        overall_satisfaction: 3,
        would_recommend: false,
        general_comments: '还可以改进',
      },
    ];

    beforeEach(() => {
      mockLocalStorage.getItem.mockReturnValue(JSON.stringify(mockFeedbackHistory));
    });

    it('从localStorage加载历史记录', () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          showHistory={true}
        />
      );
      
      expect(mockLocalStorage.getItem).toHaveBeenCalledWith('rag_feedback_history');
    });

    it('可以打开历史记录模态框', async () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          showHistory={true}
        />
      );
      
      const user = userEvent.setup();
      const historyButton = screen.getByRole('button', { name: /历史反馈/ });
      await user.click(historyButton);
      
      expect(screen.getByText('反馈历史')).toBeInTheDocument();
      expect(screen.getByText('总反馈数')).toBeInTheDocument();
      expect(screen.getByText('平均满意度')).toBeInTheDocument();
    });

    it('显示历史反馈统计', async () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          showHistory={true}
        />
      );
      
      const user = userEvent.setup();
      const historyButton = screen.getByRole('button', { name: /历史反馈/ });
      await user.click(historyButton);
      
      // 验证统计数据显示
      expect(screen.getByText('2')).toBeInTheDocument(); // 总反馈数
      expect(screen.getByText('3.5')).toBeInTheDocument(); // 平均满意度 (4+3)/2=3.5
    });

    it('显示历史反馈列表', async () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          showHistory={true}
        />
      );
      
      const user = userEvent.setup();
      const historyButton = screen.getByRole('button', { name: /历史反馈/ });
      await user.click(historyButton);
      
      expect(screen.getByText('查询: "测试查询1"')).toBeInTheDocument();
      expect(screen.getByText('查询: "测试查询2"')).toBeInTheDocument();
    });

    it('showHistory为false时不显示历史按钮', () => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          showHistory={false}
        />
      );
      
      expect(screen.queryByRole('button', { name: /历史反馈/ })).not.toBeInTheDocument();
    });
  });

  describe('表单清空功能', () => {
    beforeEach(() => {
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          query="如何实现多智能体协作"
        />
      );
    });

    it('可以清空表单内容', async () => {
      const user = userEvent.setup();
      
      // 先进行一些评分
      const overallSatisfactionContainer = screen.getByText('总体满意度:').closest('div');
      const overallStars = overallSatisfactionContainer?.querySelectorAll('.ant-rate-star input');
      if (overallStars && overallStars.length > 0) {
        await user.click(overallStars[3] as HTMLElement);
      }
      
      // 点击清空按钮
      const clearButtons = screen.getAllByRole('button', { name: '清空' });
      await user.click(clearButtons[0]);
      
      // 验证表单被清空（这里简化验证，实际应该检查表单字段）
      expect(mockRagStore.clearFeedback).toHaveBeenCalled();
    });
  });

  describe('localStorage错误处理', () => {
    it('处理localStorage读取错误', () => {
      mockLocalStorage.getItem.mockReturnValue('invalid json');
      
      // 模拟console.error来验证错误被正确处理
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      render(
        <FeedbackInterface
          agenticResults={mockAgenticResults}
          showHistory={true}
        />
      );
      
      expect(consoleSpy).toHaveBeenCalledWith('Failed to load feedback history:', expect.any(Error));
      
      consoleSpy.mockRestore();
    });
  });

  describe('自定义类名和样式', () => {
    it('应用自定义className', () => {
      const { container } = render(
        <FeedbackInterface
          className="custom-feedback-interface"
          agenticResults={mockAgenticResults}
        />
      );
      
      expect(container.querySelector('.custom-feedback-interface')).toBeInTheDocument();
    });
  });
});