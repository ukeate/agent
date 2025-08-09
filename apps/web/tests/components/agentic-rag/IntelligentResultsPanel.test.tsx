/**
 * IntelligentResultsPanel组件单元测试
 * 
 * 测试功能包括：
 * - 基础渲染和空状态显示
 * - 智能聚类展示
 * - 结果质量分析
 * - 排序和筛选功能
 * - 结果项交互
 * - 视图模式切换
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, Mock } from 'vitest';
import IntelligentResultsPanel from '../../../src/components/agentic-rag/IntelligentResultsPanel';
import { KnowledgeItem } from '../../../src/services/ragService';

// Mock dependencies
const mockRagStore = {
  queryResults: [],
  agenticResults: null,
  currentQuery: '',
};

vi.mock('../../../src/stores/ragStore', () => ({
  useRagStore: () => mockRagStore,
}));

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

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockResolvedValue(undefined),
  },
});

describe('IntelligentResultsPanel', () => {
  const mockResults: KnowledgeItem[] = [
    {
      id: '1',
      content: '多智能体系统实现方法详解，包括架构设计、协作机制和通信协议。这是一个完整的技术指南。',
      file_path: '/docs/multi-agent/architecture.md',
      score: 0.95,
      content_type: 'documentation',
      metadata: {
        source: 'documentation',
        updated_at: '2024-01-15T10:00:00Z',
        language: 'zh',
      },
    },
    {
      id: '2',
      content: 'class AgentCoordinator: def __init__(self): self.agents = [] # 代理协调器的实现',
      file_path: '/src/agent/coordinator.py',
      score: 0.88,
      content_type: 'code',
      metadata: {
        source: 'codebase',
        updated_at: '2024-01-10T15:30:00Z',
        language: 'python',
      },
    },
    {
      id: '3',
      content: '智能代理之间的通信协议定义，支持异步消息传递和状态同步。',
      file_path: '/docs/communication-protocol.md',
      score: 0.75,
      content_type: 'documentation',
      metadata: {
        source: 'documentation',
        updated_at: '2024-01-05T09:20:00Z',
      },
    },
    {
      id: '4',
      content: 'function createAgent() { return new Agent(); } // 代理创建函数',
      file_path: '/src/utils/agent-factory.js',
      score: 0.45,
      content_type: 'code',
      metadata: {
        source: 'codebase',
        language: 'javascript',
      },
    },
  ];

  const mockOnItemSelect = vi.fn();
  const mockOnItemRate = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    mockRagStore.queryResults = [];
    mockRagStore.agenticResults = null;
    mockRagStore.currentQuery = '';
  });

  describe('基础渲染', () => {
    it('无结果时显示空状态', () => {
      render(<IntelligentResultsPanel />);
      
      expect(screen.getByText('暂无检索结果')).toBeInTheDocument();
    });

    it('有结果时显示智能结果面板', () => {
      render(
        <IntelligentResultsPanel
          results={mockResults}
          query="如何实现多智能体协作"
        />
      );
      
      expect(screen.getByText('智能结果分析')).toBeInTheDocument();
      expect(screen.getByText('4')).toBeInTheDocument(); // 结果数量显示
    });

    it('显示控制栏组件', () => {
      render(<IntelligentResultsPanel results={mockResults} />);
      
      expect(screen.getByPlaceholderText('筛选结果...')).toBeInTheDocument();
      expect(screen.getByDisplayValue('相关性')).toBeInTheDocument(); // 排序选择器
      expect(screen.getByDisplayValue('全部')).toBeInTheDocument(); // 筛选选择器
    });
  });

  describe('视图模式切换', () => {
    beforeEach(() => {
      render(<IntelligentResultsPanel results={mockResults} />);
    });

    it('默认显示智能聚类标签页', () => {
      const clusterTab = screen.getByRole('tab', { name: /智能聚类/ });
      expect(clusterTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('结果聚类')).toBeInTheDocument();
    });

    it('可以切换到列表视图', async () => {
      const user = userEvent.setup();
      const listTab = screen.getByRole('tab', { name: /列表视图/ });
      
      await user.click(listTab);
      
      expect(listTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('architecture.md')).toBeInTheDocument();
      expect(screen.getByText('coordinator.py')).toBeInTheDocument();
    });

    it('可以切换到质量分析', async () => {
      const user = userEvent.setup();
      const analysisTab = screen.getByRole('tab', { name: /质量分析/ });
      
      await user.click(analysisTab);
      
      expect(analysisTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('质量分布')).toBeInTheDocument();
      expect(screen.getByText('覆盖分析')).toBeInTheDocument();
      expect(screen.getByText('内容类型分布')).toBeInTheDocument();
    });
  });

  describe('智能聚类功能', () => {
    beforeEach(() => {
      render(<IntelligentResultsPanel results={mockResults} />);
    });

    it('显示聚类列表', () => {
      expect(screen.getByText('结果聚类')).toBeInTheDocument();
      expect(screen.getByText('文档资料')).toBeInTheDocument();
      expect(screen.getByText('代码相关')).toBeInTheDocument();
    });

    it('显示每个聚类的统计信息', () => {
      // 文档类聚类应该有2个结果
      const docClusterBadges = screen.getAllByText('2');
      expect(docClusterBadges.length).toBeGreaterThan(0);
    });

    it('可以选择聚类查看详情', async () => {
      const user = userEvent.setup();
      
      // 点击文档资料聚类
      const docCluster = screen.getByText('文档资料');
      await user.click(docCluster);
      
      expect(screen.getByText('聚类结果 - 文档资料')).toBeInTheDocument();
      expect(screen.getByText('architecture.md')).toBeInTheDocument();
    });

    it('未选择聚类时显示提示信息', () => {
      expect(screen.getByText('请选择聚类查看详情')).toBeInTheDocument();
      expect(screen.getByText('请选择左侧的聚类查看结果')).toBeInTheDocument();
    });
  });

  describe('质量分析功能', () => {
    beforeEach(async () => {
      render(<IntelligentResultsPanel results={mockResults} />);
      
      const user = userEvent.setup();
      const analysisTab = screen.getByRole('tab', { name: /质量分析/ });
      await user.click(analysisTab);
    });

    it('显示质量分布统计', () => {
      expect(screen.getByText('高质量结果')).toBeInTheDocument();
      expect(screen.getByText('中等质量结果')).toBeInTheDocument();
      expect(screen.getByText('待改进结果')).toBeInTheDocument();
      
      // 验证高质量结果数量 (score >= 0.8 的有2个)
      expect(screen.getByText('2')).toBeInTheDocument();
    });

    it('显示覆盖分析指标', () => {
      expect(screen.getByText('来源多样性')).toBeInTheDocument();
      expect(screen.getByText('覆盖度评分')).toBeInTheDocument();
    });

    it('显示内容类型分布', () => {
      expect(screen.getByText('内容类型分布')).toBeInTheDocument();
      expect(screen.getByText(/documentation:/)).toBeInTheDocument();
      expect(screen.getByText(/code:/)).toBeInTheDocument();
    });
  });

  describe('排序和筛选功能', () => {
    beforeEach(async () => {
      render(<IntelligentResultsPanel results={mockResults} />);
      
      const user = userEvent.setup();
      const listTab = screen.getByRole('tab', { name: /列表视图/ });
      await user.click(listTab);
    });

    it('可以按质量筛选结果', async () => {
      const user = userEvent.setup();
      
      // 选择只显示高质量结果
      const filterSelect = screen.getByDisplayValue('全部');
      await user.click(filterSelect);
      
      const highQualityOption = screen.getByText('高质量');
      await user.click(highQualityOption);
      
      // 应该只显示高质量结果（score >= 0.8）
      expect(screen.getByText('architecture.md')).toBeInTheDocument();
      expect(screen.getByText('coordinator.py')).toBeInTheDocument();
      expect(screen.queryByText('agent-factory.js')).not.toBeInTheDocument();
    });

    it('可以按来源排序结果', async () => {
      const user = userEvent.setup();
      
      const sortSelect = screen.getByDisplayValue('相关性');
      await user.click(sortSelect);
      
      const sourceOption = screen.getByText('来源');
      await user.click(sourceOption);
      
      // 验证排序后的顺序（按文件路径字母顺序）
      const resultItems = screen.getAllByText(/\.(md|py|js)$/);
      expect(resultItems.length).toBeGreaterThan(0);
    });

    it('可以搜索筛选结果', async () => {
      const user = userEvent.setup();
      
      const searchInput = screen.getByPlaceholderText('筛选结果...');
      await user.type(searchInput, '协调器');
      
      // 应该只显示包含"协调器"的结果
      expect(screen.getByText('coordinator.py')).toBeInTheDocument();
      expect(screen.queryByText('architecture.md')).not.toBeInTheDocument();
    });
  });

  describe('结果项交互', () => {
    beforeEach(async () => {
      render(
        <IntelligentResultsPanel
          results={mockResults}
          onItemSelect={mockOnItemSelect}
          onItemRate={mockOnItemRate}
        />
      );
      
      const user = userEvent.setup();
      const listTab = screen.getByRole('tab', { name: /列表视图/ });
      await user.click(listTab);
    });

    it('点击结果项触发选择回调', async () => {
      const user = userEvent.setup();
      
      const firstResultItem = screen.getByText('architecture.md').closest('.intelligent-result-item');
      if (firstResultItem) {
        await user.click(firstResultItem);
        expect(mockOnItemSelect).toHaveBeenCalledWith(mockResults[0]);
      }
    });

    it('可以对结果进行评分', async () => {
      const user = userEvent.setup();
      
      // 查找第一个结果的评分星星
      const ratingStars = screen.getAllByRole('radio');
      const fourStarRating = ratingStars.find(star => 
        star.getAttribute('aria-label') === '4 stars'
      );
      
      if (fourStarRating) {
        await user.click(fourStarRating);
        expect(mockOnItemRate).toHaveBeenCalledWith(mockResults[0], 4);
      }
    });

    it('可以复制结果路径', async () => {
      const user = userEvent.setup();
      
      // 查找复制链接按钮
      const linkButtons = screen.getAllByRole('button');
      const linkButton = linkButtons.find(btn => 
        btn.querySelector('.anticon-link')
      );
      
      if (linkButton) {
        await user.click(linkButton);
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith('/docs/multi-agent/architecture.md');
      }
    });

    it('显示结果质量详情', async () => {
      const user = userEvent.setup();
      
      // 查找质量标签
      const qualityBadge = screen.getByText('高质量');
      await user.hover(qualityBadge);
      
      await waitFor(() => {
        expect(screen.getByText('质量评估详情')).toBeInTheDocument();
        expect(screen.getByText('相关性:')).toBeInTheDocument();
        expect(screen.getByText('准确性:')).toBeInTheDocument();
      });
    });
  });

  describe('结果展示格式', () => {
    beforeEach(async () => {
      render(<IntelligentResultsPanel results={mockResults} />);
      
      const user = userEvent.setup();
      const listTab = screen.getByRole('tab', { name: /列表视图/ });
      await user.click(listTab);
    });

    it('正确显示文件类型图标', () => {
      // 文档类型应该显示书本图标
      const docIcons = screen.getAllByRole('img', { name: /book/ });
      expect(docIcons.length).toBeGreaterThan(0);
      
      // 代码类型应该显示代码图标
      const codeIcons = screen.getAllByRole('img', { name: /code/ });
      expect(codeIcons.length).toBeGreaterThan(0);
    });

    it('显示结果元数据标签', () => {
      expect(screen.getByText('documentation')).toBeInTheDocument();
      expect(screen.getByText('code')).toBeInTheDocument();
      expect(screen.getByText('python')).toBeInTheDocument();
      expect(screen.getByText('javascript')).toBeInTheDocument();
    });

    it('显示相关度百分比', () => {
      expect(screen.getByText('相关度: 95%')).toBeInTheDocument();
      expect(screen.getByText('相关度: 88%')).toBeInTheDocument();
      expect(screen.getByText('相关度: 75%')).toBeInTheDocument();
      expect(screen.getByText('相关度: 45%')).toBeInTheDocument();
    });

    it('显示更新时间信息', () => {
      // 检查日期格式显示
      expect(screen.getByText('2024/1/15')).toBeInTheDocument();
      expect(screen.getByText('2024/1/10')).toBeInTheDocument();
    });
  });

  describe('分页功能', () => {
    const manyResults = Array.from({ length: 25 }, (_, index) => ({
      id: `result_${index}`,
      content: `测试结果内容 ${index + 1}`,
      file_path: `/test/result_${index}.md`,
      score: 0.8 - index * 0.01,
      content_type: 'documentation',
      metadata: { source: 'test' },
    }));

    beforeEach(async () => {
      render(<IntelligentResultsPanel results={manyResults} />);
      
      const user = userEvent.setup();
      const listTab = screen.getByRole('tab', { name: /列表视图/ });
      await user.click(listTab);
    });

    it('显示分页控件', () => {
      expect(screen.getByText(/第 1-10 条，共 25 条结果/)).toBeInTheDocument();
    });

    it('可以切换页面', async () => {
      const user = userEvent.setup();
      
      const nextPageButton = screen.getByRole('button', { name: '2' });
      await user.click(nextPageButton);
      
      expect(screen.getByText(/第 11-20 条，共 25 条结果/)).toBeInTheDocument();
    });
  });

  describe('使用store数据', () => {
    it('使用store中的查询结果', () => {
      mockRagStore.queryResults = mockResults;
      mockRagStore.currentQuery = '测试查询';
      
      render(<IntelligentResultsPanel />);
      
      expect(screen.getByText('智能结果分析')).toBeInTheDocument();
      expect(screen.getByText('4')).toBeInTheDocument();
    });

    it('使用store中的agentic结果', () => {
      mockRagStore.agenticResults = {
        success: true,
        query_id: 'test-query',
        results: mockResults,
        processing_time: 1000,
        timestamp: '2024-01-01T00:00:00Z',
      };
      
      render(<IntelligentResultsPanel />);
      
      expect(screen.getByText('智能结果分析')).toBeInTheDocument();
      expect(screen.getByText('4')).toBeInTheDocument();
    });
  });

  describe('自定义类名和样式', () => {
    it('应用自定义className', () => {
      const { container } = render(
        <IntelligentResultsPanel
          className="custom-results-panel"
          results={mockResults}
        />
      );
      
      expect(container.querySelector('.custom-results-panel')).toBeInTheDocument();
    });
  });
});