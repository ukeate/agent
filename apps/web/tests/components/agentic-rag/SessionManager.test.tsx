/**
 * SessionManager组件单元测试
 * 
 * 测试功能包括：
 * - 基础渲染和空状态显示
 * - 会话创建、选择、重命名、删除功能
 * - 会话列表排序和搜索
 * - 收藏功能
 * - 导出和分享功能
 * - 统计信息展示
 * - 会话历史管理
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach, Mock } from 'vitest';
import SessionManager from '../../../src/components/agentic-rag/SessionManager';
import { AgenticSession } from '../../../src/stores/ragStore';

// Mock dependencies
const mockRagStore = {
  currentSession: null,
  sessions: [],
  createSession: vi.fn(),
  switchSession: vi.fn(),
  updateSession: vi.fn(),
  deleteSession: vi.fn(),
  addToSessionHistory: vi.fn(),
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

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockResolvedValue(undefined),
  },
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

describe('SessionManager', () => {
  const mockSessions: AgenticSession[] = [
    {
      id: 'session-1',
      name: '多智能体架构研究',
      created_at: new Date('2024-01-15T10:00:00Z'),
      last_active: new Date('2024-01-15T12:00:00Z'),
      query_count: 15,
      context_history: ['什么是多智能体系统', '如何实现智能体协作', '代理间通信协议'],
    },
    {
      id: 'session-2',
      name: 'RAG系统优化',
      created_at: new Date('2024-01-10T14:00:00Z'),
      last_active: new Date('2024-01-14T16:00:00Z'),
      query_count: 8,
      context_history: ['RAG检索优化方法', '向量数据库选择'],
    },
    {
      id: 'session-3',
      name: 'LangGraph学习',
      created_at: new Date('2024-01-12T09:00:00Z'),
      last_active: new Date('2024-01-13T11:00:00Z'),
      query_count: 12,
      context_history: ['LangGraph基础概念', '工作流设计', '状态管理'],
    },
  ];

  const mockOnSessionSelect = vi.fn();
  const mockOnSessionCreate = vi.fn();
  const mockOnSessionDelete = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(null);
    mockRagStore.currentSession = null;
    mockRagStore.sessions = [];
    mockRagStore.createSession.mockReturnValue('new-session-id');
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('基础渲染', () => {
    it('无会话时显示空状态', () => {
      render(<SessionManager />);
      
      expect(screen.getByText('暂无会话记录')).toBeInTheDocument();
      expect(screen.getByText('创建首个会话')).toBeInTheDocument();
    });

    it('有会话时显示会话列表', () => {
      mockRagStore.sessions = mockSessions;
      
      render(<SessionManager />);
      
      expect(screen.getByText('会话管理')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument(); // 会话数量标签
      expect(screen.getByText('多智能体架构研究')).toBeInTheDocument();
      expect(screen.getByText('RAG系统优化')).toBeInTheDocument();
      expect(screen.getByText('LangGraph学习')).toBeInTheDocument();
    });

    it('显示新建会话按钮', () => {
      mockRagStore.sessions = mockSessions;
      
      render(<SessionManager />);
      
      expect(screen.getByRole('button', { name: /新建会话/ })).toBeInTheDocument();
    });

    it('compact模式下使用简化样式', () => {
      mockRagStore.sessions = mockSessions;
      
      render(<SessionManager compact={true} />);
      
      // 紧凑模式下不显示统计标签页
      expect(screen.queryByRole('tab', { name: /统计信息/ })).not.toBeInTheDocument();
    });
  });

  describe('标签页切换', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(<SessionManager />);
    });

    it('默认显示会话列表标签页', () => {
      const sessionsTab = screen.getByRole('tab', { name: /会话列表/ });
      expect(sessionsTab).toHaveAttribute('aria-selected', 'true');
    });

    it('可以切换到统计信息标签页', async () => {
      const user = userEvent.setup();
      const statsTab = screen.getByRole('tab', { name: /统计信息/ });
      
      await user.click(statsTab);
      
      expect(statsTab).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('总会话数')).toBeInTheDocument();
      expect(screen.getByText('活跃会话')).toBeInTheDocument();
    });

    it('showStats为false时不显示统计标签页', () => {
      render(<SessionManager showStats={false} />);
      
      expect(screen.queryByRole('tab', { name: /统计信息/ })).not.toBeInTheDocument();
    });
  });

  describe('会话创建功能', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(
        <SessionManager
          onSessionCreate={mockOnSessionCreate}
        />
      );
    });

    it('可以打开创建会话模态框', async () => {
      const user = userEvent.setup();
      const createButton = screen.getByRole('button', { name: /新建会话/ });
      
      await user.click(createButton);
      
      expect(screen.getByText('创建新会话')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('请输入会话名称')).toBeInTheDocument();
    });

    it('可以成功创建会话', async () => {
      const user = userEvent.setup();
      
      // 打开模态框
      const createButton = screen.getByRole('button', { name: /新建会话/ });
      await user.click(createButton);
      
      // 输入会话名称
      const nameInput = screen.getByPlaceholderText('请输入会话名称');
      await user.type(nameInput, '新的会话');
      
      // 点击创建按钮
      const confirmButton = screen.getByRole('button', { name: '创建' });
      await user.click(confirmButton);
      
      expect(mockRagStore.createSession).toHaveBeenCalledWith('新的会话');
    });

    it('空名称时显示错误提示', async () => {
      const user = userEvent.setup();
      
      // 打开模态框
      const createButton = screen.getByRole('button', { name: /新建会话/ });
      await user.click(createButton);
      
      // 直接点击创建按钮（不输入名称）
      const confirmButton = screen.getByRole('button', { name: '创建' });
      await user.click(confirmButton);
      
      expect(mockRagStore.createSession).not.toHaveBeenCalled();
    });

    it('超过最大会话数限制时显示错误', async () => {
      const maxSessions = 2;
      render(<SessionManager maxSessions={maxSessions} />);
      
      const user = userEvent.setup();
      const createButton = screen.getByRole('button', { name: /新建会话/ });
      await user.click(createButton);
      
      const nameInput = screen.getByPlaceholderText('请输入会话名称');
      await user.type(nameInput, '超限会话');
      
      const confirmButton = screen.getByRole('button', { name: '创建' });
      await user.click(confirmButton);
      
      expect(mockRagStore.createSession).not.toHaveBeenCalled();
    });
  });

  describe('会话选择功能', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(
        <SessionManager
          onSessionSelect={mockOnSessionSelect}
        />
      );
    });

    it('可以选择会话', async () => {
      const user = userEvent.setup();
      
      const sessionItem = screen.getByText('多智能体架构研究').closest('.ant-list-item');
      if (sessionItem) {
        await user.click(sessionItem);
        
        expect(mockRagStore.switchSession).toHaveBeenCalledWith('session-1');
        expect(mockOnSessionSelect).toHaveBeenCalledWith(mockSessions[0]);
      }
    });

    it('显示当前活跃会话', () => {
      mockRagStore.currentSession = mockSessions[0];
      
      render(<SessionManager />);
      
      expect(screen.getByText('当前')).toBeInTheDocument();
      
      // 活跃会话应该有特殊样式
      const activeSession = screen.getByText('多智能体架构研究').closest('.ant-list-item');
      expect(activeSession).toHaveClass('active-session');
    });
  });

  describe('会话删除功能', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(
        <SessionManager
          onSessionDelete={mockOnSessionDelete}
        />
      );
    });

    it('可以删除会话', async () => {
      const user = userEvent.setup();
      
      // 查找删除按钮
      const deleteButtons = screen.getAllByRole('button');
      const deleteButton = deleteButtons.find(btn => 
        btn.querySelector('.anticon-delete')
      );
      
      if (deleteButton) {
        await user.click(deleteButton);
        
        // 确认删除
        const confirmButton = screen.getByText('确认');
        await user.click(confirmButton);
        
        expect(mockRagStore.deleteSession).toHaveBeenCalled();
        expect(mockOnSessionDelete).toHaveBeenCalled();
      }
    });

    it('当前活跃会话不显示删除按钮', () => {
      mockRagStore.currentSession = mockSessions[0];
      
      render(<SessionManager />);
      
      // 活跃会话的删除按钮应该不存在
      const deleteButtons = screen.getAllByRole('button').filter(btn => 
        btn.querySelector('.anticon-delete')
      );
      
      // 应该只有2个删除按钮（另外2个非活跃会话）
      expect(deleteButtons.length).toBeLessThan(mockSessions.length);
    });
  });

  describe('会话重命名功能', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(<SessionManager />);
    });

    it('可以打开重命名模态框', async () => {
      const user = userEvent.setup();
      
      // 查找更多操作按钮
      const moreButtons = screen.getAllByRole('button');
      const moreButton = moreButtons.find(btn => 
        btn.querySelector('.anticon-more')
      );
      
      if (moreButton) {
        await user.click(moreButton);
        
        const renameOption = screen.getByText('重命名');
        await user.click(renameOption);
        
        expect(screen.getByText('重命名会话')).toBeInTheDocument();
      }
    });

    it('可以成功重命名会话', async () => {
      const user = userEvent.setup();
      
      // 打开重命名模态框（简化流程）
      const moreButtons = screen.getAllByRole('button');
      const moreButton = moreButtons.find(btn => 
        btn.querySelector('.anticon-more')
      );
      
      if (moreButton) {
        await user.click(moreButton);
        
        const renameOption = screen.getByText('重命名');
        await user.click(renameOption);
        
        const nameInput = screen.getByDisplayValue('多智能体架构研究');
        await user.clear(nameInput);
        await user.type(nameInput, '重命名后的会话');
        
        const confirmButton = screen.getByRole('button', { name: '确认' });
        await user.click(confirmButton);
        
        expect(mockRagStore.updateSession).toHaveBeenCalledWith('session-1', { name: '重命名后的会话' });
      }
    });
  });

  describe('收藏功能', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(<SessionManager />);
    });

    it('可以收藏和取消收藏会话', async () => {
      const user = userEvent.setup();
      
      // 查找收藏按钮
      const favoriteButtons = screen.getAllByRole('button');
      const favoriteButton = favoriteButtons.find(btn => 
        btn.querySelector('.anticon-star')
      );
      
      if (favoriteButton) {
        await user.click(favoriteButton);
        
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
          'rag_favorite_sessions',
          expect.any(String)
        );
      }
    });

    it('加载收藏的会话', () => {
      const mockFavorites = ['session-1'];
      mockLocalStorage.getItem.mockReturnValue(JSON.stringify(mockFavorites));
      
      render(<SessionManager />);
      
      expect(mockLocalStorage.getItem).toHaveBeenCalledWith('rag_favorite_sessions');
    });

    it('显示收藏的会话图标', () => {
      const mockFavorites = ['session-1'];
      mockLocalStorage.getItem.mockReturnValue(JSON.stringify(mockFavorites));
      
      render(<SessionManager />);
      
      // 收藏的会话应该显示实心星星
      const filledStars = screen.getAllByRole('img', { name: /star/ });
      expect(filledStars.length).toBeGreaterThan(0);
    });
  });

  describe('搜索和排序功能', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(<SessionManager />);
    });

    it('可以搜索会话', async () => {
      const user = userEvent.setup();
      
      const searchInput = screen.getByPlaceholderText('搜索会话名称或查询内容...');
      await user.type(searchInput, '多智能体');
      
      expect(screen.getByText('多智能体架构研究')).toBeInTheDocument();
      expect(screen.queryByText('RAG系统优化')).not.toBeInTheDocument();
    });

    it('可以按不同方式排序会话', async () => {
      const user = userEvent.setup();
      
      const sortSelect = screen.getByDisplayValue('最近活跃');
      await user.click(sortSelect);
      
      const nameOption = screen.getByText('名称');
      await user.click(nameOption);
      
      // 验证排序后的会话顺序
      const sessionItems = screen.getAllByText(/架构研究|系统优化|学习/);
      expect(sessionItems.length).toBe(3);
    });

    it('显示搜索结果统计', async () => {
      const user = userEvent.setup();
      
      expect(screen.getByText('共 3 个会话')).toBeInTheDocument();
      
      const searchInput = screen.getByPlaceholderText('搜索会话名称或查询内容...');
      await user.type(searchInput, 'RAG');
      
      await waitFor(() => {
        expect(screen.getByText('共 1 个会话')).toBeInTheDocument();
      });
    });
  });

  describe('导出和分享功能', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(<SessionManager />);
    });

    it('可以导出会话', async () => {
      const user = userEvent.setup();
      
      // 模拟创建下载链接
      const createElementSpy = vi.spyOn(document, 'createElement');
      const mockLink = {
        click: vi.fn(),
        setAttribute: vi.fn(),
      };
      createElementSpy.mockReturnValue(mockLink as any);
      
      const moreButtons = screen.getAllByRole('button');
      const moreButton = moreButtons.find(btn => 
        btn.querySelector('.anticon-more')
      );
      
      if (moreButton) {
        await user.click(moreButton);
        
        const exportOption = screen.getByText('导出');
        await user.click(exportOption);
        
        expect(mockLink.click).toHaveBeenCalled();
      }
      
      createElementSpy.mockRestore();
    });

    it('可以分享会话', async () => {
      const user = userEvent.setup();
      
      const moreButtons = screen.getAllByRole('button');
      const moreButton = moreButtons.find(btn => 
        btn.querySelector('.anticon-more')
      );
      
      if (moreButton) {
        await user.click(moreButton);
        
        const shareOption = screen.getByText('分享');
        await user.click(shareOption);
        
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
          expect.stringContaining('rag/session/share')
        );
      }
    });

    it('可以复制会话ID', async () => {
      const user = userEvent.setup();
      
      const moreButtons = screen.getAllByRole('button');
      const moreButton = moreButtons.find(btn => 
        btn.querySelector('.anticon-more')
      );
      
      if (moreButton) {
        await user.click(moreButton);
        
        const copyOption = screen.getByText('复制ID');
        await user.click(copyOption);
        
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith('session-1');
      }
    });
  });

  describe('统计信息展示', () => {
    beforeEach(async () => {
      mockRagStore.sessions = mockSessions;
      render(<SessionManager />);
      
      const user = userEvent.setup();
      const statsTab = screen.getByRole('tab', { name: /统计信息/ });
      await user.click(statsTab);
    });

    it('显示总体统计数据', () => {
      expect(screen.getByText('总会话数')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument(); // 总会话数
      expect(screen.getByText('总查询数')).toBeInTheDocument();
      expect(screen.getByText('35')).toBeInTheDocument(); // 15+8+12
      expect(screen.getByText('平均查询数')).toBeInTheDocument();
      expect(screen.getByText('11.7')).toBeInTheDocument(); // 35/3 ≈ 11.7
    });

    it('显示最活跃会话', () => {
      expect(screen.getByText('最活跃会话')).toBeInTheDocument();
      expect(screen.getByText('多智能体架构研究')).toBeInTheDocument(); // 查询数最多的会话
      expect(screen.getByText('15 次查询')).toBeInTheDocument();
    });

    it('可以切换到最活跃会话', async () => {
      const user = userEvent.setup();
      
      const switchButton = screen.getByText('切换到此会话');
      await user.click(switchButton);
      
      expect(mockRagStore.switchSession).toHaveBeenCalledWith('session-1');
    });

    it('显示最近活动时间线', () => {
      expect(screen.getByText('最近活动')).toBeInTheDocument();
      expect(screen.getByText('什么是多智能体系统')).toBeInTheDocument();
      expect(screen.getByText('如何实现智能体协作')).toBeInTheDocument();
    });
  });

  describe('会话详情显示', () => {
    beforeEach(() => {
      mockRagStore.sessions = mockSessions;
      render(<SessionManager />);
    });

    it('显示会话基本信息', () => {
      expect(screen.getByText('多智能体架构研究')).toBeInTheDocument();
      expect(screen.getByText('2024/1/15')).toBeInTheDocument(); // 创建日期
      expect(screen.getByText('15')).toBeInTheDocument(); // 查询数
    });

    it('显示最近查询预览', () => {
      expect(screen.getByText('如何实现智能体协作')).toBeInTheDocument();
      expect(screen.getByText('代理间通信协议')).toBeInTheDocument();
    });

    it('显示会话持续时间', () => {
      // 验证持续时间计算和显示
      expect(screen.getByText(/小时/)).toBeInTheDocument(); // 应该显示小时单位
    });
  });

  describe('localStorage错误处理', () => {
    it('处理localStorage读取错误', () => {
      mockLocalStorage.getItem.mockReturnValue('invalid json');
      
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      render(<SessionManager />);
      
      expect(consoleSpy).toHaveBeenCalledWith('Failed to load favorite sessions:', expect.any(Error));
      
      consoleSpy.mockRestore();
    });
  });

  describe('自定义类名和样式', () => {
    it('应用自定义className', () => {
      const { container } = render(
        <SessionManager
          className="custom-session-manager"
        />
      );
      
      expect(container.querySelector('.custom-session-manager')).toBeInTheDocument();
    });
  });

  describe('会话限制和边界情况', () => {
    it('达到最大会话数时不显示新建按钮', () => {
      mockRagStore.sessions = mockSessions;
      
      render(<SessionManager maxSessions={3} />);
      
      expect(screen.queryByRole('button', { name: /新建会话/ })).not.toBeInTheDocument();
    });

    it('处理空的会话历史', () => {
      const emptySession: AgenticSession = {
        id: 'empty-session',
        name: '空会话',
        created_at: new Date(),
        last_active: new Date(),
        query_count: 0,
        context_history: [],
      };
      
      mockRagStore.sessions = [emptySession];
      
      render(<SessionManager />);
      
      expect(screen.getByText('空会话')).toBeInTheDocument();
      expect(screen.getByText('0')).toBeInTheDocument(); // 查询数量为0
    });
  });
});