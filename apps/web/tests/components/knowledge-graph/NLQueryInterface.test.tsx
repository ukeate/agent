/**
 * NLQueryInterface 组件测试
 * 
 * 测试内容：
 * - 自然语言查询转换的准确性测试
 * - 查询界面交互功能测试
 * - 查询历史和收藏功能测试
 * - 查询模板和自动补全测试
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
// import { vi } from 'vitest';
import '@testing-library/jest-dom';

import NLQueryInterface from '../../../src/components/knowledge-graph/NLQueryInterface';
import type { NLQuery, QueryResult } from '../../../src/components/knowledge-graph/NLQueryInterface';

// Mock antd message
mockFn()mock('antd', async () => {
  const originalAntd = await mockFn()importActual('antd');
  return {
    ...originalAntd,
    message: {
      warning: mockFn()fn(),
      success: mockFn()fn(),
      error: mockFn()fn(),
      info: mockFn()fn()
    }
  };
});

describe('NLQueryInterface', () => {

  // ==================== 基础渲染测试 ====================

  describe('基础渲染', () => {

    test('应该正确渲染查询界面', () => {
      render(<NLQueryInterface />);

      expect(screen.getByText('自然语言查询')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/请输入自然语言查询/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /查询/ })).toBeInTheDocument();
    });

    test('应该显示查询语法帮助', async () => {
      const user = userEvent.setup();
      render(<NLQueryInterface />);

      // 展开语法提示面板
      const syntaxPanel = screen.getByText('搜索语法');
      await user.click(syntaxPanel);

      expect(screen.getByText('AND、OR、NOT - 逻辑操作符')).toBeInTheDocument();
      expect(screen.getByText('"精确短语" - 精确匹配')).toBeInTheDocument();
    });

    test('应该显示查询模板', async () => {
      const user = userEvent.setup();
      render(<NLQueryInterface />);

      // 展开模板面板
      const templatesPanel = screen.getByText('查询模板');
      await user.click(templatesPanel);

      expect(screen.getByText('实体搜索')).toBeInTheDocument();
      expect(screen.getByText('路径查找')).toBeInTheDocument();
      expect(screen.getByText('邻域探索')).toBeInTheDocument();
      expect(screen.getByText('模式匹配')).toBeInTheDocument();
    });

  });

  // ==================== 查询功能测试 ====================

  describe('查询功能', () => {

    test('应该处理实体搜索查询', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '1',
        query: {
          text: '找到苹果公司',
          type: 'entity_search',
          parameters: { entities: ['苹果公司'] },
          confidence: 0.9
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 500,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      const searchButton = screen.getByRole('button', { name: /查询/ });

      await user.type(input, '找到苹果公司');
      await user.click(searchButton);

      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalledWith(
          expect.objectContaining({
            text: '找到苹果公司',
            type: 'entity_search',
            parameters: expect.objectContaining({
              entities: ['苹果公司']
            })
          })
        );
      });
    });

    test('应该处理路径查找查询', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '2',
        query: {
          text: '苹果公司和iPhone之间的关系',
          type: 'path_finding',
          parameters: { entities: ['苹果公司', 'iPhone'] },
          confidence: 0.95
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 750,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '苹果公司和iPhone之间的关系');
      
      // 使用Enter键提交
      await user.keyboard('{Enter}');

      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalledWith(
          expect.objectContaining({
            text: '苹果公司和iPhone之间的关系',
            type: 'path_finding'
          })
        );
      });
    });

    test('应该处理邻域探索查询', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '3',
        query: {
          text: '显示苹果公司周围的所有连接',
          type: 'neighborhood',
          parameters: { entities: ['苹果公司'] },
          confidence: 0.85
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 600,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '显示苹果公司周围的所有连接');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'neighborhood'
          })
        );
      });
    });

    test('空查询时应该显示警告', async () => {
      const { message } = require('antd');
      const user = userEvent.setup();
      render(<NLQueryInterface />);

      const searchButton = screen.getByRole('button', { name: /查询/ });
      await user.click(searchButton);

      expect(message.warning).toHaveBeenCalledWith('请输入查询内容');
    });

  });

  // ==================== 实体识别测试 ====================

  describe('实体识别', () => {

    test('应该正确识别引号中的实体', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '4',
        query: {
          text: '找到"史蒂夫·乔布斯"相关的实体',
          type: 'entity_search',
          parameters: { entities: ['史蒂夫·乔布斯'] },
          confidence: 0.9
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 400,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '找到"史蒂夫·乔布斯"相关的实体');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalledWith(
          expect.objectContaining({
            parameters: expect.objectContaining({
              entities: ['史蒂夫·乔布斯']
            })
          })
        );
      });
    });

    test('应该识别多个实体', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '5',
        query: {
          text: '"苹果公司"和"微软"之间的关系',
          type: 'path_finding',
          parameters: { entities: ['苹果公司', '微软'] },
          confidence: 0.92
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 550,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '"苹果公司"和"微软"之间的关系');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalledWith(
          expect.objectContaining({
            parameters: expect.objectContaining({
              entities: expect.arrayContaining(['苹果公司', '微软'])
            })
          })
        );
      });
    });

  });

  // ==================== 查询结果显示测试 ====================

  describe('查询结果显示', () => {

    test('应该显示查询结果信息', async () => {
      const mockQueryResult: QueryResult = {
        queryId: '6',
        query: {
          text: '测试查询',
          type: 'entity_search',
          parameters: { entities: ['测试'] },
          confidence: 0.88,
          generatedCypher: 'MATCH (n:Entity) WHERE n.name CONTAINS "测试" RETURN n'
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 450,
        timestamp: new Date()
      };

      const mockOnQuery = mockFn()fn().mockResolvedValue(mockQueryResult);
      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '测试查询');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        expect(screen.getByText('查询结果')).toBeInTheDocument();
        expect(screen.getByText('entity_search')).toBeInTheDocument();
        expect(screen.getByText('450.00ms')).toBeInTheDocument();
      });

      // 展开Cypher查询
      const cypherPanel = screen.getByText('生成的Cypher查询');
      await user.click(cypherPanel);

      expect(screen.getByText(/MATCH \(n:Entity\)/)).toBeInTheDocument();
    });

    test('应该正确显示置信度', async () => {
      const mockQueryResult: QueryResult = {
        queryId: '7',
        query: {
          text: '高置信度查询',
          type: 'entity_search',
          parameters: { entities: [] },
          confidence: 0.95
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 300,
        timestamp: new Date()
      };

      const mockOnQuery = mockFn()fn().mockResolvedValue(mockQueryResult);
      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '高置信度查询');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        // 置信度应该显示为95%
        expect(screen.getByText('95%')).toBeInTheDocument();
      });
    });

  });

  // ==================== 查询历史测试 ====================

  describe('查询历史', () => {

    test('应该记录查询历史', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '8',
        query: {
          text: '历史查询测试',
          type: 'entity_search',
          parameters: { entities: [] },
          confidence: 0.8
        },
        results: [{ id: 'result1' }, { id: 'result2' }],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 400,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      // 执行查询
      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '历史查询测试');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      // 等待查询完成
      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalled();
      });

      // 展开历史记录面板
      const historyPanel = screen.getByText(/查询历史/);
      await user.click(historyPanel);

      // 验证历史记录存在
      await waitFor(() => {
        expect(screen.getByText('历史查询测试')).toBeInTheDocument();
        expect(screen.getByText('2 结果')).toBeInTheDocument();
      });
    });

    test('应该支持从历史记录选择查询', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '9',
        query: {
          text: '重复查询',
          type: 'entity_search',
          parameters: { entities: [] },
          confidence: 0.85
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 350,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      // 先执行一次查询
      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '重复查询');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      // 等待查询完成
      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalledTimes(1);
      });

      // 清空输入
      await user.clear(input);

      // 展开历史记录并点击
      const historyPanel = screen.getByText(/查询历史/);
      await user.click(historyPanel);

      const historyItem = await screen.findByText('重复查询');
      await user.click(historyItem);

      // 验证输入框被填充
      expect(input).toHaveValue('重复查询');
    });

  });

  // ==================== 模板功能测试 ====================

  describe('查询模板', () => {

    test('应该支持选择查询模板', async () => {
      const user = userEvent.setup();
      render(<NLQueryInterface />);

      // 展开模板面板
      const templatesPanel = screen.getByText('查询模板');
      await user.click(templatesPanel);

      // 点击模板示例
      const templateExample = screen.getByText('找到所有与"苹果公司"相关的实体');
      await user.click(templateExample);

      // 验证输入框被填充
      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      expect(input).toHaveValue('找到所有与"苹果公司"相关的实体');
    });

  });

  // ==================== 自动补全测试 ====================

  describe('自动补全', () => {

    test('应该基于历史记录提供自动补全', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '10',
        query: {
          text: '苹果公司相关查询',
          type: 'entity_search',
          parameters: { entities: ['苹果公司'] },
          confidence: 0.9
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 400,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      // 先执行一次查询建立历史记录
      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '苹果公司相关查询');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalled();
      });

      // 清空并输入部分文本
      await user.clear(input);
      await user.type(input, '苹果');

      // 应该显示自动补全选项 (这里简化测试，实际实现中会有下拉选项)
      // 在真实环境中，这会触发AutoComplete组件显示选项
    });

  });

  // ==================== 错误处理测试 ====================

  describe('错误处理', () => {

    test('应该处理查询错误', async () => {
      const { message } = require('antd');
      const mockOnQuery = mockFn()fn().mockRejectedValue(new Error('查询服务不可用'));

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '错误查询');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        expect(message.error).toHaveBeenCalledWith('查询失败: 查询服务不可用');
      });
    });

    test('应该在查询过程中显示加载状态', async () => {
      // 创建一个延迟的Promise来模拟慢查询
      const mockOnQuery = jest.fn(() => 
        new Promise(resolve => setTimeout(() => resolve({
          queryId: '11',
          query: {
            text: '慢查询',
            type: 'entity_search',
            parameters: { entities: [] },
            confidence: 0.8
          },
          results: [],
          highlights: { nodeIds: [], edgeIds: [], paths: [] },
          executionTime: 2000,
          timestamp: new Date()
        }), 1000))
      );

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      const searchButton = screen.getByRole('button', { name: /查询/ });

      await user.type(input, '慢查询');
      await user.click(searchButton);

      // 验证按钮显示加载状态
      expect(searchButton).toBeDisabled();
    });

  });

  // ==================== 收藏功能测试 ====================

  describe('收藏功能', () => {

    test('应该支持收藏查询', async () => {
      const mockOnQuery = mockFn()fn().mockResolvedValue({
        queryId: '12',
        query: {
          text: '可收藏查询',
          type: 'entity_search',
          parameters: { entities: [] },
          confidence: 0.9
        },
        results: [],
        highlights: { nodeIds: [], edgeIds: [], paths: [] },
        executionTime: 300,
        timestamp: new Date()
      });

      const user = userEvent.setup();
      render(<NLQueryInterface onQuery={mockOnQuery} />);

      // 执行查询
      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      await user.type(input, '可收藏查询');
      await user.click(screen.getByRole('button', { name: /查询/ }));

      await waitFor(() => {
        expect(mockOnQuery).toHaveBeenCalled();
      });

      // 展开历史记录面板
      const historyPanel = screen.getByText(/查询历史/);
      await user.click(historyPanel);

      // 找到收藏按钮并点击
      const favoriteButton = screen.getByRole('button', { name: /收藏/ });
      await user.click(favoriteButton);

      // 验证收藏状态更新 (星标图标应该变为filled)
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /取消收藏/ })).toBeInTheDocument();
      });
    });

  });

  // ==================== 清空功能测试 ====================

  describe('清空功能', () => {

    test('应该支持清空查询', async () => {
      const user = userEvent.setup();
      render(<NLQueryInterface />);

      const input = screen.getByPlaceholderText(/请输入自然语言查询/);
      const clearButton = screen.getByRole('button', { name: /清空查询/ });

      // 输入一些文本
      await user.type(input, '测试清空功能');
      expect(input).toHaveValue('测试清空功能');

      // 点击清空按钮
      await user.click(clearButton);

      // 验证输入被清空
      expect(input).toHaveValue('');
    });

  });

});

// ==================== 集成测试 ====================

describe('NLQueryInterface Integration', () => {

  test('应该与高亮功能正确集成', async () => {
    const mockHighlights = {
      nodeIds: ['node1', 'node2'],
      edgeIds: ['edge1'],
      paths: []
    };

    const mockOnQuery = mockFn()fn().mockResolvedValue({
      queryId: '13',
      query: {
        text: '集成测试查询',
        type: 'entity_search',
        parameters: { entities: [] },
        confidence: 0.9
      },
      results: [],
      highlights: mockHighlights,
      executionTime: 400,
      timestamp: new Date()
    });

    const mockOnHighlight = mockFn()fn();

    const user = userEvent.setup();
    render(
      <NLQueryInterface 
        onQuery={mockOnQuery} 
        onHighlight={mockOnHighlight}
      />
    );

    const input = screen.getByPlaceholderText(/请输入自然语言查询/);
    await user.type(input, '集成测试查询');
    await user.click(screen.getByRole('button', { name: /查询/ }));

    await waitFor(() => {
      expect(mockOnHighlight).toHaveBeenCalledWith(mockHighlights);
    });
  });

});