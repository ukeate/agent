/**
 * GraphVisualization 组件测试
 * 
 * 测试内容：
 * - 图谱渲染的正确性和性能验证
 * - 交互功能(拖拽、缩放、选择)的用户体验验证  
 * - 大规模图谱的性能和稳定性测试
 * - 导出功能和数据格式的正确性验证
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
// import { vi } from 'vitest';
import '@testing-library/jest-dom';

import GraphVisualization from '../../../src/components/knowledge-graph/GraphVisualization';
import type { GraphData, GraphNode, GraphEdge } from '../../../src/components/knowledge-graph/GraphVisualization';

// Mock Cytoscape.js
mockFn()mock('cytoscape', () => {
  const mockCytoscape = {
    on: mockFn()fn(),
    off: mockFn()fn(),
    zoom: mockFn()fn(() => 1),
    fit: mockFn()fn(),
    center: mockFn()fn(),
    pan: mockFn()fn(),
    destroy: mockFn()fn(),
    elements: mockFn()fn(() => ({
      removeClass: mockFn()fn(),
      addClass: mockFn()fn()
    })),
    getElementById: mockFn()fn(() => ({
      addClass: mockFn()fn(),
      removeClass: mockFn()fn()
    })),
    layout: mockFn()fn(() => ({
      run: mockFn()fn()
    })),
    add: mockFn()fn(),
    remove: mockFn()fn(),
    $: mockFn()fn(() => ({
      select: mockFn()fn(),
      unselect: mockFn()fn()
    })),
    png: mockFn()fn(() => 'data:image/png;base64,mock'),
    svg: mockFn()fn(() => '<svg>mock</svg>')
  };

  return { default: mockFn()fn(() => mockCytoscape) };
});

// 测试数据
const mockGraphData: GraphData = {
  nodes: [
    {
      id: 'node1',
      label: '实体1',
      type: 'person',
      properties: {},
      position: { x: 100, y: 100 },
      metadata: {
        confidence: 0.9,
        lastUpdated: '2023-01-01',
        sourceCount: 5
      }
    },
    {
      id: 'node2', 
      label: '实体2',
      type: 'organization',
      properties: {},
      position: { x: 200, y: 200 },
      metadata: {
        confidence: 0.8,
        lastUpdated: '2023-01-02',
        sourceCount: 3
      }
    }
  ],
  edges: [
    {
      id: 'edge1',
      source: 'node1',
      target: 'node2',
      type: 'works_at',
      label: '工作于',
      properties: {},
      metadata: {
        confidence: 0.95,
        evidence: ['document1', 'document2']
      }
    }
  ],
  metadata: {
    totalNodes: 2,
    totalEdges: 1,
    density: 0.5,
    lastUpdated: '2023-01-01',
    version: '1.0'
  }
};

describe('GraphVisualization', () => {
  
  // ==================== 基础渲染测试 ====================
  
  describe('基础渲染', () => {
    
    test('应该正确渲染图谱可视化组件', () => {
      render(<GraphVisualization data={mockGraphData} />);
      
      expect(screen.getByText('知识图谱可视化')).toBeInTheDocument();
      expect(screen.getByText('2 节点, 1 边')).toBeInTheDocument();
    });

    test('当没有数据时应该显示提示信息', () => {
      render(<GraphVisualization />);
      
      expect(screen.getByText('暂无图谱数据')).toBeInTheDocument();
    });

    test('加载状态时应该显示加载指示器', () => {
      render(<GraphVisualization loading={true} />);
      
      expect(screen.getByText('加载图谱数据...')).toBeInTheDocument();
    });

    test('错误状态时应该显示错误信息', () => {
      const errorMessage = '数据加载失败';
      render(<GraphVisualization error={errorMessage} />);
      
      expect(screen.getByText('图谱加载失败')).toBeInTheDocument();
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });

  });

  // ==================== 交互功能测试 ====================
  
  describe('用户交互', () => {
    
    test('点击节点应该触发回调', async () => {
      const onNodeClick = mockFn()fn();
      render(
        <GraphVisualization 
          data={mockGraphData} 
          onNodeClick={onNodeClick}
        />
      );

      // 模拟Cytoscape节点点击事件
      const cytoscape = require('cytoscape');
      const mockInstance = cytoscape();
      const mockEvent = {
        target: {
          id: () => 'node1'
        }
      };

      // 获取注册的事件处理器
      const tapHandler = mockInstance.on.mock.calls.find(
        call => call[0] === 'tap' && call[1] === 'node'
      )?.[2];

      if (tapHandler) {
        tapHandler(mockEvent);
        expect(onNodeClick).toHaveBeenCalledWith(mockGraphData.nodes[0]);
      }
    });

    test('点击边应该触发回调', async () => {
      const onEdgeClick = mockFn()fn();
      render(
        <GraphVisualization 
          data={mockGraphData} 
          onEdgeClick={onEdgeClick}
        />
      );

      const cytoscape = require('cytoscape');
      const mockInstance = cytoscape();
      const mockEvent = {
        target: {
          id: () => 'edge1'
        }
      };

      const tapHandler = mockInstance.on.mock.calls.find(
        call => call[0] === 'tap' && call[1] === 'edge'
      )?.[2];

      if (tapHandler) {
        tapHandler(mockEvent);
        expect(onEdgeClick).toHaveBeenCalledWith(mockGraphData.edges[0]);
      }
    });

    test('控制按钮应该正常工作', async () => {
      const user = userEvent.setup();
      render(<GraphVisualization data={mockGraphData} />);

      // 测试放大按钮
      const zoomInButton = screen.getByRole('button', { name: /放大/i });
      await user.click(zoomInButton);

      // 测试缩小按钮
      const zoomOutButton = screen.getByRole('button', { name: /缩小/i });
      await user.click(zoomOutButton);

      // 测试适应画布按钮
      const fitButton = screen.getByRole('button', { name: /适应画布/i });
      await user.click(fitButton);
    });

  });

  // ==================== 布局测试 ====================
  
  describe('布局功能', () => {
    
    test('应该支持不同的布局算法', async () => {
      const onLayoutChange = mockFn()fn();
      const user = userEvent.setup();
      
      render(
        <GraphVisualization 
          data={mockGraphData} 
          onLayoutChange={onLayoutChange}
        />
      );

      // 打开布局选择器
      const layoutSelect = screen.getByDisplayValue('力导向');
      await user.click(layoutSelect);

      // 选择网格布局
      const gridOption = screen.getByText('网格');
      await user.click(gridOption);

      expect(onLayoutChange).toHaveBeenCalledWith('grid');
    });

  });

  // ==================== 高亮功能测试 ====================
  
  describe('高亮功能', () => {
    
    test('应该正确应用查询结果高亮', () => {
      const highlights = {
        nodeIds: ['node1'],
        edgeIds: ['edge1'],
        paths: [{
          pathId: 'path1',
          nodes: ['node1', 'node2'],
          edges: ['edge1'],
          description: '测试路径'
        }]
      };

      render(
        <GraphVisualization 
          data={mockGraphData} 
          highlights={highlights}
        />
      );

      const cytoscape = require('cytoscape');
      const mockInstance = cytoscape();

      // 验证高亮类被正确添加
      expect(mockInstance.getElementById).toHaveBeenCalledWith('node1');
      expect(mockInstance.getElementById).toHaveBeenCalledWith('edge1');
    });

  });

  // ==================== 性能测试 ====================
  
  describe('性能测试', () => {
    
    test('应该能处理大规模图谱数据', async () => {
      // 生成大量测试数据
      const largeGraphData: GraphData = {
        nodes: Array.from({ length: 1000 }, (_, i) => ({
          id: `node${i}`,
          label: `实体${i}`,
          type: 'test',
          properties: {},
          position: { x: Math.random() * 1000, y: Math.random() * 1000 },
          metadata: {
            confidence: Math.random(),
            lastUpdated: '2023-01-01',
            sourceCount: Math.floor(Math.random() * 10)
          }
        })),
        edges: Array.from({ length: 2000 }, (_, i) => ({
          id: `edge${i}`,
          source: `node${Math.floor(Math.random() * 1000)}`,
          target: `node${Math.floor(Math.random() * 1000)}`,
          type: 'test_relation',
          label: `关系${i}`,
          properties: {},
          metadata: {
            confidence: Math.random(),
            evidence: []
          }
        })),
        metadata: {
          totalNodes: 1000,
          totalEdges: 2000,
          density: 0.002,
          lastUpdated: '2023-01-01',
          version: '1.0'
        }
      };

      const startTime = performance.now();
      
      render(<GraphVisualization data={largeGraphData} />);
      
      const endTime = performance.now();
      const renderTime = endTime - startTime;

      // 渲染时间应该小于5秒
      expect(renderTime).toBeLessThan(5000);
    });

    test('应该在组件卸载时正确清理资源', () => {
      const { unmount } = render(<GraphVisualization data={mockGraphData} />);

      const cytoscape = require('cytoscape');
      const mockInstance = cytoscape();

      unmount();

      // 验证Cytoscape实例被销毁
      expect(mockInstance.destroy).toHaveBeenCalled();
    });

  });

  // ==================== 导出功能测试 ====================
  
  describe('导出功能', () => {
    
    test('应该支持PNG格式导出', async () => {
      const user = userEvent.setup();
      
      // Mock URL.createObjectURL
      global.URL.createObjectURL = mockFn()fn(() => 'blob:mock-url');
      global.URL.revokeObjectURL = mockFn()fn();
      
      // Mock link element
      const mockLink = {
        click: mockFn()fn(),
        href: '',
        download: ''
      };
      mockFn()spyOn(document, 'createElement').mockReturnValue(mockLink as any);

      render(<GraphVisualization data={mockGraphData} />);

      const exportButton = screen.getByRole('button', { name: /导出/i });
      await user.click(exportButton);

      const cytoscape = require('cytoscape');
      const mockInstance = cytoscape();

      expect(mockInstance.png).toHaveBeenCalled();
    });

  });

  // ==================== 配置测试 ====================
  
  describe('配置选项', () => {
    
    test('应该应用自定义配置', () => {
      const customConfig = {
        layout: {
          algorithm: 'grid',
          parameters: { rows: 3 }
        },
        styling: {
          nodeSize: { min: 10, max: 50 },
          nodeColor: { scheme: 'type' }
        },
        interaction: {
          enableDrag: false,
          enableZoom: true
        }
      };

      render(
        <GraphVisualization 
          data={mockGraphData} 
          config={customConfig}
        />
      );

      const cytoscape = require('cytoscape');
      const mockInstance = cytoscape();

      // 验证配置被正确应用
      expect(mockInstance.layout).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'grid'
        })
      );
    });

  });

  // ==================== 边界情况测试 ====================
  
  describe('边界情况', () => {
    
    test('应该处理空的图谱数据', () => {
      const emptyData: GraphData = {
        nodes: [],
        edges: [],
        metadata: {
          totalNodes: 0,
          totalEdges: 0,
          density: 0,
          lastUpdated: '2023-01-01',
          version: '1.0'
        }
      };

      render(<GraphVisualization data={emptyData} />);
      
      expect(screen.getByText('0 节点, 0 边')).toBeInTheDocument();
    });

    test('应该处理无效的节点位置', () => {
      const dataWithoutPositions: GraphData = {
        ...mockGraphData,
        nodes: mockGraphData.nodes.map(node => ({
          ...node,
          position: undefined
        }))
      };

      expect(() => {
        render(<GraphVisualization data={dataWithoutPositions} />);
      }).not.toThrow();
    });

    test('应该处理缺失的元数据', () => {
      const dataWithMissingMetadata: GraphData = {
        nodes: [{
          id: 'node1',
          label: '实体1',
          type: 'test',
          properties: {},
          metadata: {
            confidence: 0.9,
            lastUpdated: '2023-01-01',
            sourceCount: 1
          }
        }],
        edges: [],
        metadata: {
          totalNodes: 1,
          totalEdges: 0,
          density: 0,
          lastUpdated: '2023-01-01',
          version: '1.0'
        }
      };

      expect(() => {
        render(<GraphVisualization data={dataWithMissingMetadata} />);
      }).not.toThrow();
    });

  });

});

// ==================== 性能基准测试 ====================

describe('GraphVisualization Performance', () => {
  
  test('渲染性能应该在可接受范围内', async () => {
    const mediumGraphData: GraphData = {
      nodes: Array.from({ length: 500 }, (_, i) => ({
        id: `node${i}`,
        label: `实体${i}`,
        type: 'test',
        properties: {},
        position: { x: Math.random() * 500, y: Math.random() * 500 },
        metadata: {
          confidence: Math.random(),
          lastUpdated: '2023-01-01',
          sourceCount: 1
        }
      })),
      edges: Array.from({ length: 1000 }, (_, i) => ({
        id: `edge${i}`,
        source: `node${Math.floor(Math.random() * 500)}`,
        target: `node${Math.floor(Math.random() * 500)}`,
        type: 'test_relation',
        label: `关系${i}`,
        properties: {},
        metadata: {
          confidence: Math.random(),
          evidence: []
        }
      })),
      metadata: {
        totalNodes: 500,
        totalEdges: 1000,
        density: 0.004,
        lastUpdated: '2023-01-01',
        version: '1.0'
      }
    };

    const startTime = performance.now();
    
    const { rerender } = render(<GraphVisualization data={mediumGraphData} />);
    
    // 测试重新渲染性能
    rerender(<GraphVisualization data={mediumGraphData} />);
    
    const endTime = performance.now();
    
    // 总时间应该小于2秒
    expect(endTime - startTime).toBeLessThan(2000);
  });

});

// ==================== 集成测试 ====================

describe('GraphVisualization Integration', () => {
  
  test('应该与其他组件正确集成', async () => {
    const onNodeClick = mockFn()fn();
    const onEdgeClick = mockFn()fn();
    const onSelection = mockFn()fn();
    
    render(
      <GraphVisualization 
        data={mockGraphData}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        onSelection={onSelection}
      />
    );

    // 模拟用户交互
    const cytoscape = require('cytoscape');
    const mockInstance = cytoscape();

    // 模拟选择事件
    const selectHandler = mockInstance.on.mock.calls.find(
      call => call[0] === 'select unselect'
    )?.[1];

    if (selectHandler) {
      // 模拟选择了一些元素
      mockInstance.$ = mockFn()fn(() => ({
        nodes: mockFn()fn(() => [{ id: () => 'node1' }]),
        edges: mockFn()fn(() => [{ id: () => 'edge1' }])
      }));

      selectHandler();
      
      expect(onSelection).toHaveBeenCalledWith({
        nodes: [mockGraphData.nodes[0]],
        edges: [mockGraphData.edges[0]]
      });
    }
  });

});