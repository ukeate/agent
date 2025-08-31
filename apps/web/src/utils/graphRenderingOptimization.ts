/**
 * 图谱渲染性能优化工具
 * 
 * 功能包括：
 * - 大图谱的虚拟化渲染策略
 * - LOD(Level of Detail)渲染机制
 * - 渐进式加载和按需数据获取
 * - 图谱导出和快照功能
 * - 内存管理和性能监控
 */

import type { GraphData, GraphNode, GraphEdge } from '../components/knowledge-graph/GraphVisualization';
import { Core as CytoscapeCore } from 'cytoscape';

// ==================== 性能配置常量 ====================

const PERFORMANCE_THRESHOLDS = {
  SMALL_GRAPH: 100,      // 小规模图谱节点数
  MEDIUM_GRAPH: 1000,    // 中等规模图谱节点数
  LARGE_GRAPH: 5000,     // 大规模图谱节点数
  VERY_LARGE_GRAPH: 15000, // 超大规模图谱节点数
  
  LOD_DISTANCE_THRESHOLD: 0.5,    // LOD切换距离阈值
  VIEWPORT_BUFFER: 100,           // 视口缓冲区像素
  MAX_VISIBLE_NODES: 2000,        // 最大可见节点数
  CHUNK_SIZE: 500,                // 分块加载大小
  
  ANIMATION_DURATION_THRESHOLD: 1000, // 动画时长阈值
  DEBOUNCE_DELAY: 150,                 // 防抖延迟
  THROTTLE_INTERVAL: 16,               // 节流间隔(60fps)
};

// ==================== 性能监控类 ====================

export class PerformanceMonitor {
  private metrics: {
    renderTime: number[];
    frameRate: number[];
    memoryUsage: number[];
    nodeCount: number;
    edgeCount: number;
  } = {
    renderTime: [],
    frameRate: [],
    memoryUsage: [],
    nodeCount: 0,
    edgeCount: 0
  };

  private frameCount = 0;
  private lastFrameTime = 0;
  private animationFrameId?: number;

  startMonitoring(): void {
    this.frameCount = 0;
    this.lastFrameTime = performance.now();
    this.measureFrameRate();
  }

  stopMonitoring(): void {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }

  recordRenderTime(duration: number): void {
    this.metrics.renderTime.push(duration);
    // 保持最近100次记录
    if (this.metrics.renderTime.length > 100) {
      this.metrics.renderTime.shift();
    }
  }

  recordMemoryUsage(): void {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      this.metrics.memoryUsage.push(memory.usedJSHeapSize);
      if (this.metrics.memoryUsage.length > 100) {
        this.metrics.memoryUsage.shift();
      }
    }
  }

  setGraphSize(nodeCount: number, edgeCount: number): void {
    this.metrics.nodeCount = nodeCount;
    this.metrics.edgeCount = edgeCount;
  }

  getMetrics() {
    const avgRenderTime = this.metrics.renderTime.length > 0 
      ? this.metrics.renderTime.reduce((a, b) => a + b, 0) / this.metrics.renderTime.length 
      : 0;
    
    const avgFrameRate = this.metrics.frameRate.length > 0
      ? this.metrics.frameRate.reduce((a, b) => a + b, 0) / this.metrics.frameRate.length
      : 0;

    const currentMemory = this.metrics.memoryUsage.length > 0
      ? this.metrics.memoryUsage[this.metrics.memoryUsage.length - 1]
      : 0;

    return {
      averageRenderTime: avgRenderTime,
      averageFrameRate: avgFrameRate,
      currentMemoryUsage: currentMemory,
      nodeCount: this.metrics.nodeCount,
      edgeCount: this.metrics.edgeCount,
      performanceScore: this.calculatePerformanceScore(avgRenderTime, avgFrameRate)
    };
  }

  private measureFrameRate(): void {
    const measure = (currentTime: number) => {
      this.frameCount++;
      
      if (currentTime - this.lastFrameTime >= 1000) {
        const fps = this.frameCount / ((currentTime - this.lastFrameTime) / 1000);
        this.metrics.frameRate.push(fps);
        
        if (this.metrics.frameRate.length > 60) {
          this.metrics.frameRate.shift();
        }
        
        this.frameCount = 0;
        this.lastFrameTime = currentTime;
      }
      
      this.animationFrameId = requestAnimationFrame(measure);
    };
    
    this.animationFrameId = requestAnimationFrame(measure);
  }

  private calculatePerformanceScore(renderTime: number, frameRate: number): number {
    // 性能评分：0-100
    let score = 100;
    
    // 渲染时间惩罚
    if (renderTime > 100) score -= Math.min(50, (renderTime - 100) / 10);
    
    // 帧率惩罚
    if (frameRate < 30) score -= Math.min(30, (30 - frameRate) * 2);
    
    return Math.max(0, score);
  }
}

// ==================== 虚拟化渲染器 ====================

export class VirtualizationRenderer {
  private viewport: {
    x: number;
    y: number;
    width: number;
    height: number;
    zoom: number;
  } = { x: 0, y: 0, width: 0, height: 0, zoom: 1 };

  private visibleNodes = new Set<string>();
  private visibleEdges = new Set<string>();
  private nodeQuadTree: QuadTree | null = null;

  updateViewport(x: number, y: number, width: number, height: number, zoom: number): void {
    this.viewport = { x, y, width, height, zoom };
  }

  buildSpatialIndex(nodes: GraphNode[]): void {
    const bounds = this.calculateBounds(nodes);
    this.nodeQuadTree = new QuadTree(bounds);
    
    nodes.forEach(node => {
      if (node.position) {
        this.nodeQuadTree!.insert({
          id: node.id,
          x: node.position.x,
          y: node.position.y,
          width: node.size || 20,
          height: node.size || 20
        });
      }
    });
  }

  getVisibleNodes(): Set<string> {
    if (!this.nodeQuadTree) return new Set();

    const buffer = PERFORMANCE_THRESHOLDS.VIEWPORT_BUFFER / this.viewport.zoom;
    const queryBounds = {
      x: this.viewport.x - buffer,
      y: this.viewport.y - buffer,
      width: this.viewport.width + 2 * buffer,
      height: this.viewport.height + 2 * buffer
    };

    const visibleItems = this.nodeQuadTree.query(queryBounds);
    this.visibleNodes.clear();
    
    visibleItems.forEach(item => {
      this.visibleNodes.add(item.id);
    });

    // 限制最大可见节点数
    if (this.visibleNodes.size > PERFORMANCE_THRESHOLDS.MAX_VISIBLE_NODES) {
      const nodes = Array.from(this.visibleNodes).slice(0, PERFORMANCE_THRESHOLDS.MAX_VISIBLE_NODES);
      this.visibleNodes = new Set(nodes);
    }

    return this.visibleNodes;
  }

  getVisibleEdges(edges: GraphEdge[]): Set<string> {
    this.visibleEdges.clear();
    
    edges.forEach(edge => {
      if (this.visibleNodes.has(edge.source) && this.visibleNodes.has(edge.target)) {
        this.visibleEdges.add(edge.id);
      }
    });

    return this.visibleEdges;
  }

  private calculateBounds(nodes: GraphNode[]): { x: number; y: number; width: number; height: number } {
    if (nodes.length === 0) {
      return { x: 0, y: 0, width: 1000, height: 1000 };
    }

    const positions = nodes
      .filter(node => node.position)
      .map(node => node.position!);

    const minX = Math.min(...positions.map(p => p.x));
    const maxX = Math.max(...positions.map(p => p.x));
    const minY = Math.min(...positions.map(p => p.y));
    const maxY = Math.max(...positions.map(p => p.y));

    return {
      x: minX - 100,
      y: minY - 100,
      width: (maxX - minX) + 200,
      height: (maxY - minY) + 200
    };
  }
}

// ==================== LOD渲染管理器 ====================

export class LODRenderer {
  private lodLevels = new Map<string, number>();
  
  calculateLOD(nodeId: string, viewport: any, nodePosition: { x: number; y: number }): number {
    // 计算节点到视口中心的距离
    const centerX = viewport.x + viewport.width / 2;
    const centerY = viewport.y + viewport.height / 2;
    const distance = Math.sqrt(
      Math.pow(nodePosition.x - centerX, 2) + Math.pow(nodePosition.y - centerY, 2)
    );

    // 基于距离和缩放级别确定LOD
    const normalizedDistance = distance / (viewport.zoom * Math.max(viewport.width, viewport.height));
    
    let lodLevel: number;
    if (normalizedDistance < 0.2) {
      lodLevel = 3; // 高细节
    } else if (normalizedDistance < 0.5) {
      lodLevel = 2; // 中等细节
    } else if (normalizedDistance < 1.0) {
      lodLevel = 1; // 低细节
    } else {
      lodLevel = 0; // 最低细节或不渲染
    }

    this.lodLevels.set(nodeId, lodLevel);
    return lodLevel;
  }

  getNodeStyle(nodeId: string, baseStyle: any): any {
    const lodLevel = this.lodLevels.get(nodeId) || 0;
    
    switch (lodLevel) {
      case 3: // 高细节
        return {
          ...baseStyle,
          'label': 'data(label)',
          'font-size': '12px',
          'border-width': 2,
          'text-outline-width': 1
        };
      
      case 2: // 中等细节
        return {
          ...baseStyle,
          'label': 'data(label)',
          'font-size': '10px',
          'border-width': 1,
          'text-outline-width': 0
        };
      
      case 1: // 低细节
        return {
          ...baseStyle,
          'label': '',
          'border-width': 1
        };
      
      case 0: // 最低细节
      default:
        return {
          ...baseStyle,
          'label': '',
          'border-width': 0,
          'width': 4,
          'height': 4
        };
    }
  }

  getEdgeStyle(edgeId: string, sourceNodeId: string, targetNodeId: string, baseStyle: any): any {
    const sourceLOD = this.lodLevels.get(sourceNodeId) || 0;
    const targetLOD = this.lodLevels.get(targetNodeId) || 0;
    const minLOD = Math.min(sourceLOD, targetLOD);
    
    switch (minLOD) {
      case 3:
      case 2:
        return {
          ...baseStyle,
          'label': 'data(label)',
          'font-size': '8px'
        };
      
      case 1:
        return {
          ...baseStyle,
          'label': '',
          'width': 'mapData(weight, 0, 1, 1, 3)'
        };
      
      case 0:
      default:
        return {
          ...baseStyle,
          'label': '',
          'width': 1,
          'opacity': 0.3
        };
    }
  }
}

// ==================== 渐进式加载器 ====================

export class ProgressiveLoader {
  private loadingQueue: Array<{
    priority: number;
    data: GraphNode[] | GraphEdge[];
    type: 'nodes' | 'edges';
  }> = [];
  
  private isLoading = false;

  addToQueue(
    data: GraphNode[] | GraphEdge[], 
    type: 'nodes' | 'edges',
    priority: number = 0
  ): void {
    this.loadingQueue.push({ data, type, priority });
    this.loadingQueue.sort((a, b) => b.priority - a.priority);
  }

  async loadNextChunk(cy: CytoscapeCore): Promise<boolean> {
    if (this.isLoading || this.loadingQueue.length === 0) {
      return false;
    }

    this.isLoading = true;

    try {
      const chunk = this.loadingQueue.shift()!;
      const chunkData = this.chunkArray(chunk.data as any[], PERFORMANCE_THRESHOLDS.CHUNK_SIZE)[0];

      if (chunk.type === 'nodes') {
        await this.loadNodesChunk(cy, chunkData as GraphNode[]);
      } else {
        await this.loadEdgesChunk(cy, chunkData as GraphEdge[]);
      }

      // 如果还有剩余数据，重新加入队列
      const remainingData = (chunk.data as any[]).slice(PERFORMANCE_THRESHOLDS.CHUNK_SIZE);
      if (remainingData.length > 0) {
        this.loadingQueue.unshift({
          ...chunk,
          data: remainingData
        });
      }

      return true;
    } finally {
      this.isLoading = false;
    }
  }

  private async loadNodesChunk(cy: CytoscapeCore, nodes: GraphNode[]): Promise<void> {
    return new Promise(resolve => {
      const elements = nodes.map(node => ({
        group: 'nodes' as const,
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          ...node.properties
        },
        position: node.position
      }));

      cy.add(elements);
      
      // 使用requestAnimationFrame确保DOM更新完成
      requestAnimationFrame(() => {
        resolve();
      });
    });
  }

  private async loadEdgesChunk(cy: CytoscapeCore, edges: GraphEdge[]): Promise<void> {
    return new Promise(resolve => {
      const elements = edges.map(edge => ({
        group: 'edges' as const,
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          type: edge.type,
          ...edge.properties
        }
      }));

      cy.add(elements);
      
      requestAnimationFrame(() => {
        resolve();
      });
    });
  }

  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  hasMoreChunks(): boolean {
    return this.loadingQueue.length > 0;
  }

  clear(): void {
    this.loadingQueue = [];
    this.isLoading = false;
  }
}

// ==================== 简单四叉树实现 ====================

interface QuadTreeItem {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Bounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

class QuadTree {
  private bounds: Bounds;
  private items: QuadTreeItem[] = [];
  private children: QuadTree[] = [];
  private maxItems = 10;
  private maxDepth = 5;
  private depth: number;

  constructor(bounds: Bounds, depth: number = 0) {
    this.bounds = bounds;
    this.depth = depth;
  }

  insert(item: QuadTreeItem): void {
    if (!this.contains(this.bounds, item)) {
      return;
    }

    if (this.items.length < this.maxItems || this.depth >= this.maxDepth) {
      this.items.push(item);
      return;
    }

    if (this.children.length === 0) {
      this.subdivide();
    }

    for (const child of this.children) {
      child.insert(item);
    }
  }

  query(queryBounds: Bounds): QuadTreeItem[] {
    const result: QuadTreeItem[] = [];

    if (!this.intersects(this.bounds, queryBounds)) {
      return result;
    }

    for (const item of this.items) {
      if (this.intersects(queryBounds, item)) {
        result.push(item);
      }
    }

    for (const child of this.children) {
      result.push(...child.query(queryBounds));
    }

    return result;
  }

  private contains(bounds: Bounds, item: QuadTreeItem): boolean {
    return (
      item.x >= bounds.x &&
      item.y >= bounds.y &&
      item.x + item.width <= bounds.x + bounds.width &&
      item.y + item.height <= bounds.y + bounds.height
    );
  }

  private intersects(bounds1: Bounds, bounds2: Bounds | QuadTreeItem): boolean {
    return !(
      bounds2.x > bounds1.x + bounds1.width ||
      bounds2.x + bounds2.width < bounds1.x ||
      bounds2.y > bounds1.y + bounds1.height ||
      bounds2.y + bounds2.height < bounds1.y
    );
  }

  private subdivide(): void {
    const halfWidth = this.bounds.width / 2;
    const halfHeight = this.bounds.height / 2;

    this.children = [
      new QuadTree({
        x: this.bounds.x,
        y: this.bounds.y,
        width: halfWidth,
        height: halfHeight
      }, this.depth + 1),
      new QuadTree({
        x: this.bounds.x + halfWidth,
        y: this.bounds.y,
        width: halfWidth,
        height: halfHeight
      }, this.depth + 1),
      new QuadTree({
        x: this.bounds.x,
        y: this.bounds.y + halfHeight,
        width: halfWidth,
        height: halfHeight
      }, this.depth + 1),
      new QuadTree({
        x: this.bounds.x + halfWidth,
        y: this.bounds.y + halfHeight,
        width: halfWidth,
        height: halfHeight
      }, this.depth + 1)
    ];

    for (const item of this.items) {
      for (const child of this.children) {
        child.insert(item);
      }
    }

    this.items = [];
  }
}

// ==================== 工具函数 ====================

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(null, args), delay);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  interval: number
): (...args: Parameters<T>) => void {
  let lastCallTime = 0;
  
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCallTime >= interval) {
      lastCallTime = now;
      func.apply(null, args);
    }
  };
}

export function getOptimalRenderingEngine(nodeCount: number, edgeCount: number): 'canvas' | 'webgl' | 'svg' {
  if (nodeCount > PERFORMANCE_THRESHOLDS.LARGE_GRAPH || edgeCount > PERFORMANCE_THRESHOLDS.LARGE_GRAPH * 5) {
    return 'webgl';
  } else if (nodeCount > PERFORMANCE_THRESHOLDS.MEDIUM_GRAPH || edgeCount > PERFORMANCE_THRESHOLDS.MEDIUM_GRAPH * 3) {
    return 'canvas';
  } else {
    return 'svg';
  }
}

export function calculateGraphComplexity(data: GraphData): {
  complexity: 'low' | 'medium' | 'high' | 'very_high';
  score: number;
  recommendations: string[];
} {
  const nodeCount = data.nodes.length;
  const edgeCount = data.edges.length;
  const density = data.metadata.density;
  
  let score = 0;
  const recommendations: string[] = [];
  
  // 节点数量评分
  if (nodeCount > PERFORMANCE_THRESHOLDS.VERY_LARGE_GRAPH) {
    score += 4;
    recommendations.push('启用虚拟化渲染');
    recommendations.push('使用WebGL渲染引擎');
  } else if (nodeCount > PERFORMANCE_THRESHOLDS.LARGE_GRAPH) {
    score += 3;
    recommendations.push('启用LOD渲染');
  } else if (nodeCount > PERFORMANCE_THRESHOLDS.MEDIUM_GRAPH) {
    score += 2;
    recommendations.push('考虑使用Canvas渲染');
  } else {
    score += 1;
  }
  
  // 边数量评分
  if (edgeCount > nodeCount * 10) {
    score += 2;
    recommendations.push('启用边过滤');
  } else if (edgeCount > nodeCount * 5) {
    score += 1;
  }
  
  // 密度评分
  if (density > 0.1) {
    score += 2;
    recommendations.push('启用边聚类');
  } else if (density > 0.05) {
    score += 1;
  }
  
  let complexity: 'low' | 'medium' | 'high' | 'very_high';
  if (score >= 7) {
    complexity = 'very_high';
  } else if (score >= 5) {
    complexity = 'high';
  } else if (score >= 3) {
    complexity = 'medium';
  } else {
    complexity = 'low';
  }
  
  return { complexity, score, recommendations };
}

// ==================== 导出和快照工具 ====================

export class ExportManager {
  
  static async exportToPNG(
    cy: CytoscapeCore,
    options: {
      full?: boolean;
      bg?: string;
      scale?: number;
      maxWidth?: number;
      maxHeight?: number;
    } = {}
  ): Promise<string> {
    const defaultOptions = {
      full: true,
      bg: 'white',
      scale: 1,
      maxWidth: 4000,
      maxHeight: 4000,
      ...options
    };

    return cy.png(defaultOptions);
  }

  static async exportToSVG(
    cy: CytoscapeCore,
    options: { full?: boolean; scale?: number } = {}
  ): Promise<string> {
    const defaultOptions = {
      full: true,
      scale: 1,
      ...options
    };

    return cy.container()(defaultOptions);
  }

  static exportToJSON(data: GraphData): string {
    return JSON.stringify(data, null, 2);
  }

  static exportToCSV(data: GraphData): { nodes: string; edges: string } {
    // 导出节点CSV
    const nodeHeaders = ['id', 'label', 'type', 'confidence', 'x', 'y'];
    const nodeRows = data.nodes.map(node => [
      node.id,
      node.label,
      node.type,
      node.metadata.confidence,
      node.position?.x || '',
      node.position?.y || ''
    ]);
    
    const nodesCSV = [
      nodeHeaders.join(','),
      ...nodeRows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    // 导出边CSV
    const edgeHeaders = ['id', 'source', 'target', 'type', 'label', 'confidence'];
    const edgeRows = data.edges.map(edge => [
      edge.id,
      edge.source,
      edge.target,
      edge.type,
      edge.label,
      edge.metadata.confidence
    ]);
    
    const edgesCSV = [
      edgeHeaders.join(','),
      ...edgeRows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    return { nodes: nodesCSV, edges: edgesCSV };
  }
}

// ==================== 默认导出 ====================

export default {
  PerformanceMonitor,
  VirtualizationRenderer,
  LODRenderer,
  ProgressiveLoader,
  ExportManager,
  debounce,
  throttle,
  getOptimalRenderingEngine,
  calculateGraphComplexity,
  PERFORMANCE_THRESHOLDS
};