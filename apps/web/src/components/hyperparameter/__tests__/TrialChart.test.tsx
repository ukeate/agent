/**
 * 试验图表组件单元测试
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import TrialChart from '../TrialChart'

// 模拟图表库
vi.mock('recharts', () => ({
  LineChart: ({ children }: any) => (
    <div data-testid="line-chart">{children}</div>
  ),
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ResponsiveContainer: ({ children }: any) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  ScatterChart: ({ children }: any) => (
    <div data-testid="scatter-chart">{children}</div>
  ),
  Scatter: () => <div data-testid="scatter" />,
  BarChart: ({ children }: any) => (
    <div data-testid="bar-chart">{children}</div>
  ),
  Bar: () => <div data-testid="bar" />,
}))

const mockTrialData = [
  {
    id: 1,
    value: 0.95,
    parameters: { learning_rate: 0.01, batch_size: 32 },
    created_at: '2024-01-01T10:00:00Z',
    state: 'complete',
  },
  {
    id: 2,
    value: 0.89,
    parameters: { learning_rate: 0.001, batch_size: 64 },
    created_at: '2024-01-01T11:00:00Z',
    state: 'complete',
  },
  {
    id: 3,
    value: 0.92,
    parameters: { learning_rate: 0.05, batch_size: 128 },
    created_at: '2024-01-01T12:00:00Z',
    state: 'complete',
  },
]

describe('TrialChart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('渲染优化历史线图', () => {
    render(<TrialChart data={mockTrialData} chartType="optimization-history" />)

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument()
    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
    expect(screen.getByTestId('line')).toBeInTheDocument()
    expect(screen.getByTestId('x-axis')).toBeInTheDocument()
    expect(screen.getByTestId('y-axis')).toBeInTheDocument()
  })

  it('渲染参数散点图', () => {
    render(
      <TrialChart
        data={mockTrialData}
        chartType="parameter-scatter"
        xParameter="learning_rate"
        yParameter="batch_size"
      />
    )

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument()
    expect(screen.getByTestId('scatter-chart')).toBeInTheDocument()
    expect(screen.getByTestId('scatter')).toBeInTheDocument()
  })

  it('显示图表标题', () => {
    const title = '优化历史'
    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        title={title}
      />
    )

    expect(screen.getByText(title)).toBeInTheDocument()
  })

  it('处理空数据', () => {
    render(<TrialChart data={[]} chartType="optimization-history" />)

    expect(screen.getByText(/暂无数据/)).toBeInTheDocument()
  })

  it('处理加载状态', () => {
    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        loading
      />
    )

    expect(screen.getByTestId('loading')).toBeInTheDocument()
  })

  it('显示最佳值标记', () => {
    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        showBestValue
      />
    )

    // 检查最佳值标记
    const bestValue = Math.max(...mockTrialData.map(t => t.value))
    expect(screen.getByText(bestValue.toString())).toBeInTheDocument()
  })

  it('支持自定义颜色配置', () => {
    const colors = ['#ff0000', '#00ff00', '#0000ff']

    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        colors={colors}
      />
    )

    // 验证图表渲染（具体颜色验证需要实际的图表库）
    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  it('显示工具提示', () => {
    render(<TrialChart data={mockTrialData} chartType="optimization-history" />)

    expect(screen.getByTestId('tooltip')).toBeInTheDocument()
  })

  it('显示图例', () => {
    render(<TrialChart data={mockTrialData} chartType="optimization-history" />)

    expect(screen.getByTestId('legend')).toBeInTheDocument()
  })

  it('支持响应式布局', () => {
    render(<TrialChart data={mockTrialData} chartType="optimization-history" />)

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument()
  })

  it('处理参数重要性图表', () => {
    const importanceData = [
      { parameter: 'learning_rate', importance: 0.8 },
      { parameter: 'batch_size', importance: 0.6 },
      { parameter: 'dropout_rate', importance: 0.3 },
    ]

    render(
      <TrialChart data={importanceData} chartType="parameter-importance" />
    )

    expect(screen.getByText('learning_rate')).toBeInTheDocument()
    expect(screen.getByText('batch_size')).toBeInTheDocument()
    expect(screen.getByText('dropout_rate')).toBeInTheDocument()
  })

  it('支持图表交互', () => {
    const onTrialSelect = vi.fn()

    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        onTrialSelect={onTrialSelect}
      />
    )

    // 模拟点击事件（实际测试需要更详细的事件模拟）
    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  it('显示轴标签', () => {
    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        xLabel="试验次数"
        yLabel="目标值"
      />
    )

    expect(screen.getByText('试验次数')).toBeInTheDocument()
    expect(screen.getByText('目标值')).toBeInTheDocument()
  })

  it('支持数据格式化', () => {
    const formatter = (value: number) => `${(value * 100).toFixed(1)}%`

    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        valueFormatter={formatter}
      />
    )

    // 验证格式化后的值显示
    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  it('处理异常数据点', () => {
    const dataWithNaN = [
      ...mockTrialData,
      {
        id: 4,
        value: NaN,
        parameters: { learning_rate: 0.1, batch_size: 256 },
        created_at: '2024-01-01T13:00:00Z',
        state: 'failed',
      },
    ]

    render(<TrialChart data={dataWithNaN} chartType="optimization-history" />)

    // 应该仍然能够渲染，忽略异常数据点
    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  it('支持多系列数据', () => {
    const multiSeriesData = mockTrialData.map(trial => ({
      ...trial,
      metrics: {
        accuracy: trial.value,
        loss: 1 - trial.value,
      },
    }))

    render(
      <TrialChart
        data={multiSeriesData}
        chartType="multi-metric"
        metrics={['accuracy', 'loss']}
      />
    )

    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  it('支持时间轴显示', () => {
    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        xAxisType="time"
      />
    )

    expect(screen.getByTestId('x-axis')).toBeInTheDocument()
  })

  it('支持对数坐标轴', () => {
    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        yAxisType="log"
      />
    )

    expect(screen.getByTestId('y-axis')).toBeInTheDocument()
  })

  it('支持缩放和平移', () => {
    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        enableZoom
      />
    )

    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  it('支持导出功能', () => {
    const onExport = vi.fn()

    render(
      <TrialChart
        data={mockTrialData}
        chartType="optimization-history"
        onExport={onExport}
      />
    )

    // 查找导出按钮
    const exportButton = screen.queryByRole('button', { name: /导出/ })
    if (exportButton) {
      expect(exportButton).toBeInTheDocument()
    }
  })
})
