/**
 * 试验图表组件
 */
import React from 'react';
import { Empty, Spin, Typography } from 'antd';
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const { Title } = Typography;

interface TrialChartProps {
  data: any[];
  chartType: 'optimization-history' | 'parameter-scatter' | 'parameter-importance' | 'multi-metric';
  title?: string;
  loading?: boolean;
  showBestValue?: boolean;
  xParameter?: string;
  yParameter?: string;
  colors?: string[];
  xLabel?: string;
  yLabel?: string;
  valueFormatter?: (value: number) => string;
  onTrialSelect?: (trial: any) => void;
  xAxisType?: 'number' | 'time';
  yAxisType?: 'number' | 'log';
  enableZoom?: boolean;
  onExport?: () => void;
  metrics?: string[];
}

const TrialChart: React.FC<TrialChartProps> = ({
  data,
  chartType,
  title,
  loading,
  showBestValue,
  xParameter,
  yParameter,
  colors = ['#8884d8', '#82ca9d', '#ffc658'],
  xLabel,
  yLabel,
  valueFormatter,
  onTrialSelect,
  xAxisType = 'number',
  yAxisType = 'number',
  enableZoom,
  onExport,
  metrics = []
}) => {
  if (loading) {
    return <div data-testid="loading"><Spin size="large" /></div>;
  }

  if (!data || data.length === 0) {
    return <Empty description="暂无数据" />;
  }

  // 处理参数重要性数据
  if (chartType === 'parameter-importance') {
    return (
      <div>
        {title && <Title level={4}>{title}</Title>}
        {xLabel && <div>{xLabel}</div>}
        {yLabel && <div>{yLabel}</div>}
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="parameter" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="importance" fill={colors[0]} />
          </BarChart>
        </ResponsiveContainer>
        {data.map(item => (
          <div key={item.parameter}>{item.parameter}</div>
        ))}
      </div>
    );
  }

  // 处理散点图
  if (chartType === 'parameter-scatter') {
    const scatterData = data.map(trial => ({
      x: trial.parameters?.[xParameter!] || trial[xParameter!],
      y: trial.parameters?.[yParameter!] || trial[yParameter!],
      ...trial
    }));

    return (
      <div>
        {title && <Title level={4}>{title}</Title>}
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart data={scatterData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" name={xParameter} />
            <YAxis dataKey="y" name={yParameter} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter name="试验" data={scatterData} fill={colors[0]} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // 处理多指标图表
  if (chartType === 'multi-metric') {
    return (
      <div>
        {title && <Title level={4}>{title}</Title>}
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="trial" />
            <YAxis />
            <Tooltip />
            <Legend />
            {metrics.map((metric, index) => (
              <Line 
                key={metric}
                type="monotone" 
                dataKey={`metrics.${metric}`}
                stroke={colors[index % colors.length]} 
                name={metric}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // 默认优化历史图表
  const chartData = data.map((trial, index) => ({
    trial: trial.id || index + 1,
    value: trial.value,
    ...trial
  }));

  // 找出最佳值
  const bestValue = showBestValue ? Math.max(...data.map(t => t.value || 0)) : null;

  return (
    <div>
      {title && <Title level={4}>{title}</Title>}
      {xLabel && <div>{xLabel}</div>}
      {yLabel && <div>{yLabel}</div>}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="trial" type={xAxisType as any} />
          <YAxis type={yAxisType as any} />
          <Tooltip formatter={valueFormatter} />
          <Legend />
          <Line type="monotone" dataKey="value" stroke={colors[0]} />
        </LineChart>
      </ResponsiveContainer>
      {bestValue && <div>{bestValue}</div>}
      {onExport && (
        <button onClick={onExport}>导出</button>
      )}
    </div>
  );
};

export default TrialChart;