/**
 * 指标图表组件
 */
import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  Brush,
  ErrorBar,
  Cell
} from 'recharts';
import { Box, Paper, Typography, Chip, Stack } from '@mui/material';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

// 图表类型
export type ChartType = 'line' | 'area' | 'bar' | 'scatter' | 'composed';

// 数据点
export interface DataPoint {
  x: string | number;
  y: number;
  yMin?: number;
  yMax?: number;
  variant?: string;
  category?: string;
  metadata?: any;
}

// 数据系列
export interface DataSeries {
  name: string;
  data: DataPoint[];
  color?: string;
  type?: 'line' | 'bar' | 'area' | 'scatter';
  yAxisId?: string;
  strokeDasharray?: string;
  opacity?: number;
}

// 图表配置
export interface ChartConfig {
  showGrid?: boolean;
  showLegend?: boolean;
  showBrush?: boolean;
  showTooltip?: boolean;
  showReferenceLine?: boolean;
  referenceValue?: number;
  referenceLabel?: string;
  confidenceInterval?: boolean;
  xAxisLabel?: string;
  yAxisLabel?: string;
  yAxisDomain?: [number | 'auto', number | 'auto'];
  dateFormat?: string;
  numberFormat?: string;
  height?: number;
}

// 统计信息
export interface Statistics {
  mean?: number;
  median?: number;
  stdDev?: number;
  min?: number;
  max?: number;
  pValue?: number;
  significant?: boolean;
}

interface MetricChartProps {
  title?: string;
  subtitle?: string;
  type?: ChartType;
  series: DataSeries[];
  config?: ChartConfig;
  statistics?: Statistics;
  annotations?: Array<{
    type: 'line' | 'area' | 'text';
    value?: number | [number, number];
    label?: string;
    color?: string;
  }>;
  onDataPointClick?: (point: DataPoint) => void;
}

// 默认颜色
const DEFAULT_COLORS = [
  '#3f51b5', // primary
  '#4caf50', // success
  '#ff9800', // warning
  '#f44336', // error
  '#2196f3', // info
  '#9c27b0', // purple
  '#00bcd4', // cyan
  '#8bc34a'  // light green
];

const MetricChart: React.FC<MetricChartProps> = ({
  title,
  subtitle,
  type = 'line',
  series,
  config = {},
  statistics,
  annotations = [],
  onDataPointClick
}) => {
  const {
    showGrid = true,
    showLegend = true,
    showBrush = false,
    showTooltip = true,
    showReferenceLine = false,
    referenceValue = 0,
    referenceLabel = '',
    confidenceInterval = false,
    xAxisLabel,
    yAxisLabel,
    yAxisDomain,
    dateFormat = 'MM-dd',
    numberFormat = '.2f',
    height = 300
  } = config;

  // 处理数据
  const chartData = useMemo(() => {
    if (series.length === 0) return [];

    // 合并所有系列的数据点
    const dataMap = new Map<string | number, any>();
    
    series.forEach((s, index) => {
      const color = s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length];
      
      s.data.forEach(point => {
        const key = point.x;
        if (!dataMap.has(key)) {
          dataMap.set(key, { x: key });
        }
        
        const item = dataMap.get(key);
        item[s.name] = point.y;
        
        if (confidenceInterval && point.yMin !== undefined && point.yMax !== undefined) {
          item[`${s.name}_min`] = point.yMin;
          item[`${s.name}_max`] = point.yMax;
          item[`${s.name}_error`] = [point.yMin, point.yMax];
        }
        
        if (point.metadata) {
          item[`${s.name}_metadata`] = point.metadata;
        }
      });
    });

    return Array.from(dataMap.values()).sort((a, b) => {
      if (typeof a.x === 'number' && typeof b.x === 'number') {
        return a.x - b.x;
      }
      return String(a.x).localeCompare(String(b.x));
    });
  }, [series, confidenceInterval]);

  // 自定义Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload) return null;

    return (
      <Paper sx={{ p: 1.5 }}>
        <Typography variant="body2" gutterBottom>
          {label}
        </Typography>
        {payload.map((entry: any, index: number) => (
          <Stack key={index} direction="row" spacing={1} alignItems="center">
            <Box
              sx={{
                width: 12,
                height: 12,
                bgcolor: entry.color,
                borderRadius: '50%'
              }}
            />
            <Typography variant="body2">
              {entry.name}: {Number(entry.value).toFixed(2)}
            </Typography>
          </Stack>
        ))}
      </Paper>
    );
  };

  // 格式化X轴
  const formatXAxis = (value: any) => {
    if (typeof value === 'number') {
      return value.toFixed(0);
    }
    if (value instanceof Date || !isNaN(Date.parse(value))) {
      return format(new Date(value), dateFormat, { locale: zhCN });
    }
    return value;
  };

  // 格式化Y轴
  const formatYAxis = (value: number) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toFixed(1);
  };

  // 渲染图表
  const renderChart = () => {
    const commonProps = {
      data: chartData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 }
    };

    const commonAxisProps = {
      stroke: '#666',
      style: { fontSize: 12 }
    };

    switch (type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />}
            <XAxis
              dataKey="x"
              tickFormatter={formatXAxis}
              label={xAxisLabel ? { value: xAxisLabel, position: 'insideBottom', offset: -5 } : undefined}
              {...commonAxisProps}
            />
            <YAxis
              tickFormatter={formatYAxis}
              label={yAxisLabel ? { value: yAxisLabel, angle: -90, position: 'insideLeft' } : undefined}
              domain={yAxisDomain}
              {...commonAxisProps}
            />
            {showTooltip && <Tooltip content={<CustomTooltip />} />}
            {showLegend && <Legend />}
            {series.map((s, index) => (
              <Line
                key={s.name}
                type="monotone"
                dataKey={s.name}
                stroke={s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
                strokeWidth={2}
                strokeDasharray={s.strokeDasharray}
                dot={chartData.length < 20}
                activeDot={onDataPointClick ? { r: 8, onClick: onDataPointClick } : undefined}
              >
                {confidenceInterval && (
                  <ErrorBar
                    dataKey={`${s.name}_error`}
                    width={4}
                    stroke={s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
                    opacity={0.3}
                  />
                )}
              </Line>
            ))}
            {showReferenceLine && (
              <ReferenceLine
                y={referenceValue}
                stroke="red"
                strokeDasharray="5 5"
                label={referenceLabel}
              />
            )}
            {annotations.map((ann, index) => {
              if (ann.type === 'line' && typeof ann.value === 'number') {
                return (
                  <ReferenceLine
                    key={index}
                    y={ann.value}
                    stroke={ann.color || 'gray'}
                    strokeDasharray="3 3"
                    label={ann.label}
                  />
                );
              }
              if (ann.type === 'area' && Array.isArray(ann.value)) {
                return (
                  <ReferenceArea
                    key={index}
                    y1={ann.value[0]}
                    y2={ann.value[1]}
                    fill={ann.color || 'gray'}
                    fillOpacity={0.1}
                    label={ann.label}
                  />
                );
              }
              return null;
            })}
            {showBrush && <Brush dataKey="x" height={30} stroke="#8884d8" />}
          </LineChart>
        );

      case 'area':
        return (
          <AreaChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />}
            <XAxis
              dataKey="x"
              tickFormatter={formatXAxis}
              label={xAxisLabel ? { value: xAxisLabel, position: 'insideBottom', offset: -5 } : undefined}
              {...commonAxisProps}
            />
            <YAxis
              tickFormatter={formatYAxis}
              label={yAxisLabel ? { value: yAxisLabel, angle: -90, position: 'insideLeft' } : undefined}
              domain={yAxisDomain}
              {...commonAxisProps}
            />
            {showTooltip && <Tooltip content={<CustomTooltip />} />}
            {showLegend && <Legend />}
            {series.map((s, index) => (
              <Area
                key={s.name}
                type="monotone"
                dataKey={s.name}
                stroke={s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
                fill={s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
                fillOpacity={s.opacity || 0.6}
                strokeWidth={2}
              />
            ))}
            {showBrush && <Brush dataKey="x" height={30} stroke="#8884d8" />}
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />}
            <XAxis
              dataKey="x"
              tickFormatter={formatXAxis}
              label={xAxisLabel ? { value: xAxisLabel, position: 'insideBottom', offset: -5 } : undefined}
              {...commonAxisProps}
            />
            <YAxis
              tickFormatter={formatYAxis}
              label={yAxisLabel ? { value: yAxisLabel, angle: -90, position: 'insideLeft' } : undefined}
              domain={yAxisDomain}
              {...commonAxisProps}
            />
            {showTooltip && <Tooltip content={<CustomTooltip />} />}
            {showLegend && <Legend />}
            {series.map((s, index) => (
              <Bar
                key={s.name}
                dataKey={s.name}
                fill={s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
                onClick={onDataPointClick}
              >
                {confidenceInterval && (
                  <ErrorBar
                    dataKey={`${s.name}_error`}
                    width={4}
                    stroke={s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
                  />
                )}
              </Bar>
            ))}
            {showReferenceLine && (
              <ReferenceLine
                y={referenceValue}
                stroke="red"
                strokeDasharray="5 5"
                label={referenceLabel}
              />
            )}
          </BarChart>
        );

      case 'scatter':
        return (
          <ScatterChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />}
            <XAxis
              dataKey="x"
              tickFormatter={formatXAxis}
              label={xAxisLabel ? { value: xAxisLabel, position: 'insideBottom', offset: -5 } : undefined}
              {...commonAxisProps}
            />
            <YAxis
              tickFormatter={formatYAxis}
              label={yAxisLabel ? { value: yAxisLabel, angle: -90, position: 'insideLeft' } : undefined}
              domain={yAxisDomain}
              {...commonAxisProps}
            />
            {showTooltip && <Tooltip content={<CustomTooltip />} />}
            {showLegend && <Legend />}
            {series.map((s, index) => (
              <Scatter
                key={s.name}
                name={s.name}
                data={s.data.map(d => ({ x: d.x, y: d.y }))}
                fill={s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}
                onClick={onDataPointClick}
              />
            ))}
          </ScatterChart>
        );

      case 'composed':
        return (
          <ComposedChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />}
            <XAxis
              dataKey="x"
              tickFormatter={formatXAxis}
              label={xAxisLabel ? { value: xAxisLabel, position: 'insideBottom', offset: -5 } : undefined}
              {...commonAxisProps}
            />
            <YAxis
              tickFormatter={formatYAxis}
              label={yAxisLabel ? { value: yAxisLabel, angle: -90, position: 'insideLeft' } : undefined}
              domain={yAxisDomain}
              {...commonAxisProps}
            />
            {showTooltip && <Tooltip content={<CustomTooltip />} />}
            {showLegend && <Legend />}
            {series.map((s, index) => {
              const color = s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length];
              
              switch (s.type) {
                case 'bar':
                  return (
                    <Bar
                      key={s.name}
                      dataKey={s.name}
                      fill={color}
                      yAxisId={s.yAxisId}
                    />
                  );
                case 'area':
                  return (
                    <Area
                      key={s.name}
                      type="monotone"
                      dataKey={s.name}
                      fill={color}
                      stroke={color}
                      fillOpacity={0.6}
                      yAxisId={s.yAxisId}
                    />
                  );
                case 'line':
                default:
                  return (
                    <Line
                      key={s.name}
                      type="monotone"
                      dataKey={s.name}
                      stroke={color}
                      strokeWidth={2}
                      dot={false}
                      yAxisId={s.yAxisId}
                    />
                  );
              }
            })}
            {showBrush && <Brush dataKey="x" height={30} stroke="#8884d8" />}
          </ComposedChart>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      {(title || subtitle || statistics) && (
        <Box sx={{ mb: 2 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
            <Box>
              {title && (
                <Typography variant="h6" gutterBottom>
                  {title}
                </Typography>
              )}
              {subtitle && (
                <Typography variant="body2" color="text.secondary">
                  {subtitle}
                </Typography>
              )}
            </Box>
            {statistics && (
              <Stack direction="row" spacing={1}>
                {statistics.mean !== undefined && (
                  <Chip
                    label={`均值: ${statistics.mean.toFixed(2)}`}
                    size="small"
                    variant="outlined"
                  />
                )}
                {statistics.pValue !== undefined && (
                  <Chip
                    label={`P值: ${statistics.pValue.toFixed(4)}`}
                    size="small"
                    color={statistics.significant ? 'success' : 'default'}
                    variant={statistics.significant ? 'filled' : 'outlined'}
                  />
                )}
              </Stack>
            )}
          </Stack>
        </Box>
      )}
      <ResponsiveContainer width="100%" height={height}>
        {renderChart()}
      </ResponsiveContainer>
    </Box>
  );
};

export default MetricChart;