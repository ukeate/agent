import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Badge } from '../ui/Badge';

interface AlgorithmData {
  algorithm: string;
  average_reward: number;
  total_pulls: number;
  regret: number;
  usage_count: number;
}

interface AlgorithmVisualizationProps {
  data: AlgorithmData[];
  title?: string;
  width?: number;
  height?: number;
}

const AlgorithmVisualization: React.FC<AlgorithmVisualizationProps> = ({
  data,
  title = "算法性能对比",
  width = 600,
  height = 400
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    // 清除之前的绘制
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current);
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // 创建主要的图表容器
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // 设置比例尺
    const xScale = d3.scaleBand()
      .domain(data.map(d => d.algorithm))
      .range([0, chartWidth])
      .padding(0.2);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.average_reward) || 1])
      .range([chartHeight, 0]);

    // 颜色比例尺
    const colorScale = d3.scaleOrdinal()
      .domain(data.map(d => d.algorithm))
      .range(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']);

    // 绘制柱状图
    g.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.algorithm) || 0)
      .attr('y', chartHeight)
      .attr('width', xScale.bandwidth())
      .attr('height', 0)
      .attr('fill', d => colorScale(d.algorithm) as string)
      .attr('opacity', 0.8)
      .transition()
      .duration(800)
      .attr('y', d => yScale(d.average_reward))
      .attr('height', d => chartHeight - yScale(d.average_reward));

    // 添加数值标签
    g.selectAll('.label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', d => (xScale(d.algorithm) || 0) + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.average_reward) - 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .style('fill', '#333')
      .text(d => d.average_reward.toFixed(3));

    // 添加x轴
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .style('text-anchor', 'middle')
      .style('font-size', '11px');

    // 添加y轴
    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale).ticks(5))
      .style('font-size', '11px');

    // 添加y轴标题
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (chartHeight / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', '#666')
      .text('平均奖励');

    // 添加鼠标悬停效果
    g.selectAll('.bar')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('opacity', 1)
          .attr('stroke', '#333')
          .attr('stroke-width', 2);

        // 显示tooltip
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('opacity', 0);

        tooltip.transition()
          .duration(200)
          .style('opacity', 1);

        tooltip.html(`
          <strong>${d.algorithm}</strong><br/>
          平均奖励: ${d.average_reward.toFixed(3)}<br/>
          总选择数: ${d.total_pulls}<br/>
          累积遗憾: ${d.regret.toFixed(3)}<br/>
          使用次数: ${d.usage_count}
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('opacity', 0.8)
          .attr('stroke', 'none');

        d3.selectAll('.tooltip').remove();
      });

  }, [data, width, height]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          {title}
          <div className="flex space-x-2">
            {data.map((d, i) => (
              <Badge key={i} variant="outline" className="text-xs">
                {d.algorithm}: {d.average_reward.toFixed(3)}
              </Badge>
            ))}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ border: '1px solid #e5e5e5', borderRadius: '4px' }}
        />
      </CardContent>
    </Card>
  );
};

export default AlgorithmVisualization;