import React from 'react';
import { Card, Statistic, Row, Col, Typography, Tag, Progress, Tooltip } from 'antd';
import { DollarOutlined, FileTextOutlined, ApiOutlined } from '@ant-design/icons';
import { Pie } from '@ant-design/charts';

const { Text } = Typography;

interface CostMonitorProps {
  totalCost: number;
  totalTokens: number;
  modelUsage: Record<string, number>;
}

const CostMonitor: React.FC<CostMonitorProps> = ({ totalCost, totalTokens, modelUsage }) => {
  // 准备饼图数据
  const pieData = Object.entries(modelUsage).map(([model, count]) => ({
    type: model,
    value: count
  }));

  const pieConfig = {
    data: pieData,
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'inner',
      offset: '-30%',
      content: '{value}',
      style: {
        fontSize: 14,
        textAlign: 'center'
      }
    },
    interactions: [{ type: 'element-active' }],
    height: 160
  };

  return (
    <Card title="成本与使用监控" className="mb-4">
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Statistic
            title="总成本"
            value={totalCost}
            precision={4}
            prefix={<DollarOutlined />}
            valueStyle={{ color: totalCost > 1 ? '#cf1322' : '#3f8600' }}
          />
        </Col>
        <Col span={12}>
          <Statistic
            title="总Token使用"
            value={totalTokens}
            prefix={<FileTextOutlined />}
            suffix="tokens"
          />
        </Col>
      </Row>

      {pieData.length > 0 && (
        <div className="mt-4">
          <Text strong>模型使用分布</Text>
          <Pie {...pieConfig} />
        </div>
      )}

      <div className="mt-4">
        <Text strong>成本预警</Text>
        <Progress
          percent={Math.min((totalCost / 10) * 100, 100)}
          status={totalCost > 8 ? 'exception' : totalCost > 5 ? 'normal' : 'success'}
          format={() => `$${totalCost.toFixed(2)} / $10.00`}
        />
        <Text type="secondary" className="text-xs">
          每日预算限制
        </Text>
      </div>

      <div className="mt-4 p-3 bg-gray-50 rounded">
        <Text type="secondary" className="text-xs">
          <ApiOutlined /> 成本计算说明：
        </Text>
        <ul className="mt-2 text-xs text-gray-600 space-y-1">
          <li>• GPT-4o: $5/1K输入, $15/1K输出</li>
          <li>• GPT-4o-mini: $0.15/1K输入, $0.6/1K输出</li>
          <li>• GPT-5: $12.5/1K输入, $25/1K输出</li>
          <li>• GPT-5-nano: $0.05/1K输入, $0.4/1K输出</li>
        </ul>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <Tooltip title="最经济的选择">
          <Tag color="green">
            <DollarOutlined /> 推荐: GPT-5-nano (简单任务)
          </Tag>
        </Tooltip>
        <Tooltip title="最佳性价比">
          <Tag color="blue">
            <ApiOutlined /> 推荐: GPT-4o-mini (平衡选择)
          </Tag>
        </Tooltip>
      </div>
    </Card>
  );
};

export default CostMonitor;
