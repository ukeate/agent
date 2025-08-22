/**
 * 时序向量分析面板
 * 展示向量轨迹跟踪和模式检测功能
 */

import React, { useState } from 'react';
import { Card, Table, Button, DatePicker, Select, Space, Alert, Statistic, Row, Col } from 'antd';
import { LineChartOutlined, RadarChartOutlined } from '@ant-design/icons';

const TemporalVectorPanel: React.FC = () => {
  const [timeRange, setTimeRange] = useState(null);
  const [entityId, setEntityId] = useState(null);

  const mockTrajectories = [
    { id: 'entity_1', patterns: 3, distance: 12.5, trend: 'increasing' },
    { id: 'entity_2', patterns: 1, distance: 8.3, trend: 'stable' },
    { id: 'entity_3', patterns: 5, distance: 15.7, trend: 'volatile' }
  ];

  const columns = [
    { title: '实体ID', dataIndex: 'id', key: 'id' },
    { title: '检测模式', dataIndex: 'patterns', key: 'patterns' },
    { title: '轨迹距离', dataIndex: 'distance', key: 'distance' },
    { title: '趋势', dataIndex: 'trend', key: 'trend' }
  ];

  return (
    <div>
      <Alert
        message="时序向量分析"
        description="跟踪向量随时间的变化轨迹，检测收敛、发散、周期性等模式，分析向量变化趋势。"
        variant="default"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        <Col span={8}>
          <Card title="分析配置" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <DatePicker.RangePicker placeholder={['开始时间', '结束时间']} />
              <Select placeholder="选择实体ID" style={{ width: '100%' }}>
                <Select.Option value="entity_1">Entity 1</Select.Option>
                <Select.Option value="entity_2">Entity 2</Select.Option>
              </Select>
              <Button type="primary" block>分析轨迹</Button>
            </Space>
          </Card>

          <Card title="时序统计" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={12}>
                <Statistic title="活跃实体" value={127} />
              </Col>
              <Col span={12}>
                <Statistic title="检测模式" value={45} />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Card title="轨迹分析结果" size="small">
            <Table
              columns={columns}
              dataSource={mockTrajectories}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </Card>

          <Card title="向量轨迹可视化" size="small" style={{ marginTop: 16 }}>
            <div 
              style={{ 
                height: 200, 
                backgroundColor: '#fafafa', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                border: '2px dashed #d9d9d9'
              }}
            >
              <Space direction="vertical" align="center">
                <LineChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                <span>轨迹时序图表</span>
              </Space>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TemporalVectorPanel;