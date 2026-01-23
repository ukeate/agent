/**
 * 向量可视化面板
 * 展示t-SNE、UMAP、PCA等降维可视化功能
 */

import React, { useState } from 'react'
import {
  Card,
  Radio,
  Slider,
  Button,
  Space,
  Alert,
  Select,
  Row,
  Col,
} from 'antd'
import {
  EyeOutlined,
  DotChartOutlined,
  BarChartOutlined,
} from '@ant-design/icons'

const VectorVisualizationPanel: React.FC = () => {
  const [algorithm, setAlgorithm] = useState('tsne')
  const [perplexity, setPerplexity] = useState(30)
  const [dimensions, setDimensions] = useState(2)

  return (
    <div>
      <Alert
        message="向量可视化"
        description="使用t-SNE、UMAP、PCA等算法将高维向量降维到2D/3D空间进行可视化，支持聚类着色和交互式探索。"
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        <Col span={8}>
          <Card title="可视化配置" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <label>降维算法</label>
                <Radio.Group
                  value={algorithm}
                  onChange={e => setAlgorithm(e.target.value)}
                >
                  <Radio value="tsne">t-SNE</Radio>
                  <Radio value="umap">UMAP</Radio>
                  <Radio value="pca">PCA</Radio>
                </Radio.Group>
              </div>

              {algorithm === 'tsne' && (
                <div>
                  <label>困惑度: {perplexity}</label>
                  <Slider
                    min={5}
                    max={100}
                    value={perplexity}
                    onChange={setPerplexity}
                    marks={{ 30: '30', 50: '50' }}
                  />
                </div>
              )}

              <div>
                <label>输出维度</label>
                <Select
                  value={dimensions}
                  onChange={setDimensions}
                  style={{ width: '100%' }}
                >
                  <Select.Option value={2}>2D</Select.Option>
                  <Select.Option value={3}>3D</Select.Option>
                </Select>
              </div>

              <Button type="primary" block>
                生成可视化
              </Button>
            </Space>
          </Card>

          <Card title="可视化类型" size="small" style={{ marginTop: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>✓ 散点图</div>
              <div>✓ 密度图</div>
              <div>✓ 等高线图</div>
              <div>✓ 热力图</div>
              <div>✓ 轨迹图</div>
            </Space>
          </Card>
        </Col>

        <Col span={16}>
          <Card title="向量分布可视化" size="small">
            <div
              style={{
                height: 400,
                backgroundColor: '#fafafa',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '2px dashed #d9d9d9',
              }}
            >
              <Space direction="vertical" align="center">
                <DotChartOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />
                <span>向量分布散点图</span>
                <span style={{ fontSize: 12, color: '#999' }}>
                  支持交互式缩放和聚类着色
                </span>
              </Space>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default VectorVisualizationPanel
