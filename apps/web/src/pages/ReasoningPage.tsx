import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Tabs,
  Button,
  Space,
  Typography,
  Divider,
  Alert,
  Tag
} from 'antd';
import {
  BulbOutlined,
  ExperimentOutlined,
  ControlOutlined,
  HistoryOutlined,
  BarChartOutlined
} from '@ant-design/icons';

import { ReasoningInput } from '../components/reasoning/ReasoningInput';
import { ReasoningChainVisualization } from '../components/reasoning/ReasoningChainVisualization';
import { ReasoningStrategiesComparison } from '../components/reasoning/ReasoningStrategiesComparison';
import { ReasoningQualityControl } from '../components/reasoning/ReasoningQualityControl';
import { ReasoningHistory } from '../components/reasoning/ReasoningHistory';
import { ReasoningStats } from '../components/reasoning/ReasoningStats';
import { useReasoningStore } from '../stores/reasoningStore';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const ReasoningPage: React.FC = () => {
  const {
    currentChain,
    isExecuting,
    streamingSteps,
    validationResults,
    recoveryStats,
    executeReasoning,
    streamReasoning,
    validateChain,
    getReasoningHistory
  } = useReasoningStore();

  const [activeTab, setActiveTab] = useState('input');
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(true);

  useEffect(() => {
    // 加载推理历史
    getReasoningHistory();
  }, []);

  const handleReasoningComplete = (chainId: string) => {
    setActiveTab('visualization');
  };

  return (
    <div className="reasoning-page p-6 bg-gray-50 min-h-screen">
      {/* 页面标题和技术说明 */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <Title level={2} className="mb-2">
              <BulbOutlined className="mr-2" />
              链式思考(CoT)推理系统
            </Title>
            <Text type="secondary">
              技术演示：Zero-shot、Few-shot、Auto-CoT推理策略
            </Text>
          </div>
          <Space>
            <Button
              type={showTechnicalDetails ? 'primary' : 'default'}
              icon={<ControlOutlined />}
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            >
              技术细节
            </Button>
          </Space>
        </div>

        {showTechnicalDetails && (
          <Alert
            message="技术架构说明"
            description={
              <div>
                <p><strong>推理引擎：</strong>BaseCoTEngine + 三种策略实现</p>
                <p><strong>质量控制：</strong>一致性验证、置信度检查、自我验证</p>
                <p><strong>恢复机制：</strong>回溯、分支、重启、细化、替代路径</p>
                <p><strong>状态管理：</strong>LangGraph集成，支持检查点和分支</p>
              </div>
            }
            variant="default"
            showIcon
            className="mb-4"
          />
        )}
      </div>

      {/* 主要内容区域 */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card>
            <Tabs
              activeKey={activeTab}
              onChange={setActiveTab}
              type="card"
              className="reasoning-tabs"
            >
              <TabPane
                tab={
                  <span>
                    <ExperimentOutlined />
                    推理输入
                  </span>
                }
                key="input"
              >
                <ReasoningInput
                  onReasoningStart={() => setActiveTab('visualization')}
                  onReasoningComplete={handleReasoningComplete}
                />
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <BulbOutlined />
                    推理可视化
                    {currentChain && <Tag color="blue" className="ml-2">运行中</Tag>}
                  </span>
                }
                key="visualization"
              >
                <ReasoningChainVisualization
                  chain={currentChain}
                  streamingSteps={streamingSteps}
                  isExecuting={isExecuting}
                />
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <ExperimentOutlined />
                    策略对比
                  </span>
                }
                key="strategies"
              >
                <ReasoningStrategiesComparison />
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <ControlOutlined />
                    质量控制
                    {validationResults && (
                      <Tag 
                        color={validationResults.is_valid ? 'green' : 'red'} 
                        className="ml-2"
                      >
                        {validationResults.is_valid ? '通过' : '失败'}
                      </Tag>
                    )}
                  </span>
                }
                key="quality"
              >
                <ReasoningQualityControl
                  currentChain={currentChain}
                  validationResults={validationResults}
                  recoveryStats={recoveryStats}
                />
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <HistoryOutlined />
                    推理历史
                  </span>
                }
                key="history"
              >
                <ReasoningHistory />
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <BarChartOutlined />
                    统计分析
                  </span>
                }
                key="stats"
              >
                <ReasoningStats />
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>

      {/* 技术状态面板 */}
      {showTechnicalDetails && (
        <Row gutter={[16, 16]} className="mt-4">
          <Col span={24}>
            <Card title="系统状态" size="small">
              <Row gutter={[16, 16]}>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {isExecuting ? '执行中' : '就绪'}
                    </div>
                    <div className="text-gray-500">推理状态</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {streamingSteps?.length || 0}
                    </div>
                    <div className="text-gray-500">当前步骤数</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {currentChain?.confidence_score ? 
                        `${(currentChain.confidence_score * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </div>
                    <div className="text-gray-500">平均置信度</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {recoveryStats?.total_failures || 0}
                    </div>
                    <div className="text-gray-500">恢复次数</div>
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default ReasoningPage;