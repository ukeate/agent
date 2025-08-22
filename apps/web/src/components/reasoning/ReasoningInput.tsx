import React, { useState } from 'react';
import {
  Card,
  Form,
  Input,
  Select,
  Switch,
  Button,
  Row,
  Col,
  InputNumber,
  Space,
  Typography,
  Collapse,
  Tag,
  Alert,
  Divider
} from 'antd';
import {
  PlayCircleOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  ExperimentOutlined
} from '@ant-design/icons';
import { useReasoningStore } from '../../stores/reasoningStore';

const { TextArea } = Input;
const { Option } = Select;
const { Text, Title } = Typography;
const { Panel } = Collapse;

interface ReasoningInputProps {
  onReasoningStart?: () => void;
  onReasoningComplete?: (chainId: string) => void;
}

export const ReasoningInput: React.FC<ReasoningInputProps> = ({
  onReasoningStart,
  onReasoningComplete
}) => {
  const [form] = Form.useForm();
  const {
    executeReasoning,
    streamReasoning,
    isExecuting,
    clearStreamingSteps
  } = useReasoningStore();

  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = async (values: any) => {
    onReasoningStart?.();
    clearStreamingSteps();

    try {
      const request = {
        problem: values.problem,
        strategy: values.strategy,
        context: values.context,
        max_steps: values.max_steps || 10,
        stream: values.stream || false,
        enable_branching: values.enable_branching || false,
        examples: values.examples || []
      };

      if (values.stream) {
        await streamReasoning(request);
      } else {
        const chain = await executeReasoning(request);
        onReasoningComplete?.(chain.id);
      }
    } catch (error) {
      console.error('推理执行失败:', error);
    }
  };

  const strategyInfo = {
    ZERO_SHOT: {
      name: 'Zero-shot CoT',
      description: '直接使用"让我们一步一步思考"提示',
      pros: ['简单直接', '无需示例', '通用性强'],
      cons: ['可能缺乏指导', '复杂问题效果有限']
    },
    FEW_SHOT: {
      name: 'Few-shot CoT', 
      description: '提供示例来指导推理过程',
      pros: ['有具体指导', '质量较高', '适合特定领域'],
      cons: ['需要准备示例', '可能过拟合']
    },
    AUTO_COT: {
      name: 'Auto-CoT',
      description: '自动选择最佳推理策略',
      pros: ['智能选择', '适应性强', '效果最佳'],
      cons: ['复杂度高', '执行时间长']
    }
  };

  const exampleProblems = [
    {
      category: '数学推理',
      problems: [
        '一个水池有两个进水管和一个出水管。第一个进水管每小时进水20升，第二个进水管每小时进水30升，出水管每小时出水15升。如果同时打开所有管子，多长时间能把容量为180升的空水池装满？',
        '一个班级有30名学生，其中60%是女生，40%是男生。如果新来了5名女生，现在女生占全班人数的百分比是多少？'
      ]
    },
    {
      category: '逻辑推理',
      problems: [
        '有三个盒子：红盒子、蓝盒子、绿盒子。每个盒子里都有一个球，分别是红球、蓝球、绿球，但是球的颜色和盒子的颜色不一定相同。已知：红盒子里不是红球，蓝盒子里不是蓝球。请问每个盒子里是什么颜色的球？',
        '在一个小镇上，理发师只给不给自己理发的人理发。请问：理发师给自己理发吗？'
      ]
    },
    {
      category: '商业分析',
      problems: [
        '一家科技公司最近用户流失率上升了15%，同时新功能使用率只有30%，客服投诉增加了25%，但是付费用户转化率提升了5%。分析可能的原因并提出改进建议。',
        '某电商平台发现移动端转化率比PC端低40%，但移动端流量占总流量的70%。应该如何优化移动端用户体验？'
      ]
    }
  ];

  return (
    <div className="reasoning-input">
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          strategy: 'ZERO_SHOT',
          max_steps: 10,
          stream: false,
          enable_branching: true
        }}
      >
        {/* 主要输入区域 */}
        <Card title="推理配置" className="mb-4">
          <Row gutter={[24, 24]}>
            <Col span={12}>
              <Form.Item
                label="推理问题"
                name="problem"
                rules={[{ required: true, message: '请输入要推理的问题' }]}
              >
                <TextArea
                  rows={6}
                  placeholder="请输入您要进行推理的问题..."
                  showCount
                  maxLength={1000}
                />
              </Form.Item>
            </Col>
            
            <Col span={12}>
              <Form.Item
                label="背景信息（可选）"
                name="context"
              >
                <TextArea
                  rows={6}
                  placeholder="提供相关的背景信息或约束条件..."
                  showCount
                  maxLength={500}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={[24, 16]}>
            <Col span={8}>
              <Form.Item
                label="推理策略"
                name="strategy"
              >
                <Select>
                  {Object.entries(strategyInfo).map(([key, info]) => (
                    <Option key={key} value={key}>
                      <div>
                        <Text strong>{info.name}</Text>
                        <br />
                        <Text type="secondary" className="text-xs">
                          {info.description}
                        </Text>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>

            <Col span={8}>
              <Form.Item
                label="最大推理步骤"
                name="max_steps"
              >
                <InputNumber
                  min={1}
                  max={20}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>

            <Col span={8}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Form.Item
                  label="流式输出"
                  name="stream"
                  valuePropName="checked"
                  className="mb-2"
                >
                  <Switch />
                </Form.Item>
                
                <Form.Item
                  label="启用分支"
                  name="enable_branching"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>
              </Space>
            </Col>
          </Row>

          <Divider />

          <div className="text-center">
            <Button
              type="primary"
              htmlType="submit"
              icon={<PlayCircleOutlined />}
              size="large"
              loading={isExecuting}
              disabled={isExecuting}
            >
              {isExecuting ? '推理进行中...' : '开始推理'}
            </Button>
          </div>
        </Card>

        {/* 策略说明 */}
        <Card title="策略说明" size="small" className="mb-4">
          <Row gutter={16}>
            {Object.entries(strategyInfo).map(([key, info]) => (
              <Col span={8} key={key}>
                <Card size="small" className="h-full">
                  <div className="text-center mb-3">
                    <Tag color="blue" className="mb-2">{info.name}</Tag>
                    <div className="text-sm text-gray-600">{info.description}</div>
                  </div>
                  
                  <div className="space-y-2">
                    <div>
                      <Text type="success" className="text-xs font-medium">优势:</Text>
                      <ul className="text-xs text-gray-600 mt-1 ml-4">
                        {info.pros.map((pro, i) => (
                          <li key={i}>{pro}</li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <Text variant="warning" className="text-xs font-medium">劣势:</Text>
                      <ul className="text-xs text-gray-600 mt-1 ml-4">
                        {info.cons.map((con, i) => (
                          <li key={i}>{con}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>

        {/* 示例问题 */}
        <Card 
          title="示例问题" 
          size="small"
          extra={
            <Button
              size="small"
              icon={<InfoCircleOutlined />}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? '隐藏' : '显示'}高级设置
            </Button>
          }
        >
          <Collapse ghost>
            {exampleProblems.map((category, index) => (
              <Panel header={category.category} key={index}>
                <div className="space-y-2">
                  {category.problems.map((problem, i) => (
                    <div
                      key={i}
                      className="p-3 bg-gray-50 rounded cursor-pointer hover:bg-gray-100 transition-colors"
                      onClick={() => form.setFieldsValue({ problem })}
                    >
                      <Text className="text-sm">{problem}</Text>
                    </div>
                  ))}
                </div>
              </Panel>
            ))}
          </Collapse>
        </Card>

        {/* 高级设置 */}
        {showAdvanced && (
          <Card title="高级设置" size="small" className="mt-4">
            <Alert
              message="技术参数说明"
              description={
                <div className="text-sm space-y-1">
                  <div><strong>流式输出:</strong> 实时显示推理过程，适合观察推理步骤</div>
                  <div><strong>启用分支:</strong> 当置信度较低时自动创建替代推理路径</div>
                  <div><strong>最大步骤:</strong> 限制推理步骤数量，防止无限循环</div>
                </div>
              }
              variant="default"
              showIcon
              className="mb-4"
            />

            <Row gutter={[24, 16]}>
              <Col span={12}>
                <Form.Item
                  label="置信度阈值"
                  name="confidence_threshold"
                  help="低于此阈值时触发分支创建"
                >
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.1}
                    style={{ width: '100%' }}
                    placeholder="0.6"
                  />
                </Form.Item>
              </Col>

              <Col span={12}>
                <Form.Item
                  label="超时时间（秒）"
                  name="timeout"
                  help="单个推理步骤的超时时间"
                >
                  <InputNumber
                    min={10}
                    max={300}
                    style={{ width: '100%' }}
                    placeholder="60"
                  />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              label="自定义提示词前缀"
              name="prompt_prefix"
              help="在问题前添加的自定义指令"
            >
              <TextArea
                rows={3}
                placeholder="例如：作为一个专业的分析师..."
              />
            </Form.Item>
          </Card>
        )}
      </Form>
    </div>
  );
};