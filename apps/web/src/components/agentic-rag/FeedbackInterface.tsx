/**
 * 反馈接口组件
 * 
 * 功能包括：
 * - 收集用户对检索结果质量的评价和反馈
 * - 支持多维度评分和详细意见提交
 * - 实现反馈数据的本地缓存和批量提交
 * - 提供反馈历史查看和管理功能
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Card,
  Space,
  Typography,
  Row,
  Col,
  Rate,
  Button,
  Input,
  Select,
  Radio,
  Checkbox,
  Form,
  Modal,
  Alert,
  Divider,
  Tag,
  List,
  Badge,
  Statistic,
  message,
  Empty,
  Tabs,
} from 'antd';
import {
  LikeOutlined,
  DislikeOutlined,
  MessageOutlined,
  SendOutlined,
  HistoryOutlined,
  StarOutlined,
  LikeOutlined as ThumbsUpOutlined,
  DislikeOutlined as ThumbsDownOutlined,
  QuestionCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  EditOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';
import { KnowledgeItem, AgenticQueryResponse } from '../../services/ragService';

const { TextArea } = Input;
const { Text, Title } = Typography;
const { Group: RadioGroup } = Radio;
const { Group: CheckboxGroup } = Checkbox;
const { TabPane } = Tabs;

// ==================== 组件props类型 ====================

interface FeedbackInterfaceProps {
  agenticResults?: AgenticQueryResponse | null;
  query?: string;
  className?: string;
  onFeedbackSubmit?: (feedback: FeedbackData) => void;
  showHistory?: boolean;
  compact?: boolean;
}

// ==================== 辅助类型 ====================

interface FeedbackData {
  id: string;
  query_id: string;
  query_text: string;
  timestamp: Date;
  overall_satisfaction: number;
  dimensions: {
    relevance: number;
    accuracy: number;
    completeness: number;
    usefulness: number;
    clarity: number;
  };
  result_feedback: Array<{
    result_id: string;
    rating: number;
    helpful: boolean;
    issues: string[];
    comments?: string;
  }>;
  system_feedback: {
    response_time: 'too_slow' | 'acceptable' | 'fast';
    interface_usability: number;
    feature_requests: string[];
    bugs_encountered: string[];
  };
  general_comments: string;
  improvement_suggestions: string[];
  would_recommend: boolean;
  contact_info?: string;
}

interface FeedbackStats {
  total_submissions: number;
  average_satisfaction: number;
  most_common_issues: string[];
  improvement_trends: Record<string, number>;
}

// ==================== 预定义选项 ====================

const DIMENSION_LABELS = {
  relevance: '相关性',
  accuracy: '准确性', 
  completeness: '完整性',
  usefulness: '实用性',
  clarity: '清晰度',
};

const COMMON_ISSUES = [
  '内容过时',
  '信息不准确',
  '相关性低',
  '内容重复',
  '来源不可信',
  '格式混乱',
  '缺少关键信息',
  '语言表达不清',
];

const IMPROVEMENT_SUGGESTIONS = [
  '增加更多数据源',
  '提升检索准确性',
  '优化结果排序',
  '改善用户界面',
  '加快响应速度',
  '增强结果解释',
  '支持多语言',
  '添加历史记录',
];

const RESPONSE_TIME_OPTIONS = [
  { label: '太慢', value: 'too_slow' },
  { label: '可接受', value: 'acceptable' },
  { label: '很快', value: 'fast' },
];

// ==================== 主组件 ====================

const FeedbackInterface: React.FC<FeedbackInterfaceProps> = ({
  agenticResults: propAgenticResults,
  query: propQuery,
  className = '',
  onFeedbackSubmit,
  showHistory = true,
  compact = false,
}) => {
  // ==================== 状态管理 ====================
  
  const {
    agenticResults,
    currentQuery,
    // feedbackData,
    setFeedbackData,
    clearFeedback,
  } = useRagStore();

  // 使用props数据或store数据
  const results = propAgenticResults || agenticResults;
  const query = propQuery || currentQuery;

  // ==================== 本地状态 ====================
  
  const [form] = Form.useForm();
  const [activeTab, setActiveTab] = useState<string>('rating');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedbackHistory, setFeedbackHistory] = useState<FeedbackData[]>([]);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [currentFeedback, setCurrentFeedback] = useState<Partial<FeedbackData>>({});
  const [resultRatings, setResultRatings] = useState<Record<string, number>>({});
  const [resultHelpful, setResultHelpful] = useState<Record<string, boolean>>({});

  // ==================== 生命周期 ====================
  
  useEffect(() => {
    // 从localStorage加载反馈历史
    const savedHistory = localStorage.getItem('rag_feedback_history');
    if (savedHistory) {
      try {
        const history = JSON.parse(savedHistory);
        setFeedbackHistory(history);
      } catch (error) {
        console.error('Failed to load feedback history:', error);
      }
    }
  }, []);

  useEffect(() => {
    // 初始化当前反馈数据
    if (results && query) {
      const initialFeedback: Partial<FeedbackData> = {
        query_id: `query_${Date.now()}`,
        query_text: query,
        timestamp: new Date(),
        dimensions: {
          relevance: 0,
          accuracy: 0,
          completeness: 0,
          usefulness: 0,
          clarity: 0,
        },
        system_feedback: {
          response_time: 'acceptable',
          interface_usability: 0,
          feature_requests: [],
          bugs_encountered: [],
        },
        result_feedback: results.results?.map(r => ({
          result_id: r.id,
          rating: 0,
          helpful: false,
          issues: [],
        })) || [],
        general_comments: '',
        improvement_suggestions: [],
        would_recommend: true,
      };
      
      setCurrentFeedback(initialFeedback);
      form.setFieldsValue(initialFeedback);
    }
  }, [results, query, form]);

  // ==================== 事件处理 ====================
  
  const handleDimensionRating = useCallback((dimension: keyof FeedbackData['dimensions'], rating: number) => {
    const updatedFeedback = {
      ...currentFeedback,
      dimensions: {
        ...currentFeedback.dimensions,
        [dimension]: rating,
      },
    };
    setCurrentFeedback(updatedFeedback);
    
    // 计算总体满意度
    const dimensionScores = Object.values(updatedFeedback.dimensions || {});
    const avgScore = dimensionScores.reduce((sum, score) => sum + score, 0) / dimensionScores.length;
    updatedFeedback.overall_satisfaction = avgScore;
    
    form.setFieldsValue(updatedFeedback);
  }, [currentFeedback, form]);

  const handleResultRating = useCallback((resultId: string, rating: number) => {
    setResultRatings(prev => ({ ...prev, [resultId]: rating }));
    
    const updatedResultFeedback = currentFeedback.result_feedback?.map(rf => 
      rf.result_id === resultId ? { ...rf, rating } : rf
    );
    
    setCurrentFeedback(prev => ({
      ...prev,
      result_feedback: updatedResultFeedback,
    }));
  }, [currentFeedback.result_feedback]);

  const handleResultHelpful = useCallback((resultId: string, helpful: boolean) => {
    setResultHelpful(prev => ({ ...prev, [resultId]: helpful }));
    
    const updatedResultFeedback = currentFeedback.result_feedback?.map(rf => 
      rf.result_id === resultId ? { ...rf, helpful } : rf
    );
    
    setCurrentFeedback(prev => ({
      ...prev,
      result_feedback: updatedResultFeedback,
    }));
  }, [currentFeedback.result_feedback]);

  const handleSubmitFeedback = useCallback(async () => {
    try {
      const formValues = await form.validateFields();
      setIsSubmitting(true);

      const feedbackData: FeedbackData = {
        ...currentFeedback,
        ...formValues,
        id: `feedback_${Date.now()}`,
        timestamp: new Date(),
      } as FeedbackData;

      // 保存到历史记录
      const newHistory = [feedbackData, ...feedbackHistory].slice(0, 100);
      setFeedbackHistory(newHistory);
      localStorage.setItem('rag_feedback_history', JSON.stringify(newHistory));

      // 更新store
      setFeedbackData(feedbackData);

      // 回调通知
      onFeedbackSubmit?.({
        query_id: feedbackData.query_id || '',
        ratings: feedbackData.dimensions || {},
        comments: feedbackData.feedback_text || ''
      });

      message.success('反馈提交成功！感谢您的宝贵意见');
      
      // 重置表单
      form.resetFields();
      setCurrentFeedback({});
      setResultRatings({});
      setResultHelpful({});

    } catch (error: any) {
      message.error('反馈提交失败: ' + (error.message || '未知错误'));
    } finally {
      setIsSubmitting(false);
    }
  }, [currentFeedback, form, feedbackHistory, setFeedbackData, onFeedbackSubmit]);

  const handleClearFeedback = useCallback(() => {
    form.resetFields();
    setCurrentFeedback({});
    setResultRatings({});
    setResultHelpful({});
    clearFeedback();
    message.info('反馈已清空');
  }, [form, clearFeedback]);

  // ==================== 计算反馈统计 ====================
  
  const feedbackStats = React.useMemo((): FeedbackStats => {
    if (feedbackHistory.length === 0) {
      return {
        total_submissions: 0,
        average_satisfaction: 0,
        most_common_issues: [],
        improvement_trends: {},
      };
    }

    const avgSatisfaction = feedbackHistory.reduce((sum, fb) => 
      sum + (fb.overall_satisfaction || 0), 0
    ) / feedbackHistory.length;

    const issueCount: Record<string, number> = {};
    const improvementCount: Record<string, number> = {};

    feedbackHistory.forEach(fb => {
      fb.result_feedback?.forEach(rf => {
        rf.issues?.forEach(issue => {
          issueCount[issue] = (issueCount[issue] || 0) + 1;
        });
      });
      
      fb.improvement_suggestions?.forEach(suggestion => {
        improvementCount[suggestion] = (improvementCount[suggestion] || 0) + 1;
      });
    });

    const mostCommonIssues = Object.entries(issueCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([issue]) => issue);

    return {
      total_submissions: feedbackHistory.length,
      average_satisfaction: avgSatisfaction,
      most_common_issues: mostCommonIssues,
      improvement_trends: improvementCount,
    };
  }, [feedbackHistory]);

  // ==================== 渲染辅助函数 ====================
  
  const renderDimensionRating = () => (
    <Space direction="vertical" style={{ width: '100%' }}>
      {Object.entries(DIMENSION_LABELS).map(([key, label]) => (
        <Row key={key} align="middle" justify="space-between">
          <Col span={8}>
            <Text>{label}:</Text>
          </Col>
          <Col span={12}>
            <Rate
              value={currentFeedback.dimensions?.[key as keyof FeedbackData['dimensions']] || 0}
              onChange={(value) => handleDimensionRating(key as keyof FeedbackData['dimensions'], value)}
            />
          </Col>
          <Col span={4} style={{ textAlign: 'right' }}>
            <Text type="secondary">
              {currentFeedback.dimensions?.[key as keyof FeedbackData['dimensions']] || 0}/5
            </Text>
          </Col>
        </Row>
      ))}
      
      <Divider style={{ margin: '16px 0' }} />
      
      <Row align="middle" justify="space-between">
        <Col>
          <Text strong>总体满意度:</Text>
        </Col>
        <Col>
          <Space>
            <Rate
              value={currentFeedback.overall_satisfaction || 0}
              onChange={(value) => setCurrentFeedback(prev => ({ 
                ...prev, 
                overall_satisfaction: value 
              }))}
              style={{ fontSize: 20 }}
            />
            <Text strong style={{ fontSize: 16 }}>
              {(currentFeedback.overall_satisfaction || 0).toFixed(1)}/5
            </Text>
          </Space>
        </Col>
      </Row>
    </Space>
  );

  const renderResultFeedback = () => {
    if (!results?.results || results.results.length === 0) {
      return <Empty description="暂无检索结果可评价" />;
    }

    return (
      <List
        dataSource={results.results}
        renderItem={(item: KnowledgeItem, index) => (
          <List.Item key={item.id}>
            <Card size="small" style={{ width: '100%' }}>
              <Row align="middle" justify="space-between">
                <Col span={14}>
                  <Space direction="vertical" size="small">
                    <Text strong ellipsis style={{ maxWidth: 300 }}>
                      结果 {index + 1}: {item.file_path?.split('/').pop() || `文档 ${item.id.substring(0, 8)}`}
                    </Text>
                    <Text type="secondary" ellipsis style={{ maxWidth: 350 }}>
                      {item.content.substring(0, 100)}...
                    </Text>
                  </Space>
                </Col>
                
                <Col span={10}>
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <Row align="middle" justify="space-between">
                      <Text>评分:</Text>
                      <Rate
                        size="small"
                        value={resultRatings[item.id] || 0}
                        onChange={(value) => handleResultRating(item.id, value)}
                      />
                    </Row>
                    
                    <Row align="middle" justify="space-between">
                      <Text>有帮助:</Text>
                      <Space>
                        <Button
                          size="small"
                          type={resultHelpful[item.id] === true ? 'primary' : 'default'}
                          icon={<ThumbsUpOutlined />}
                          onClick={() => handleResultHelpful(item.id, true)}
                        />
                        <Button
                          size="small"
                          type={resultHelpful[item.id] === false ? 'primary' : 'default'}
                          icon={<ThumbsDownOutlined />}
                          onClick={() => handleResultHelpful(item.id, false)}
                        />
                      </Space>
                    </Row>
                  </Space>
                </Col>
              </Row>

              <Divider style={{ margin: '12px 0' }} />

              <Form.Item 
                name={['result_feedback', index, 'issues']}
                label="问题标记"
              >
                <CheckboxGroup options={COMMON_ISSUES} />
              </Form.Item>

              <Form.Item 
                name={['result_feedback', index, 'comments']}
                label="具体意见"
              >
                <TextArea
                  rows={2}
                  placeholder="对这个结果的具体意见或建议..."
                  maxLength={200}
                  showCount
                />
              </Form.Item>
            </Card>
          </List.Item>
        )}
      />
    );
  };

  const renderSystemFeedback = () => (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      
      {/* 响应时间评价 */}
      <div>
        <Text strong>响应时间感受:</Text>
        <Form.Item name={['system_feedback', 'response_time']} style={{ marginTop: 8 }}>
          <RadioGroup options={RESPONSE_TIME_OPTIONS} />
        </Form.Item>
      </div>

      {/* 界面易用性评分 */}
      <Row align="middle" justify="space-between">
        <Col span={8}>
          <Text strong>界面易用性:</Text>
        </Col>
        <Col span={12}>
          <Form.Item name={['system_feedback', 'interface_usability']} style={{ margin: 0 }}>
            <Rate />
          </Form.Item>
        </Col>
      </Row>

      {/* 功能需求 */}
      <div>
        <Text strong>希望增加的功能:</Text>
        <Form.Item name={['system_feedback', 'feature_requests']} style={{ marginTop: 8 }}>
          <CheckboxGroup 
            options={IMPROVEMENT_SUGGESTIONS.map(s => ({ label: s, value: s }))}
          />
        </Form.Item>
      </div>

      {/* Bug报告 */}
      <div>
        <Text strong>遇到的问题:</Text>
        <Form.Item name={['system_feedback', 'bugs_encountered']} style={{ marginTop: 8 }}>
          <Select
            mode="tags"
            placeholder="描述遇到的问题或Bug..."
            style={{ width: '100%' }}
          />
        </Form.Item>
      </div>

    </Space>
  );

  const renderGeneralFeedback = () => (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      
      {/* 总体评价 */}
      <div>
        <Text strong>总体评价:</Text>
        <Form.Item name="general_comments" style={{ marginTop: 8 }}>
          <TextArea
            rows={4}
            placeholder="请分享您对本次智能检索体验的整体感受..."
            maxLength={500}
            showCount
          />
        </Form.Item>
      </div>

      {/* 改进建议 */}
      <div>
        <Text strong>改进建议:</Text>
        <Form.Item name="improvement_suggestions" style={{ marginTop: 8 }}>
          <CheckboxGroup 
            options={IMPROVEMENT_SUGGESTIONS.map(s => ({ label: s, value: s }))}
          />
        </Form.Item>
      </div>

      {/* 推荐意愿 */}
      <div>
        <Text strong>您是否愿意向他人推荐本系统:</Text>
        <Form.Item name="would_recommend" style={{ marginTop: 8 }}>
          <RadioGroup>
            <Radio value={true}>是的，我会推荐</Radio>
            <Radio value={false}>不会推荐</Radio>
          </RadioGroup>
        </Form.Item>
      </div>

      {/* 联系信息 */}
      <div>
        <Text strong>联系方式 (可选):</Text>
        <Form.Item 
          name="contact_info" 
          style={{ marginTop: 8 }}
        >
          <Input 
            placeholder="如果您希望我们就反馈内容与您联系，请留下邮箱或其他联系方式"
            maxLength={100}
          />
        </Form.Item>
      </div>

    </Space>
  );

  const renderHistoryModal = () => (
    <Modal
      title="反馈历史"
      open={showHistoryModal}
      onCancel={() => setShowHistoryModal(false)}
      footer={null}
      width={800}
    >
      {feedbackHistory.length === 0 ? (
        <Empty description="暂无反馈历史" />
      ) : (
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          
          {/* 统计摘要 */}
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="总反馈数"
                value={feedbackStats.total_submissions}
                prefix={<MessageOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="平均满意度"
                value={feedbackStats.average_satisfaction.toFixed(1)}
                suffix="/ 5"
                prefix={<StarOutlined />}
              />
            </Col>
            <Col span={12}>
              <div>
                <Text strong>常见问题:</Text>
                <div style={{ marginTop: 4 }}>
                  {feedbackStats.most_common_issues.map(issue => (
                    <Tag key={issue} color="orange" style={{ margin: '2px' }}>
                      {issue}
                    </Tag>
                  ))}
                </div>
              </div>
            </Col>
          </Row>

          <Divider />

          {/* 历史记录列表 */}
          <List
            dataSource={feedbackHistory.slice(0, 10)}
            renderItem={(feedback) => (
              <List.Item>
                <Card size="small" style={{ width: '100%' }}>
                  <Row justify="space-between" align="middle">
                    <Col span={16}>
                      <Space direction="vertical" size="small">
                        <Text strong>查询: "{feedback.query_text}"</Text>
                        <Text type="secondary">
                          {new Date(feedback.timestamp).toLocaleString()}
                        </Text>
                        <Space>
                          <Rate disabled value={feedback.overall_satisfaction} size="small" />
                          <Text type="secondary">
                            {feedback.overall_satisfaction?.toFixed(1)}/5
                          </Text>
                        </Space>
                      </Space>
                    </Col>
                    <Col span={8} style={{ textAlign: 'right' }}>
                      <Space>
                        {feedback.would_recommend ? (
                          <Tag color="green" icon={<CheckCircleOutlined />}>
                            推荐
                          </Tag>
                        ) : (
                          <Tag color="red" icon={<ExclamationCircleOutlined />}>
                            不推荐
                          </Tag>
                        )}
                      </Space>
                    </Col>
                  </Row>
                </Card>
              </List.Item>
            )}
          />
        </Space>
      )}
    </Modal>
  );

  // ==================== 渲染主组件 ====================

  return (
    <div className={`feedback-interface ${className}`}>
      
      {!results && (
        <Card>
          <Empty 
            description="请先进行智能检索以提供反馈"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </Card>
      )}

      {results && (
        <Card
          title={
            <Space>
              <MessageOutlined />
              <Title level={4} style={{ margin: 0 }}>智能检索反馈</Title>
              <Tag color="blue">改进系统</Tag>
            </Space>
          }
          extra={
            !compact && showHistory && (
              <Space>
                <Button
                  icon={<HistoryOutlined />}
                  onClick={() => setShowHistoryModal(true)}
                  size="small"
                >
                  历史反馈
                </Button>
                <Button
                  danger
                  size="small"
                  onClick={handleClearFeedback}
                  disabled={isSubmitting}
                >
                  清空
                </Button>
              </Space>
            )
          }
        >
          
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSubmitFeedback}
          >
            <Tabs 
              activeKey={activeTab} 
              onChange={setActiveTab}
              type={compact ? 'line' : 'card'}
            >
              
              {/* 质量评分标签页 */}
              <TabPane
                tab={
                  <Space>
                    <StarOutlined />
                    质量评分
                  </Space>
                }
                key="rating"
              >
                <Alert
                  message="请对本次检索结果的各个维度进行评分"
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
                {renderDimensionRating()}
              </TabPane>

              {/* 结果反馈标签页 */}
              <TabPane
                tab={
                  <Space>
                    <LikeOutlined />
                    结果反馈
                    {results.results && (
                      <Badge count={results.results.length} size="small" />
                    )}
                  </Space>
                }
                key="results"
              >
                <Alert
                  message="请对每个检索结果进行评价"
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
                {renderResultFeedback()}
              </TabPane>

              {/* 系统反馈标签页 */}
              <TabPane
                tab={
                  <Space>
                    <QuestionCircleOutlined />
                    系统反馈
                  </Space>
                }
                key="system"
              >
                <Alert
                  message="请评价系统的性能和易用性"
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
                {renderSystemFeedback()}
              </TabPane>

              {/* 总体反馈标签页 */}
              <TabPane
                tab={
                  <Space>
                    <EditOutlined />
                    总体反馈
                  </Space>
                }
                key="general"
              >
                <Alert
                  message="请提供您的总体评价和建议"
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
                {renderGeneralFeedback()}
              </TabPane>

            </Tabs>

            {/* 提交按钮 */}
            <Row justify="center" style={{ marginTop: 24 }}>
              <Space size="middle">
                <Button onClick={handleClearFeedback} disabled={isSubmitting}>
                  清空
                </Button>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={isSubmitting}
                  icon={<SendOutlined />}
                  size="large"
                >
                  {isSubmitting ? '提交中...' : '提交反馈'}
                </Button>
              </Space>
            </Row>

          </Form>

        </Card>
      )}

      {/* 历史记录模态框 */}
      {renderHistoryModal()}

    </div>
  );
};

export default FeedbackInterface;