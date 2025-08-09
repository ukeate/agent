/**
 * Agentic RAG 主页面
 * 
 * 功能包括：
 * - 整合所有Agentic RAG高级组件
 * - 提供完整的智能RAG交互体验
 * - 支持多代理协作检索和分析
 * - 实现会话管理和状态持久化
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Layout,
  Row,
  Col,
  Card,
  Space,
  Typography,
  Button,
  Divider,
  Affix,
  BackTop,
  Drawer,
  Modal,
  message,
  Tour,
  FloatButton,
  Badge,
} from 'antd';
import {
  RobotOutlined,
  SettingOutlined,
  QuestionCircleOutlined,
  HistoryOutlined,
  BulbOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  MenuOutlined,
  FullscreenOutlined,
  FullscreenExitOutlined,
} from '@ant-design/icons';
import { useRagStore, AgenticSession } from '../stores/ragStore';
import AgenticQueryPanel from '../components/agentic-rag/AgenticQueryPanel';
import RetrievalProcessViewer from '../components/agentic-rag/RetrievalProcessViewer';
import IntelligentResultsPanel from '../components/agentic-rag/IntelligentResultsPanel';
import ExplanationViewer from '../components/agentic-rag/ExplanationViewer';
import FeedbackInterface from '../components/agentic-rag/FeedbackInterface';
import SessionManager from '../components/agentic-rag/SessionManager';
import FallbackHandler from '../components/agentic-rag/FallbackHandler';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;

// ==================== 接口定义 ====================

interface AgenticRagPageProps {
  className?: string;
}

interface LayoutSettings {
  siderCollapsed: boolean;
  showProcessViewer: boolean;
  showExplanation: boolean;
  showFeedback: boolean;
  showSessionManager: boolean;
  compactMode: boolean;
  fullscreen: boolean;
}

// ==================== 主页面组件 ====================

const AgenticRagPage: React.FC<AgenticRagPageProps> = ({ 
  className = '' 
}) => {
  // ==================== 状态管理 ====================
  
  const {
    isQuerying,
    agenticResults,
    // currentQuery,
    currentSession,
    // queryProcessSteps: [] as any[],
    showExplanation,
    setShowExplanation,
  } = useRagStore();

  // ==================== 本地状态 ====================
  
  const [layoutSettings, setLayoutSettings] = useState<LayoutSettings>({
    siderCollapsed: false,
    showProcessViewer: true,
    showExplanation: true,
    showFeedback: true,
    showSessionManager: false,
    compactMode: false,
    fullscreen: false,
  });

  const [showTour, setShowTour] = useState(false);
  const [showMobileDrawer, setShowMobileDrawer] = useState(false);
  const [activeComponent, setActiveComponent] = useState<string>('query');
  const [componentRefs, setComponentRefs] = useState<Record<string, React.RefObject<HTMLDivElement>>>({});

  // ==================== 生命周期 ====================
  
  useEffect(() => {
    // 初始化组件引用
    const refs = {
      query: React.createRef<HTMLDivElement>(),
      process: React.createRef<HTMLDivElement>(),
      results: React.createRef<HTMLDivElement>(),
      explanation: React.createRef<HTMLDivElement>(),
      feedback: React.createRef<HTMLDivElement>(),
    };
    setComponentRefs(refs);

    // 检查是否需要显示引导
    const hasSeenTour = localStorage.getItem('agentic_rag_tour_seen');
    if (!hasSeenTour) {
      setShowTour(true);
    }
  }, []);

  // ==================== 布局控制 ====================
  
  const updateLayoutSettings = useCallback((updates: Partial<LayoutSettings>) => {
    setLayoutSettings(prev => ({ ...prev, ...updates }));
  }, []);

  const toggleFullscreen = useCallback(() => {
    const newFullscreen = !layoutSettings.fullscreen;
    updateLayoutSettings({ fullscreen: newFullscreen });
    
    if (newFullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }, [layoutSettings.fullscreen, updateLayoutSettings]);

  const handleSessionSelect = useCallback((session: AgenticSession) => {
    message.info(`已切换到会话: ${session.name}`);
    updateLayoutSettings({ showSessionManager: false });
  }, [updateLayoutSettings]);

  // ==================== 引导配置 ====================
  
  const tourSteps = [
    {
      title: '欢迎使用Agentic RAG系统',
      description: '这是一个智能多代理检索增强生成系统，让我们开始探索其强大功能',
      target: () => componentRefs.query?.current || null,
    },
    {
      title: '智能查询面板',
      description: '在这里输入您的查询，系统将自动分析意图并选择最佳策略',
      target: () => componentRefs.query?.current || null,
    },
    {
      title: '检索过程可视化',
      description: '实时观察多代理协作的检索过程，了解系统如何工作',
      target: () => componentRefs.process?.current || null,
    },
    {
      title: '智能结果分析',
      description: '查看经过质量评估和聚类分析的检索结果',
      target: () => componentRefs.results?.current || null,
    },
    {
      title: '解释与透明度',
      description: '深入了解系统的推理过程和决策逻辑',
      target: () => componentRefs.explanation?.current || null,
    },
    {
      title: '反馈与改进',
      description: '您的反馈将帮助系统不断学习和改进',
      target: () => componentRefs.feedback?.current || null,
    },
  ];

  // ==================== 渲染函数 ====================
  
  const renderHeader = () => (
    <Header 
      style={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: '0 24px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      }}
    >
      <Row align="middle" justify="space-between">
        <Col>
          <Space>
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => setShowMobileDrawer(true)}
              style={{ color: 'white' }}
              className="mobile-menu-btn"
            />
            <Space>
              <RobotOutlined style={{ fontSize: 24, color: 'white' }} />
              <Title level={3} style={{ margin: 0, color: 'white' }}>
                Agentic RAG 智能系统
              </Title>
            </Space>
          </Space>
        </Col>
        
        <Col>
          <Space>
            {currentSession && (
              <Badge dot={isQuerying}>
                <Text style={{ color: 'white' }}>
                  会话: {currentSession.name}
                </Text>
              </Badge>
            )}
            
            <Button
              type="text"
              icon={<HistoryOutlined />}
              onClick={() => updateLayoutSettings({ showSessionManager: true })}
              style={{ color: 'white' }}
            >
              会话管理
            </Button>
            
            <Button
              type="text"
              icon={<QuestionCircleOutlined />}
              onClick={() => setShowTour(true)}
              style={{ color: 'white' }}
            >
              使用指南
            </Button>
            
            <Button
              type="text"
              icon={layoutSettings.fullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
              onClick={toggleFullscreen}
              style={{ color: 'white' }}
            >
              {layoutSettings.fullscreen ? '退出全屏' : '全屏模式'}
            </Button>
          </Space>
        </Col>
      </Row>
    </Header>
  );

  const renderSider = () => (
    <Sider
      width={320}
      collapsed={layoutSettings.siderCollapsed}
      collapsible
      onCollapse={(collapsed) => updateLayoutSettings({ siderCollapsed: collapsed })}
      style={{ 
        background: 'white',
        boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
      }}
    >
      <div style={{ padding: layoutSettings.siderCollapsed ? 8 : 16 }}>
        {!layoutSettings.siderCollapsed ? (
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            
            {/* 会话管理 */}
            <Card size="small" title="当前会话">
              <SessionManager
                compact
                maxSessions={10}
                onSessionSelect={handleSessionSelect}
              />
            </Card>

            {/* 系统状态 */}
            <Card size="small" title="系统状态">
              <Space direction="vertical" style={{ width: '100%' }} size="small">
                <Row justify="space-between">
                  <Text>查询状态:</Text>
                  <Badge status={isQuerying ? 'processing' : 'default'} 
                         text={isQuerying ? '执行中' : '就绪'} />
                </Row>
                
                <Row justify="space-between">
                  <Text>结果数量:</Text>
                  <Text strong>{agenticResults?.results?.length || 0}</Text>
                </Row>
                
                {agenticResults?.confidence && (
                  <Row justify="space-between">
                    <Text>置信度:</Text>
                    <Text strong>{Math.round(agenticResults.confidence * 100)}%</Text>
                  </Row>
                )}

                {agenticResults?.processing_time && (
                  <Row justify="space-between">
                    <Text>处理时间:</Text>
                    <Text strong>{agenticResults.processing_time}ms</Text>
                  </Row>
                )}
              </Space>
            </Card>

            {/* 快速操作 */}
            <Card size="small" title="快速操作">
              <Space direction="vertical" style={{ width: '100%' }} size="small">
                <Button
                  block
                  size="small"
                  icon={<ThunderboltOutlined />}
                  onClick={() => setActiveComponent('process')}
                >
                  查看检索过程
                </Button>
                
                <Button
                  block
                  size="small"
                  icon={<BulbOutlined />}
                  onClick={() => setShowExplanation(!showExplanation)}
                >
                  {showExplanation ? '隐藏解释' : '显示解释'}
                </Button>
                
                <Button
                  block
                  size="small"
                  icon={<ExperimentOutlined />}
                  onClick={() => updateLayoutSettings({ 
                    compactMode: !layoutSettings.compactMode 
                  })}
                >
                  {layoutSettings.compactMode ? '标准模式' : '紧凑模式'}
                </Button>
              </Space>
            </Card>

          </Space>
        ) : (
          <Space direction="vertical" align="center" size="small">
            <Button
              type="text"
              icon={<HistoryOutlined />}
              onClick={() => updateLayoutSettings({ showSessionManager: true })}
            />
            <Button
              type="text"
              icon={<ThunderboltOutlined />}
              onClick={() => setActiveComponent('process')}
            />
            <Button
              type="text"
              icon={<BulbOutlined />}
              onClick={() => setShowExplanation(!showExplanation)}
            />
          </Space>
        )}
      </div>
    </Sider>
  );

  const renderMainContent = () => (
    <Content style={{ padding: '24px', minHeight: 'calc(100vh - 64px)' }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        
        {/* 查询面板 */}
        <div ref={componentRefs.query}>
          <AgenticQueryPanel 
            onSearch={() => {
              setActiveComponent('results');
              message.info('智能检索已启动');
            }}
            // compact={layoutSettings.compactMode}
          />
        </div>

        {/* 检索过程可视化 */}
        {(layoutSettings.showProcessViewer || activeComponent === 'process') && (
          <div ref={componentRefs.process}>
            <RetrievalProcessViewer
              compact={layoutSettings.compactMode}
            />
          </div>
        )}

        {/* 结果展示区域 */}
        <Row gutter={[16, 16]}>
          <Col span={layoutSettings.showExplanation ? 14 : 24}>
            <div ref={componentRefs.results}>
              <IntelligentResultsPanel
                onItemSelect={() => {
                  message.info('选中结果');
                }}
                onItemRate={() => {
                  message.success('评分完成');
                }}
              />
            </div>
          </Col>
          
          {layoutSettings.showExplanation && (
            <Col span={10}>
              <Affix offsetTop={100}>
                <div ref={componentRefs.explanation}>
                  <ExplanationViewer
                    compact={layoutSettings.compactMode}
                    onShare={() => {
                      message.success('解释内容已准备分享');
                    }}
                    onExport={() => {
                      message.success('解释内容已导出');
                    }}
                  />
                </div>
              </Affix>
            </Col>
          )}
        </Row>

        {/* 后备处理器 */}
        {agenticResults && (
          <FallbackHandler
            autoTrigger
            onFallbackSuccess={() => {
              message.success('后备处理成功');
            }}
            onFallbackFailed={() => {
              message.error('后备处理失败');
            }}
          />
        )}

        {/* 反馈区域 */}
        {layoutSettings.showFeedback && agenticResults && (
          <div ref={componentRefs.feedback}>
            <FeedbackInterface
              onFeedbackSubmit={() => {
                message.success('反馈提交成功，感谢您的宝贵意见！');
              }}
              compact={layoutSettings.compactMode}
            />
          </div>
        )}

      </Space>
    </Content>
  );

  const renderMobileDrawer = () => (
    <Drawer
      title="系统导航"
      placement="left"
      open={showMobileDrawer}
      onClose={() => setShowMobileDrawer(false)}
      width={300}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Button
          block
          icon={<ThunderboltOutlined />}
          onClick={() => {
            setActiveComponent('query');
            setShowMobileDrawer(false);
          }}
        >
          智能查询
        </Button>
        
        <Button
          block
          icon={<ExperimentOutlined />}
          onClick={() => {
            setActiveComponent('process');
            setShowMobileDrawer(false);
          }}
        >
          检索过程
        </Button>
        
        <Button
          block
          icon={<BulbOutlined />}
          onClick={() => {
            setActiveComponent('results');
            setShowMobileDrawer(false);
          }}
        >
          智能结果
        </Button>
        
        <Button
          block
          icon={<QuestionCircleOutlined />}
          onClick={() => {
            setShowExplanation(!showExplanation);
            setShowMobileDrawer(false);
          }}
        >
          {showExplanation ? '隐藏解释' : '显示解释'}
        </Button>
        
        <Divider />
        
        <Button
          block
          icon={<HistoryOutlined />}
          onClick={() => {
            updateLayoutSettings({ showSessionManager: true });
            setShowMobileDrawer(false);
          }}
        >
          会话管理
        </Button>
        
        <Button
          block
          icon={<SettingOutlined />}
          onClick={() => {
            updateLayoutSettings({ 
              compactMode: !layoutSettings.compactMode 
            });
            setShowMobileDrawer(false);
          }}
        >
          {layoutSettings.compactMode ? '标准模式' : '紧凑模式'}
        </Button>
      </Space>
    </Drawer>
  );

  // ==================== 渲染主组件 ====================

  return (
    <div className={`agentic-rag-page ${className}`}>
      
      <Layout style={{ minHeight: '100vh' }}>
        {renderHeader()}
        
        <Layout>
          {renderSider()}
          {renderMainContent()}
        </Layout>
      </Layout>

      {/* 移动端抽屉 */}
      {renderMobileDrawer()}

      {/* 会话管理模态框 */}
      <Modal
        title="会话管理"
        open={layoutSettings.showSessionManager}
        onCancel={() => updateLayoutSettings({ showSessionManager: false })}
        footer={null}
        width={1000}
        style={{ top: 50 }}
      >
        <SessionManager
          onSessionSelect={handleSessionSelect}
          onSessionCreate={(session) => {
            message.success(`会话 "${session.name}" 创建成功`);
          }}
          onSessionDelete={() => {
            message.info('会话已删除');
          }}
        />
      </Modal>

      {/* 系统引导 */}
      <Tour
        open={showTour}
        onClose={() => {
          setShowTour(false);
          localStorage.setItem('agentic_rag_tour_seen', 'true');
        }}
        steps={tourSteps}
        indicatorsRender={(current, total) => (
          <Space>
            {Array.from({ length: total }, (_, i) => (
              <div
                key={i}
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: i === current ? '#1890ff' : '#d9d9d9',
                }}
              />
            ))}
          </Space>
        )}
      />

      {/* 浮动按钮 */}
      <FloatButton.Group
        trigger="hover"
        type="primary"
        style={{ right: 24 }}
        icon={<RobotOutlined />}
      >
        <FloatButton
          icon={<QuestionCircleOutlined />}
          tooltip="使用指南"
          onClick={() => setShowTour(true)}
        />
        <FloatButton
          icon={<HistoryOutlined />}
          tooltip="会话管理"
          onClick={() => updateLayoutSettings({ showSessionManager: true })}
        />
        <FloatButton
          icon={<BulbOutlined />}
          tooltip={showExplanation ? '隐藏解释' : '显示解释'}
          onClick={() => setShowExplanation(!showExplanation)}
        />
      </FloatButton.Group>

      {/* 回到顶部 */}
      <BackTop />

      <style>{`
        .agentic-rag-page {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        @media (max-width: 768px) {
          .mobile-menu-btn {
            display: block !important;
          }
        }

        @media (min-width: 769px) {
          .mobile-menu-btn {
            display: none !important;
          }
        }

        .selected-cluster {
          background-color: #e6f7ff;
          border-color: #91d5ff;
        }

        .intelligent-result-item:hover {
          background-color: #f0f8ff;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .active-session {
          background-color: #f0f8ff;
        }
      `}</style>

    </div>
  );
};

export default AgenticRagPage;