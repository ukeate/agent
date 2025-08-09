/**
 * RAG系统主页面
 * 
 * 功能包括：
 * - 集成查询面板、结果展示、状态监控组件的统一布局
 * - 实现响应式设计支持桌面和移动端
 * - 添加页面级错误处理、loading状态和空数据状态
 * - 实现页面导航和面包屑导航
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Layout,
  Row,
  Col,
  Breadcrumb,
  Space,
  Button,
  Drawer,
  Typography,
  Alert,
  Spin,
  BackTop,
  FloatButton,
  Tooltip,
  message,
} from 'antd';
import {
  HomeOutlined,
  SearchOutlined,
  SettingOutlined,
  QuestionCircleOutlined,
  ExpandOutlined,
  CompressOutlined,
  MenuOutlined,
} from '@ant-design/icons';
import RagQueryPanel from '../components/rag/RagQueryPanel';
import RagResultsList from '../components/rag/RagResultsList';
import RagIndexStatus from '../components/rag/RagIndexStatus';
import { useRagStore } from '../stores/ragStore';
import { QueryRequest } from '../services/ragService';

const { Content } = Layout;
const { Title, Text } = Typography;

// ==================== 页面状态类型 ====================

type LayoutMode = 'desktop' | 'tablet' | 'mobile';
type PanelLayout = 'horizontal' | 'vertical' | 'focus';

// ==================== 主组件 ====================

const RagPage: React.FC = () => {
  // ==================== 状态管理 ====================
  
  const {
    // queryResults,
    isQuerying,
    error,
    currentQuery,
    clearErrors,
  } = useRagStore();

  // ==================== 本地状态 ====================
  
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('desktop');
  const [panelLayout, setPanelLayout] = useState<PanelLayout>('horizontal');
  const [showStatusDrawer, setShowStatusDrawer] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  // ==================== 响应式布局检测 ====================
  
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      if (width >= 1200) {
        setLayoutMode('desktop');
        setPanelLayout('horizontal');
      } else if (width >= 768) {
        setLayoutMode('tablet');
        setPanelLayout('vertical');
      } else {
        setLayoutMode('mobile');
        setPanelLayout('focus');
        setShowMobileMenu(false);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // ==================== 事件处理 ====================
  
  const handleSearch = useCallback((request: QueryRequest) => {
    clearErrors();
    message.info(`开始搜索: ${request.query}`);
    
    // 在移动端搜索后关闭菜单
    if (layoutMode === 'mobile') {
      setShowMobileMenu(false);
    }
  }, [clearErrors, layoutMode]);

  const handleResults = useCallback((response: any) => {
    if (response.success) {
      message.success(`搜索完成，找到 ${response.results.length} 个结果`);
    }
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const handleLayoutChange = useCallback((layout: PanelLayout) => {
    setPanelLayout(layout);
    message.info(`布局已切换为 ${layout === 'horizontal' ? '水平' : layout === 'vertical' ? '垂直' : '聚焦'} 模式`);
  }, []);

  // ==================== 渲染辅助函数 ====================
  
  const renderBreadcrumb = () => (
    <Row align="middle" justify="space-between" style={{ margin: '16px 0' }}>
      <Col>
        <Breadcrumb>
          <Breadcrumb.Item href="/">
            <HomeOutlined />
          </Breadcrumb.Item>
          <Breadcrumb.Item>
            <SearchOutlined />
            <span>RAG 搜索</span>
          </Breadcrumb.Item>
          {currentQuery && (
            <Breadcrumb.Item>
              <Text ellipsis style={{ maxWidth: 200 }}>
                "{currentQuery}"
              </Text>
            </Breadcrumb.Item>
          )}
        </Breadcrumb>
      </Col>
      <Col>
        <Space>
          <Button
            type="primary"
            ghost
            onClick={() => window.location.href = '/agentic-rag'}
            style={{ borderStyle: 'dashed' }}
          >
            升级到 Agentic RAG
          </Button>
          <Tooltip title="体验智能多代理检索系统">
            <Button
              type="link"
              onClick={() => window.location.href = '/agentic-rag'}
              style={{ fontSize: 12 }}
            >
              🚀 智能升级
            </Button>
          </Tooltip>
        </Space>
      </Col>
    </Row>
  );

  const renderDesktopLayout = () => (
    <Row gutter={[16, 16]}>
      {panelLayout === 'horizontal' ? (
        <>
          {/* 上方：查询面板 */}
          <Col span={24}>
            <RagQueryPanel
              onSearch={handleSearch}
              onResults={handleResults}
            />
          </Col>
          
          {/* 下方：结果和状态 */}
          <Col span={16}>
            <RagResultsList query={currentQuery} />
          </Col>
          <Col span={8}>
            <RagIndexStatus />
          </Col>
        </>
      ) : (
        <>
          {/* 左侧：查询面板和状态 */}
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <RagQueryPanel
                onSearch={handleSearch}
                onResults={handleResults}
              />
              <RagIndexStatus />
            </Space>
          </Col>
          
          {/* 右侧：结果展示 */}
          <Col span={16}>
            <RagResultsList query={currentQuery} />
          </Col>
        </>
      )}
    </Row>
  );

  const renderTabletLayout = () => (
    <Row gutter={[16, 16]}>
      {/* 上方：查询面板 */}
      <Col span={24}>
        <RagQueryPanel
          onSearch={handleSearch}
          onResults={handleResults}
        />
      </Col>
      
      {/* 中间：结果展示 */}
      <Col span={24}>
        <RagResultsList query={currentQuery} />
      </Col>
      
      {/* 底部：状态监控按钮 */}
      <Col span={24} style={{ textAlign: 'center' }}>
        <Button
          icon={<SettingOutlined />}
          onClick={() => setShowStatusDrawer(true)}
        >
          查看索引状态
        </Button>
      </Col>
    </Row>
  );

  const renderMobileLayout = () => (
    <Space direction="vertical" style={{ width: '100%' }} size="middle">
      {/* 查询面板 */}
      <RagQueryPanel
        onSearch={handleSearch}
        onResults={handleResults}
      />
      
      {/* 结果展示 */}
      <RagResultsList 
        query={currentQuery}
        pageSize={5} // 移动端减少每页显示数量
      />
    </Space>
  );

  const renderLayoutControls = () => {
    if (layoutMode === 'mobile') return null;
    
    return (
      <FloatButton.Group
        trigger="hover"
        type="primary"
        style={{ right: 24 }}
        icon={<SettingOutlined />}
      >
        <Tooltip title="水平布局" placement="left">
          <FloatButton
            icon={<ExpandOutlined />}
            onClick={() => handleLayoutChange('horizontal')}
          />
        </Tooltip>
        <Tooltip title="垂直布局" placement="left">
          <FloatButton
            icon={<CompressOutlined />}
            onClick={() => handleLayoutChange('vertical')}
          />
        </Tooltip>
        <Tooltip title="全屏模式" placement="left">
          <FloatButton
            icon={isFullscreen ? <CompressOutlined /> : <ExpandOutlined />}
            onClick={handleToggleFullscreen}
          />
        </Tooltip>
      </FloatButton.Group>
    );
  };

  // ==================== 渲染主组件 ====================

  return (
    <div className="rag-page">
      {/* 页面头部 */}
      <div style={{ padding: layoutMode === 'mobile' ? '8px 16px' : '16px 24px' }}>
        
        {/* 移动端头部 */}
        {layoutMode === 'mobile' && (
          <Row align="middle" justify="space-between" style={{ marginBottom: 16 }}>
            <Col>
              <Title level={3} style={{ margin: 0 }}>
                RAG 搜索
              </Title>
            </Col>
            <Col>
              <Space>
                <Button
                  icon={<SettingOutlined />}
                  onClick={() => setShowStatusDrawer(true)}
                  size="small"
                />
                <Button
                  icon={<MenuOutlined />}
                  onClick={() => setShowMobileMenu(true)}
                  size="small"
                />
              </Space>
            </Col>
          </Row>
        )}

        {/* 桌面端面包屑 */}
        {layoutMode !== 'mobile' && renderBreadcrumb()}

        {/* 全局错误提示 */}
        {error && (
          <Alert
            message="系统错误"
            description={error}
            type="error"
            showIcon
            closable
            onClose={clearErrors}
            style={{ marginBottom: 16 }}
          />
        )}

        {/* 全局加载状态 */}
        {isQuerying && (
          <Alert
            message={
              <Space>
                <Spin size="small" />
                <Text>正在搜索中...</Text>
              </Space>
            }
            type="info"
            style={{ marginBottom: 16 }}
          />
        )}

      </div>

      {/* 主内容区域 */}
      <Content style={{ 
        padding: layoutMode === 'mobile' ? '0 16px' : '0 24px',
        minHeight: 'calc(100vh - 200px)'
      }}>
        
        {/* 响应式布局渲染 */}
        {layoutMode === 'desktop' && renderDesktopLayout()}
        {layoutMode === 'tablet' && renderTabletLayout()}
        {layoutMode === 'mobile' && renderMobileLayout()}

        {/* 布局控制按钮 */}
        {renderLayoutControls()}

        {/* 回到顶部 */}
        <BackTop />

      </Content>

      {/* 状态监控抽屉 */}
      <Drawer
        title="索引状态监控"
        placement={layoutMode === 'mobile' ? 'bottom' : 'right'}
        width={layoutMode === 'mobile' ? undefined : 400}
        height={layoutMode === 'mobile' ? '70%' : undefined}
        open={showStatusDrawer}
        onClose={() => setShowStatusDrawer(false)}
      >
        <RagIndexStatus />
      </Drawer>

      {/* 移动端菜单抽屉 */}
      <Drawer
        title="菜单"
        placement="right"
        width={280}
        open={showMobileMenu}
        onClose={() => setShowMobileMenu(false)}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Button
            block
            icon={<SearchOutlined />}
            onClick={() => {
              setShowMobileMenu(false);
              // 滚动到查询面板
              document.querySelector('.rag-query-panel')?.scrollIntoView({ 
                behavior: 'smooth' 
              });
            }}
          >
            搜索面板
          </Button>
          
          <Button
            block
            icon={<SettingOutlined />}
            onClick={() => {
              setShowMobileMenu(false);
              setShowStatusDrawer(true);
            }}
          >
            索引状态
          </Button>
          
          <Button
            block
            icon={<QuestionCircleOutlined />}
            onClick={() => {
              setShowMobileMenu(false);
              message.info('帮助文档功能开发中...');
            }}
          >
            帮助文档
          </Button>
        </Space>
      </Drawer>

      {/* 页面样式 */}
      <style>{`
        .rag-page {
          min-height: 100vh;
          background-color: #f5f5f5;
        }
        
        @media (max-width: 768px) {
          .rag-page .ant-card {
            margin-bottom: 12px;
          }
          
          .rag-page .ant-card-body {
            padding: 12px;
          }
        }
      `}</style>
    </div>
  );
};

export default RagPage;