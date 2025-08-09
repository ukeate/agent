/**
 * 会话管理器组件
 * 
 * 功能包括：
 * - 创建、切换、删除和管理RAG会话
 * - 会话历史记录和上下文保持
 * - 会话统计和性能分析
 * - 会话导出和分享功能
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  Card,
  List,
  Space,
  Typography,
  Row,
  Col,
  Button,
  Input,
  Select,
  Modal,
  Popconfirm,
  Tag,
  Badge,
  Statistic,
  Empty,
  Divider,
  Tooltip,
  Avatar,
  Progress,
  Alert,
  Tabs,
  Timeline,
  Dropdown,
  Menu,
  message,
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  ShareAltOutlined,
  DownloadOutlined,
  HistoryOutlined,
  StarOutlined,
  StarFilled,
  ClockCircleOutlined,
  MessageOutlined,
  SearchOutlined,
  MoreOutlined,
  CopyOutlined,
  FolderOutlined,
  FileTextOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import { useRagStore, AgenticSession } from '../../stores/ragStore';

const { Text, Title, Paragraph } = Typography;
const { Search } = Input;
const { Option } = Select;
const { TabPane } = Tabs;

// ==================== 组件props类型 ====================

interface SessionManagerProps {
  className?: string;
  showStats?: boolean;
  maxSessions?: number;
  compact?: boolean;
  onSessionSelect?: (session: AgenticSession) => void;
  onSessionCreate?: (session: AgenticSession) => void;
  onSessionDelete?: (sessionId: string) => void;
}

// ==================== 辅助类型 ====================

interface SessionStats {
  total_sessions: number;
  active_sessions: number;
  total_queries: number;
  avg_queries_per_session: number;
  most_active_session: AgenticSession | null;
  recent_activity: Array<{
    session_id: string;
    session_name: string;
    query: string;
    timestamp: Date;
  }>;
}

interface SessionExportData {
  session: AgenticSession;
  context_history: string[];
  statistics: {
    duration: number;
    query_count: number;
    created_at: string;
    last_active: string;
  };
  export_timestamp: string;
}

// ==================== 主组件 ====================

const SessionManager: React.FC<SessionManagerProps> = ({
  className = '',
  showStats = true,
  maxSessions = 50,
  compact = false,
  onSessionSelect,
  onSessionCreate,
  onSessionDelete,
}) => {
  // ==================== 状态管理 ====================
  
  const {
    currentSession,
    sessions,
    createSession,
    switchSession,
    updateSession,
    deleteSession,
    addToSessionHistory,
  } = useRagStore();

  // ==================== 本地状态 ====================
  
  const [activeTab, setActiveTab] = useState<string>('sessions');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'created' | 'activity' | 'queries'>('activity');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [selectedSession, setSelectedSession] = useState<AgenticSession | null>(null);
  const [newSessionName, setNewSessionName] = useState('');
  const [favoriteSessionIds, setFavoriteSessionIds] = useState<Set<string>>(new Set());

  // ==================== 生命周期 ====================
  
  useEffect(() => {
    // 从localStorage加载收藏的会话
    const savedFavorites = localStorage.getItem('rag_favorite_sessions');
    if (savedFavorites) {
      try {
        const favorites = JSON.parse(savedFavorites);
        setFavoriteSessionIds(new Set(favorites));
      } catch (error) {
        console.error('Failed to load favorite sessions:', error);
      }
    }
  }, []);

  // ==================== 数据处理 ====================
  
  // 筛选和排序会话
  const filteredAndSortedSessions = useMemo(() => {
    let filtered = sessions.filter(session =>
      session.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      session.context_history.some(query => 
        query.toLowerCase().includes(searchTerm.toLowerCase())
      )
    );

    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'created':
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        case 'activity':
          return new Date(b.last_active).getTime() - new Date(a.last_active).getTime();
        case 'queries':
          return b.query_count - a.query_count;
        default:
          return 0;
      }
    });

    return filtered;
  }, [sessions, searchTerm, sortBy]);

  // 会话统计
  const sessionStats = useMemo((): SessionStats => {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

    const activeSessions = sessions.filter(s => 
      new Date(s.last_active) > oneHourAgo
    );

    const totalQueries = sessions.reduce((sum, s) => sum + s.query_count, 0);
    const avgQueries = sessions.length > 0 ? totalQueries / sessions.length : 0;

    const mostActive = sessions.reduce((max, session) => 
      session.query_count > (max?.query_count || 0) ? session : max, 
      null as AgenticSession | null
    );

    const recentActivity = sessions
      .flatMap(session => 
        session.context_history.slice(-3).map(query => ({
          session_id: session.id,
          session_name: session.name,
          query,
          timestamp: session.last_active,
        }))
      )
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 10);

    return {
      total_sessions: sessions.length,
      active_sessions: activeSessions.length,
      total_queries: totalQueries,
      avg_queries_per_session: avgQueries,
      most_active_session: mostActive,
      recent_activity: recentActivity,
    };
  }, [sessions]);

  // ==================== 事件处理 ====================
  
  const handleCreateSession = useCallback(() => {
    if (!newSessionName.trim()) {
      message.error('请输入会话名称');
      return;
    }

    if (sessions.length >= maxSessions) {
      message.error(`最多只能创建 ${maxSessions} 个会话`);
      return;
    }

    const sessionId = createSession(newSessionName.trim());
    const newSession = sessions.find(s => s.id === sessionId);
    
    if (newSession) {
      onSessionCreate?.(newSession);
      message.success(`会话 "${newSessionName.trim()}" 创建成功`);
    }

    setNewSessionName('');
    setShowCreateModal(false);
  }, [newSessionName, createSession, sessions, maxSessions, onSessionCreate]);

  const handleSelectSession = useCallback((session: AgenticSession) => {
    switchSession(session.id);
    onSessionSelect?.(session);
    message.info(`已切换到会话: ${session.name}`);
  }, [switchSession, onSessionSelect]);

  const handleRenameSession = useCallback(() => {
    if (!selectedSession || !newSessionName.trim()) {
      message.error('请输入新的会话名称');
      return;
    }

    updateSession(selectedSession.id, { name: newSessionName.trim() });
    message.success('会话名称修改成功');
    
    setNewSessionName('');
    setSelectedSession(null);
    setShowRenameModal(false);
  }, [selectedSession, newSessionName, updateSession]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId);
    if (!session) return;

    deleteSession(sessionId);
    onSessionDelete?.(sessionId);
    message.success(`会话 "${session.name}" 已删除`);

    // 如果删除的是收藏的会话，从收藏中移除
    if (favoriteSessionIds.has(sessionId)) {
      const newFavorites = new Set(favoriteSessionIds);
      newFavorites.delete(sessionId);
      setFavoriteSessionIds(newFavorites);
      localStorage.setItem('rag_favorite_sessions', JSON.stringify(Array.from(newFavorites)));
    }
  }, [sessions, deleteSession, onSessionDelete, favoriteSessionIds]);

  const handleToggleFavorite = useCallback((sessionId: string) => {
    const newFavorites = new Set(favoriteSessionIds);
    
    if (newFavorites.has(sessionId)) {
      newFavorites.delete(sessionId);
      message.info('已取消收藏');
    } else {
      newFavorites.add(sessionId);
      message.success('已添加到收藏');
    }
    
    setFavoriteSessionIds(newFavorites);
    localStorage.setItem('rag_favorite_sessions', JSON.stringify(Array.from(newFavorites)));
  }, [favoriteSessionIds]);

  const handleExportSession = useCallback((session: AgenticSession) => {
    const exportData: SessionExportData = {
      session,
      context_history: session.context_history,
      statistics: {
        duration: new Date(session.last_active).getTime() - new Date(session.created_at).getTime(),
        query_count: session.query_count,
        created_at: new Date(session.created_at).toISOString(),
        last_active: new Date(session.last_active).toISOString(),
      },
      export_timestamp: new Date().toISOString(),
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `rag_session_${session.name}_${Date.now()}.json`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    message.success('会话已导出');
  }, []);

  const handleShareSession = useCallback((session: AgenticSession) => {
    const shareData = {
      session_name: session.name,
      query_count: session.query_count,
      created_at: session.created_at,
      sample_queries: session.context_history.slice(-3),
    };
    
    const shareUrl = `${window.location.origin}/rag/session/share?data=${encodeURIComponent(JSON.stringify(shareData))}`;
    
    navigator.clipboard.writeText(shareUrl).then(() => {
      message.success('分享链接已复制到剪贴板');
    }).catch(() => {
      message.error('分享链接复制失败');
    });
  }, []);

  // ==================== 渲染辅助函数 ====================
  
  const renderSessionActions = (session: AgenticSession) => {
    const menuItems = [
      {
        key: 'rename',
        icon: <EditOutlined />,
        label: '重命名',
        onClick: () => {
          setSelectedSession(session);
          setNewSessionName(session.name);
          setShowRenameModal(true);
        },
      },
      {
        key: 'export',
        icon: <DownloadOutlined />,
        label: '导出',
        onClick: () => handleExportSession(session),
      },
      {
        key: 'share',
        icon: <ShareAltOutlined />,
        label: '分享',
        onClick: () => handleShareSession(session),
      },
      {
        key: 'copy',
        icon: <CopyOutlined />,
        label: '复制ID',
        onClick: () => {
          navigator.clipboard.writeText(session.id);
          message.success('会话ID已复制');
        },
      },
    ];

    return (
      <Dropdown
        menu={{
          items: menuItems,
          onClick: ({ key }) => {
            const item = menuItems.find(m => m.key === key);
            item?.onClick?.();
          }
        }}
        trigger={['click']}
        placement="bottomRight"
      >
        <Button type="text" icon={<MoreOutlined />} size="small" />
      </Dropdown>
    );
  };

  const renderSessionItem = (session: AgenticSession) => {
    const isActive = currentSession?.id === session.id;
    const isFavorite = favoriteSessionIds.has(session.id);
    const duration = new Date(session.last_active).getTime() - new Date(session.created_at).getTime();
    const durationText = duration > 24 * 60 * 60 * 1000 ? 
      `${Math.floor(duration / (24 * 60 * 60 * 1000))}天` :
      duration > 60 * 60 * 1000 ?
      `${Math.floor(duration / (60 * 60 * 1000))}小时` :
      `${Math.floor(duration / (60 * 1000))}分钟`;

    return (
      <List.Item
        key={session.id}
        className={isActive ? 'active-session' : ''}
        style={{
          border: isActive ? '2px solid #1890ff' : '1px solid #d9d9d9',
          borderRadius: 8,
          marginBottom: 8,
          padding: 16,
          backgroundColor: isActive ? '#f0f8ff' : 'white',
        }}
      >
        <Row style={{ width: '100%' }} align="middle">
          <Col span={1}>
            <Avatar
              size="small"
              style={{ 
                backgroundColor: isActive ? '#1890ff' : '#f0f0f0',
                color: isActive ? 'white' : '#666'
              }}
            >
              {session.name.charAt(0).toUpperCase()}
            </Avatar>
          </Col>
          
          <Col span={compact ? 12 : 10}>
            <Space direction="vertical" size="small">
              <Space>
                <Text strong ellipsis style={{ maxWidth: 150 }}>
                  {session.name}
                </Text>
                {isActive && <Badge status="processing" text="当前" />}
                {isFavorite && <StarFilled style={{ color: '#faad14' }} />}
              </Space>
              <Text type="secondary" style={{ fontSize: 12 }}>
                创建于: {new Date(session.created_at).toLocaleDateString()}
              </Text>
            </Space>
          </Col>

          {!compact && (
            <Col span={6}>
              <Space direction="vertical" size="small">
                <Statistic
                  title="查询数"
                  value={session.query_count}
                  prefix={<SearchOutlined />}
                  valueStyle={{ fontSize: 14 }}
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  活跃: {durationText}
                </Text>
              </Space>
            </Col>
          )}

          <Col span={compact ? 8 : 4}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {new Date(session.last_active).toLocaleString()}
            </Text>
          </Col>

          <Col span={compact ? 3 : 3}>
            <Space>
              <Tooltip title={isFavorite ? '取消收藏' : '收藏'}>
                <Button
                  type="text"
                  size="small"
                  icon={isFavorite ? <StarFilled /> : <StarOutlined />}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleToggleFavorite(session.id);
                  }}
                  style={{ color: isFavorite ? '#faad14' : undefined }}
                />
              </Tooltip>
              
              {!isActive && (
                <Popconfirm
                  title="确认删除此会话？"
                  description="删除后无法恢复"
                  onConfirm={() => handleDeleteSession(session.id)}
                  okText="确认"
                  cancelText="取消"
                >
                  <Button
                    type="text"
                    size="small"
                    icon={<DeleteOutlined />}
                    danger
                    onClick={(e) => e.stopPropagation()}
                  />
                </Popconfirm>
              )}
              
              {renderSessionActions(session)}
            </Space>
          </Col>
        </Row>

        {/* 最近查询预览 */}
        {!compact && session.context_history.length > 0 && (
          <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #f0f0f0' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              最近查询: 
            </Text>
            <div style={{ marginTop: 4 }}>
              {session.context_history.slice(-2).map((query, index) => (
                <Tag key={index} color="blue" style={{ margin: '2px', fontSize: 11 }}>
                  {query.length > 30 ? query.substring(0, 30) + '...' : query}
                </Tag>
              ))}
            </div>
          </div>
        )}
      </List.Item>
    );
  };

  const renderStatsTab = () => (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      
      {/* 总体统计 */}
      <Row gutter={16}>
        <Col span={6}>
          <Statistic
            title="总会话数"
            value={sessionStats.total_sessions}
            prefix={<FolderOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="活跃会话"
            value={sessionStats.active_sessions}
            prefix={<MessageOutlined />}
            suffix={`/ ${sessionStats.total_sessions}`}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="总查询数"
            value={sessionStats.total_queries}
            prefix={<SearchOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="平均查询数"
            value={sessionStats.avg_queries_per_session.toFixed(1)}
            prefix={<BarChartOutlined />}
            suffix="/ 会话"
          />
        </Col>
      </Row>

      {/* 最活跃会话 */}
      {sessionStats.most_active_session && (
        <Card size="small" title="最活跃会话">
          <Space>
            <Avatar>
              {sessionStats.most_active_session.name.charAt(0).toUpperCase()}
            </Avatar>
            <div>
              <Text strong>{sessionStats.most_active_session.name}</Text>
              <br />
              <Text type="secondary">
                {sessionStats.most_active_session.query_count} 次查询
              </Text>
            </div>
            <Button
              size="small"
              onClick={() => handleSelectSession(sessionStats.most_active_session!)}
            >
              切换到此会话
            </Button>
          </Space>
        </Card>
      )}

      {/* 最近活动 */}
      <Card size="small" title="最近活动">
        {sessionStats.recent_activity.length === 0 ? (
          <Empty description="暂无活动记录" />
        ) : (
          <Timeline size="small">
            {sessionStats.recent_activity.slice(0, 5).map((activity, index) => (
              <Timeline.Item key={index}>
                <Space direction="vertical" size="small">
                  <Space>
                    <Text strong>{activity.session_name}</Text>
                    <Text type="secondary">
                      {new Date(activity.timestamp).toLocaleString()}
                    </Text>
                  </Space>
                  <Text type="secondary" ellipsis style={{ maxWidth: 400 }}>
                    {activity.query}
                  </Text>
                </Space>
              </Timeline.Item>
            ))}
          </Timeline>
        )}
      </Card>

    </Space>
  );

  // ==================== 渲染主组件 ====================

  return (
    <div className={`session-manager ${className}`}>
      
      <Card
        title={
          <Space>
            <HistoryOutlined />
            <Title level={4} style={{ margin: 0 }}>会话管理</Title>
            <Badge count={sessions.length} showZero />
          </Space>
        }
        extra={
          <Space>
            {sessions.length < maxSessions && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setShowCreateModal(true)}
                size="small"
              >
                新建会话
              </Button>
            )}
          </Space>
        }
      >
        
        {sessions.length === 0 ? (
          <Empty 
            description="暂无会话记录"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          >
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setShowCreateModal(true)}
            >
              创建首个会话
            </Button>
          </Empty>
        ) : (
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            
            {/* 会话列表标签页 */}
            <TabPane
              tab={
                <Space>
                  <FolderOutlined />
                  会话列表
                </Space>
              }
              key="sessions"
            >
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                
                {/* 搜索和排序 */}
                <Row gutter={16} align="middle">
                  <Col span={12}>
                    <Search
                      placeholder="搜索会话名称或查询内容..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      allowClear
                    />
                  </Col>
                  <Col span={6}>
                    <Select
                      value={sortBy}
                      onChange={setSortBy}
                      style={{ width: '100%' }}
                      placeholder="排序方式"
                    >
                      <Option value="activity">最近活跃</Option>
                      <Option value="created">创建时间</Option>
                      <Option value="name">名称</Option>
                      <Option value="queries">查询数量</Option>
                    </Select>
                  </Col>
                  <Col span={6}>
                    <Text type="secondary">
                      共 {filteredAndSortedSessions.length} 个会话
                    </Text>
                  </Col>
                </Row>

                {/* 会话列表 */}
                <List
                  dataSource={filteredAndSortedSessions}
                  renderItem={renderSessionItem}
                  onClick={(session: AgenticSession) => handleSelectSession(session)}
                  pagination={{
                    pageSize: compact ? 5 : 10,
                    showSizeChanger: !compact,
                    showQuickJumper: !compact,
                    showTotal: (total, range) => 
                      `第 ${range[0]}-${range[1]} 个，共 ${total} 个会话`
                  }}
                />

              </Space>
            </TabPane>

            {/* 统计信息标签页 */}
            {showStats && !compact && (
              <TabPane
                tab={
                  <Space>
                    <BarChartOutlined />
                    统计信息
                  </Space>
                }
                key="stats"
              >
                {renderStatsTab()}
              </TabPane>
            )}

          </Tabs>
        )}

      </Card>

      {/* 创建会话模态框 */}
      <Modal
        title="创建新会话"
        open={showCreateModal}
        onOk={handleCreateSession}
        onCancel={() => {
          setShowCreateModal(false);
          setNewSessionName('');
        }}
        okText="创建"
        cancelText="取消"
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Input
            placeholder="请输入会话名称"
            value={newSessionName}
            onChange={(e) => setNewSessionName(e.target.value)}
            onPressEnter={handleCreateSession}
            maxLength={50}
            showCount
          />
          <Alert
            message={`当前已有 ${sessions.length} 个会话，最多可创建 ${maxSessions} 个会话`}
            type="info"
            showIcon
          />
        </Space>
      </Modal>

      {/* 重命名会话模态框 */}
      <Modal
        title="重命名会话"
        open={showRenameModal}
        onOk={handleRenameSession}
        onCancel={() => {
          setShowRenameModal(false);
          setNewSessionName('');
          setSelectedSession(null);
        }}
        okText="确认"
        cancelText="取消"
      >
        <Input
          placeholder="请输入新的会话名称"
          value={newSessionName}
          onChange={(e) => setNewSessionName(e.target.value)}
          onPressEnter={handleRenameSession}
          maxLength={50}
          showCount
        />
      </Modal>

    </div>
  );
};

export default SessionManager;