import React from 'react'
import { List, Card, Typography, Space, Button, Empty, Tag, Avatar } from 'antd'
import { 
import { logger } from '../../utils/logger'
  MessageOutlined, 
  DeleteOutlined, 
  CalendarOutlined,
  TeamOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  CheckCircleOutlined,
  StopOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { useMultiAgentStore, ConversationSession } from '../../stores/multiAgentStore'

const { Text, Title } = Typography

interface MultiAgentHistoryProps {
  visible: boolean
  onSelectSession: (session: ConversationSession) => void
}

const MultiAgentHistory: React.FC<MultiAgentHistoryProps> = ({
  visible,
  onSelectSession,
}) => {
  const { 
    sessions, 
    currentSession, 
    deleteSession, 
    getSessionSummary,
    loadSessionHistory 
  } = useMultiAgentStore()

  const handleDeleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    deleteSession(sessionId)
  }

  const handleSelectSession = async (session: ConversationSession) => {
    try {
      await loadSessionHistory(session.session_id)
      onSelectSession(session)
    } catch (error) {
      logger.error('选择会话失败:', error)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <PlayCircleOutlined className="text-green-500" />
      case 'paused':
        return <PauseCircleOutlined className="text-orange-500" />
      case 'completed':
        return <CheckCircleOutlined className="text-blue-500" />
      case 'terminated':
        return <StopOutlined className="text-red-500" />
      default:
        return <MessageOutlined className="text-gray-400" />
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active':
        return '进行中'
      case 'paused':
        return '已暂停'
      case 'completed':
        return '已完成'
      case 'terminated':
        return '已终止'
      case 'created':
        return '待启动'
      default:
        return '未知'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'green'
      case 'paused':
        return 'orange'
      case 'completed':
        return 'blue'
      case 'terminated':
        return 'red'
      case 'created':
        return 'gray'
      default:
        return 'default'
    }
  }

  if (!visible) return null

  // 按更新时间倒序排序
  const sortedSessions = [...sessions].sort((a, b) => 
    new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  )

  return (
    <div className="p-4">
      <div className="mb-4">
        <Title level={4} className="!mb-2">
          协作历史
        </Title>
        <Text type="secondary">共 {sessions.length} 个会话</Text>
      </div>

      {sessions.length === 0 ? (
        <Empty
          description="暂无对话历史"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <List
          dataSource={sortedSessions}
          renderItem={(session) => {
            const summary = getSessionSummary(session)
            const isCurrentSession = currentSession?.session_id === session.session_id
            
            return (
              <List.Item className="!px-0">
                <Card
                  hoverable
                  size="small"
                  className={`w-full cursor-pointer ${
                    isCurrentSession
                      ? 'border-blue-400 bg-blue-50'
                      : ''
                  }`}
                  onClick={() => handleSelectSession(session)}
                  actions={[
                    <Button
                      key="delete"
                      type="text"
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={(e) => handleDeleteSession(session.session_id, e)}
                    />,
                  ]}
                >
                  <div className="space-y-3">
                    {/* 标题和状态 */}
                    <div className="flex items-center justify-between">
                      <Text strong className="text-sm truncate flex-1 mr-2">
                        {summary.title}
                      </Text>
                      <Tag 
                        color={getStatusColor(session.status)} 
                        icon={getStatusIcon(session.status)}
                      >
                        {getStatusText(session.status)}
                      </Tag>
                    </div>

                    {/* 参与者信息 */}
                    <div className="flex items-center space-x-2">
                      <TeamOutlined className="text-gray-400" />
                      <div className="flex items-center space-x-1 flex-1">
                        <Avatar.Group size="small" maxCount={4}>
                          {session.participants.map((participant, index) => (
                            <Avatar
                              key={index}
                              size="small"
                              style={{
                                backgroundColor: [
                                  '#f56565',
                                  '#ed8936',
                                  '#38b2ac',
                                  '#4299e1',
                                  '#9f7aea'
                                ][index % 5]
                              }}
                            >
                              {participant.name.substring(0, 2)}
                            </Avatar>
                          ))}
                        </Avatar.Group>
                        <Text type="secondary" className="text-xs ml-2">
                          {session.participants.length} 个智能体
                        </Text>
                      </div>
                    </div>

                    {/* 对话统计和预览 */}
                    <div className="space-y-2">
                      <Text
                        type="secondary"
                        className="text-xs line-clamp-2"
                      >
                        {summary.preview}
                      </Text>
                      
                      <div className="flex items-center justify-between">
                        <Space size="small">
                          <MessageOutlined className="text-gray-400" />
                          <Text type="secondary" className="text-xs">
                            {summary.messageCount} 条消息
                          </Text>
                          {session.round_count > 0 && (
                            <>
                              <span className="text-gray-300">·</span>
                              <Text type="secondary" className="text-xs">
                                {session.round_count} 轮讨论
                              </Text>
                            </>
                          )}
                        </Space>
                        <Space size="small">
                          <CalendarOutlined className="text-gray-400" />
                          <Text type="secondary" className="text-xs">
                            {dayjs(session.updated_at).format('MM-DD HH:mm')}
                          </Text>
                        </Space>
                      </div>
                    </div>
                  </div>
                </Card>
              </List.Item>
            )
          }}
        />
      )}
    </div>
  )
}

export default MultiAgentHistory