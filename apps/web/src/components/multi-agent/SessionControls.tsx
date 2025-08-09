import React, { useState } from 'react'
import { ConversationSession } from '../../stores/multiAgentStore'

interface SessionControlsProps {
  session: ConversationSession | null
  loading: boolean
  onPause: () => void
  onResume: () => void
  onTerminate: (reason?: string) => void
  className?: string
}

export const SessionControls: React.FC<SessionControlsProps> = ({
  session,
  loading,
  onPause,
  onResume,
  onTerminate,
  className = '',
}) => {
  const [showTerminateConfirm, setShowTerminateConfirm] = useState(false)
  const [terminateReason, setTerminateReason] = useState('')

  if (!session) {
    return null
  }

  const canPause = session.status === 'active' && !loading
  const canResume = session.status === 'paused' && !loading
  const canTerminate = ['active', 'paused'].includes(session.status) && !loading
  
  const handleTerminate = () => {
    onTerminate(terminateReason || '用户手动终止')
    setShowTerminateConfirm(false)
    setTerminateReason('')
  }

  const getStatusDisplay = (status: string) => {
    const statusMap = {
      created: { text: '已创建', color: 'text-gray-600', bg: 'bg-gray-100' },
      active: { text: '进行中', color: 'text-green-600', bg: 'bg-green-100' },
      paused: { text: '已暂停', color: 'text-yellow-600', bg: 'bg-yellow-100' },
      completed: { text: '已完成', color: 'text-blue-600', bg: 'bg-blue-100' },
      terminated: { text: '已终止', color: 'text-red-600', bg: 'bg-red-100' },
      error: { text: '错误', color: 'text-red-600', bg: 'bg-red-100' },
    }
    
    return statusMap[status as keyof typeof statusMap] || statusMap.created
  }

  const statusInfo = getStatusDisplay(session.status)

  return (
    <div className={`bg-white border border-gray-200 rounded-lg p-4 ${className}`}>
      {/* 会话信息 */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-1">
            会话控制
          </h3>
          <div className="flex items-center gap-2">
            <span
              className={`
                inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
                ${statusInfo.bg} ${statusInfo.color}
              `}
            >
              {statusInfo.text}
            </span>
            <span className="text-sm text-gray-500">
              ID: {session.session_id.slice(-8)}
            </span>
          </div>
        </div>
      </div>

      {/* 会话统计 */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {session.message_count}
          </div>
          <div className="text-xs text-gray-600">消息数</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {session.round_count}
          </div>
          <div className="text-xs text-gray-600">轮次</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {session.participants.length}
          </div>
          <div className="text-xs text-gray-600">参与者</div>
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-2">
        {canPause && (
          <button
            onClick={onPause}
            disabled={loading}
            className="
              flex-1 bg-yellow-500 hover:bg-yellow-600 text-white
              px-4 py-2 rounded-md text-sm font-medium
              disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors
            "
          >
            {loading ? '处理中...' : '⏸️ 暂停'}
          </button>
        )}

        {canResume && (
          <button
            onClick={onResume}
            disabled={loading}
            className="
              flex-1 bg-green-500 hover:bg-green-600 text-white
              px-4 py-2 rounded-md text-sm font-medium
              disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors
            "
          >
            {loading ? '处理中...' : '▶️ 恢复'}
          </button>
        )}

        {canTerminate && (
          <button
            onClick={() => setShowTerminateConfirm(true)}
            disabled={loading}
            className="
              flex-1 bg-red-500 hover:bg-red-600 text-white
              px-4 py-2 rounded-md text-sm font-medium
              disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors
            "
          >
            🛑 终止
          </button>
        )}
      </div>

      {/* 会话配置信息 */}
      {session.config && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-900 mb-2">配置信息</h4>
          <div className="space-y-1 text-sm text-gray-600">
            <div className="flex justify-between">
              <span>最大轮次:</span>
              <span>{session.config.max_rounds}</span>
            </div>
            <div className="flex justify-between">
              <span>超时时间:</span>
              <span>{session.config.timeout_seconds}s</span>
            </div>
            <div className="flex justify-between">
              <span>自动回复:</span>
              <span>{session.config.auto_reply ? '启用' : '禁用'}</span>
            </div>
          </div>
        </div>
      )}

      {/* 终止确认弹窗 */}
      {showTerminateConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              确认终止会话
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              终止后将无法恢复会话，确定要继续吗？
            </p>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                终止原因 (可选)
              </label>
              <input
                type="text"
                value={terminateReason}
                onChange={(e) => setTerminateReason(e.target.value)}
                placeholder="请输入终止原因..."
                className="
                  w-full px-3 py-2 border border-gray-300 rounded-md
                  focus:outline-none focus:ring-2 focus:ring-blue-500
                "
              />
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={() => setShowTerminateConfirm(false)}
                className="
                  flex-1 bg-gray-300 hover:bg-gray-400 text-gray-700
                  px-4 py-2 rounded-md text-sm font-medium
                  transition-colors
                "
              >
                取消
              </button>
              <button
                onClick={handleTerminate}
                className="
                  flex-1 bg-red-500 hover:bg-red-600 text-white
                  px-4 py-2 rounded-md text-sm font-medium
                  transition-colors
                "
              >
                确认终止
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}