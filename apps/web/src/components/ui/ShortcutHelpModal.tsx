import React from 'react'
import { List, Modal, Space, Tag, Typography } from 'antd'

const { Text } = Typography

type ShortcutItem = {
  keys: string[]
  label: string
  description?: string
}

const SHORTCUTS: ShortcutItem[] = [
  {
    keys: ['Ctrl/Cmd', 'K'],
    label: '打开快速搜索',
    description: '全局导航与路径直达',
  },
  {
    keys: ['/'],
    label: '打开快速搜索',
    description: '无需组合键快速唤起',
  },
  {
    keys: ['↑', '↓'],
    label: '在搜索结果中移动',
    description: '配合回车快速跳转',
  },
  {
    keys: ['Enter'],
    label: '确认跳转/发送消息',
  },
  {
    keys: ['Shift', 'Enter'],
    label: '输入换行',
  },
  {
    keys: ['Esc'],
    label: '关闭面板/停止生成',
    description: '关闭搜索面板或停止流式输出',
  },
]

type ShortcutHelpModalProps = {
  open: boolean
  onClose: () => void
}

const ShortcutHelpModal: React.FC<ShortcutHelpModalProps> = ({
  open,
  onClose,
}) => {
  return (
    <Modal
      title="快捷键速览"
      open={open}
      onCancel={onClose}
      footer={null}
      width={520}
      centered
    >
      <List
        dataSource={SHORTCUTS}
        renderItem={item => (
          <List.Item>
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Space wrap size={[6, 6]}>
                {item.keys.map(key => (
                  <Tag key={`${item.label}-${key}`}>{key}</Tag>
                ))}
              </Space>
              <Text strong>{item.label}</Text>
              {item.description && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {item.description}
                </Text>
              )}
            </Space>
          </List.Item>
        )}
      />
    </Modal>
  )
}

export default ShortcutHelpModal
