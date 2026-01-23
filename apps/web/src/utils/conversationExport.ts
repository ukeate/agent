import { Message, ReasoningStep, ToolCall } from '@/types'

export const safeStringify = (value: unknown) => {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

const formatReasoningSteps = (steps: ReasoningStep[]) => {
  const typeLabelMap: Record<ReasoningStep['type'], string> = {
    thought: '思考',
    action: '行动',
    observation: '观察',
  }
  return steps
    .map((step, index) => {
      const label = typeLabelMap[step.type] || '步骤'
      const content = step.content?.trim()
      return content
        ? `- ${index + 1}. ${label}: ${content}`
        : `- ${index + 1}. ${label}`
    })
    .join('\n')
}

const formatToolCalls = (toolCalls: ToolCall[]) => {
  const statusLabelMap: Record<ToolCall['status'], string> = {
    success: '成功',
    error: '失败',
    pending: '执行中',
  }
  return toolCalls
    .map((toolCall, index) => {
      const lines = [
        `- ${index + 1}. ${toolCall.name} (${statusLabelMap[toolCall.status]})`,
      ]
      const args = safeStringify(toolCall.args)
      if (args) lines.push(`  参数: ${args}`)
      const result = safeStringify(toolCall.result)
      if (result) lines.push(`  结果: ${result}`)
      return lines.join('\n')
    })
    .join('\n')
}

const buildMessageHeader = (message: Message) => {
  const roleLabel = message.role === 'user' ? '用户' : 'AI助手'
  const timestamp = message.timestamp
    ? new Date(message.timestamp).toLocaleString()
    : ''
  return timestamp ? `${roleLabel} ${timestamp}` : roleLabel
}

export const buildMessageExportText = (
  message: Message,
  options?: { includeHeader?: boolean }
) => {
  const sections: string[] = []
  if (options?.includeHeader) {
    sections.push(buildMessageHeader(message))
  }
  if (message.content?.trim()) {
    sections.push(message.content)
  }
  if (message.role !== 'user') {
    if (message.reasoningSteps && message.reasoningSteps.length > 0) {
      sections.push(`推理过程:\n${formatReasoningSteps(message.reasoningSteps)}`)
    }
    if (message.toolCalls && message.toolCalls.length > 0) {
      sections.push(`工具调用:\n${formatToolCalls(message.toolCalls)}`)
    }
  }
  return sections.join('\n')
}

export const buildConversationExportText = (messages: Message[]) => {
  const parts = messages
    .map(message =>
      buildMessageExportText(message, { includeHeader: true }).trim()
    )
    .filter(Boolean)
  return parts.join('\n\n')
}
