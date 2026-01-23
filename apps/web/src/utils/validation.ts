export const MESSAGE_MAX_LENGTH = 2000

// 消息验证规则
export const validateMessage = (
  message: string
): { isValid: boolean; error?: string } => {
  const normalized = message.trim()
  if (!normalized) {
    return { isValid: false, error: '消息不能为空' }
  }

  if (normalized.length > MESSAGE_MAX_LENGTH) {
    return {
      isValid: false,
      error: `消息长度不能超过${MESSAGE_MAX_LENGTH}个字符`,
    }
  }

  return { isValid: true }
}

// 文件名验证
export const validateFileName = (
  fileName: string
): { isValid: boolean; error?: string } => {
  if (!fileName || fileName.trim().length === 0) {
    return { isValid: false, error: '文件名不能为空' }
  }

  const invalidChars = /[<>:"/\\|?*]/
  if (invalidChars.test(fileName)) {
    return { isValid: false, error: '文件名包含非法字符' }
  }

  if (fileName.length > 255) {
    return { isValid: false, error: '文件名过长' }
  }

  return { isValid: true }
}

// URL验证
export const validateUrl = (
  url: string
): { isValid: boolean; error?: string } => {
  try {
    new URL(url)
    return { isValid: true }
  } catch {
    return { isValid: false, error: '无效的URL格式' }
  }
}

// 清理用户输入
export const sanitizeInput = (input: string): string => {
  return input
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;')
    .trim()
}

// 检查内容长度
export const checkContentLength = (
  content: string,
  maxLength: number = MESSAGE_MAX_LENGTH
): boolean => {
  return content.length <= maxLength
}

// 检查是否为空白内容
export const isEmptyContent = (content: string): boolean => {
  return !content || content.trim().length === 0
}
