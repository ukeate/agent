// 消息验证规则
export const validateMessage = (message: string): { isValid: boolean; error?: string } => {
  if (!message || message.trim().length === 0) {
    return { isValid: false, error: '消息不能为空' }
  }

  if (message.trim().length > 2000) {
    return { isValid: false, error: '消息长度不能超过2000个字符' }
  }

  // 检查是否包含恶意内容
  const maliciousPatterns = [
    /<script[^>]*>.*?<\/script>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi,
  ]

  for (const pattern of maliciousPatterns) {
    if (pattern.test(message)) {
      return { isValid: false, error: '消息包含不允许的内容' }
    }
  }

  return { isValid: true }
}

// 文件名验证
export const validateFileName = (fileName: string): { isValid: boolean; error?: string } => {
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
export const validateUrl = (url: string): { isValid: boolean; error?: string } => {
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
export const checkContentLength = (content: string, maxLength: number = 2000): boolean => {
  return content.length <= maxLength
}

// 检查是否为空白内容
export const isEmptyContent = (content: string): boolean => {
  return !content || content.trim().length === 0
}