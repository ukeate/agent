/**
 * 验证工具函数
 */

// 邮箱验证
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// URL验证
export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

// 手机号验证（中国）
export const isValidPhone = (phone: string): boolean => {
  const phoneRegex = /^1[3-9]\d{9}$/;
  return phoneRegex.test(phone);
};

// 身份证验证（中国）
export const isValidIdCard = (idCard: string): boolean => {
  const idCardRegex = /(^\d{15}$)|(^\d{18}$)|(^\d{17}(\d|X|x)$)/;
  return idCardRegex.test(idCard);
};

// UUID验证
export const isValidUUID = (uuid: string): boolean => {
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  return uuidRegex.test(uuid);
};

// 密码强度验证
export const validatePasswordStrength = (password: string): {
  isValid: boolean;
  score: number;
  message: string;
} => {
  let score = 0;
  let message = '';

  if (password.length < 8) {
    return { isValid: false, score: 0, message: '密码长度至少8位' };
  }

  // 长度检查
  if (password.length >= 12) {
    score += 2;
  } else if (password.length >= 8) {
    score += 1;
  }

  // 复杂度检查
  if (/[a-z]/.test(password)) {
    score += 1; // 小写字母
  }
  if (/[A-Z]/.test(password)) {
    score += 1; // 大写字母
  }
  if (/\d/.test(password)) {
    score += 1; // 数字
  }

  const specialChars = '!@#$%^&*()_+-={}[];:\'"\\|,.<>/?';
  let hasSpecial = false;
  for (const char of specialChars) {
    if (password.includes(char)) {
      hasSpecial = true;
      break;
    }
  }
  if (hasSpecial) {
    score += 2; // 特殊字符
  }

  if (score >= 6) {
    message = '强密码';
  } else if (score >= 4) {
    message = '中等强度密码';
  } else if (score >= 2) {
    message = '弱密码';
  } else {
    message = '密码过于简单';
  }

  return {
    isValid: score >= 4,
    score,
    message
  };
};

// 非空验证
export const isNotEmpty = (value: unknown): boolean => {
  if (value === null || value === undefined) {
    return false;
  }
  if (typeof value === 'string') {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
};

// 数字范围验证
export const isInRange = (value: number, min: number, max: number): boolean => {
  return value >= min && value <= max;
};

// 字符串长度验证
export const isValidLength = (str: string, min: number, max?: number): boolean => {
  if (str.length < min) {
    return false;
  }
  if (max !== undefined && str.length > max) {
    return false;
  }
  return true;
};

// JSON验证
export const isValidJSON = (str: string): boolean => {
  try {
    JSON.parse(str);
    return true;
  } catch {
    return false;
  }
};

// 文件类型验证
export const isValidFileType = (fileName: string, allowedTypes: string[]): boolean => {
  const extension = fileName.split('.').pop()?.toLowerCase();
  return extension ? allowedTypes.includes(extension) : false;
};

// 文件大小验证
export const isValidFileSize = (fileSize: number, maxSize: number): boolean => {
  return fileSize <= maxSize;
};
