/**
 * 格式化工具函数
 */

// 格式化文件大小
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) {
    return '0 Bytes';
  }
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// 格式化数字，添加千分位分隔符
export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat('zh-CN').format(num);
};

// 格式化百分比
export const formatPercent = (value: number, decimals: number = 1): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

// 格式化货币
export const formatCurrency = (amount: number, currency: string = 'CNY'): string => {
  return new Intl.NumberFormat('zh-CN', {
    style: 'currency',
    currency: currency
  }).format(amount);
};

// 格式化时间
export const formatTime = (date: Date | string | number): string => {
  const d = new Date(date);
  return d.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

// 格式化日期
export const formatDate = (date: Date | string | number): string => {
  const d = new Date(date);
  return d.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  });
};

// 格式化相对时间
export const formatRelativeTime = (date: Date | string | number): string => {
  const now = new Date();
  const target = new Date(date);
  const diffInSeconds = Math.floor((now.getTime() - target.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return '刚刚';
  }
  if (diffInSeconds < 3600) {
    return `${Math.floor(diffInSeconds / 60)}分钟前`;
  }
  if (diffInSeconds < 86400) {
    return `${Math.floor(diffInSeconds / 3600)}小时前`;
  }
  if (diffInSeconds < 2592000) {
    return `${Math.floor(diffInSeconds / 86400)}天前`;
  }
  if (diffInSeconds < 31536000) {
    return `${Math.floor(diffInSeconds / 2592000)}个月前`;
  }
  return `${Math.floor(diffInSeconds / 31536000)}年前`;
};

// 格式化持续时间
export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

// 截断字符串
export const truncateString = (str: string, maxLength: number, suffix: string = '...'): string => {
  if (str.length <= maxLength) {
    return str;
  }
  return str.substring(0, maxLength - suffix.length) + suffix;
};

// 首字母大写
export const capitalize = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

// 驼峰命名转换
export const toCamelCase = (str: string): string => {
  return str.replace(/([-_][a-z])/g, (group) =>
    group.toUpperCase().replace('-', '').replace('_', '')
  );
};

// 蛇形命名转换
export const toSnakeCase = (str: string): string => {
  return str.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '');
};

// 短横线命名转换
export const toKebabCase = (str: string): string => {
  return str.replace(/([A-Z])/g, '-$1').toLowerCase().replace(/^-/, '');
};

// 掩码处理（如手机号、邮箱）
export const maskString = (str: string, start: number, end: number, mask: string = '*'): string => {
  if (str.length <= start + end) {
    return str;
  }
  const startStr = str.substring(0, start);
  const endStr = str.substring(str.length - end);
  const maskLength = str.length - start - end;
  return startStr + mask.repeat(maskLength) + endStr;
};

// 手机号掩码
export const maskPhone = (phone: string): string => {
  return maskString(phone, 3, 4, '*');
};

// 邮箱掩码
export const maskEmail = (email: string): string => {
  const [username, domain] = email.split('@');
  if (username.length <= 2) {
    return email;
  }
  const maskedUsername = username.charAt(0) + '*'.repeat(username.length - 2) + username.charAt(username.length - 1);
  return `${maskedUsername}@${domain}`;
};
