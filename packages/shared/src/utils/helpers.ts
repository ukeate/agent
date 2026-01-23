/**
 * 通用辅助工具函数
 */

// 生成UUID
export const generateUUID = (): string => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

// 生成随机字符串
export const generateRandomString = (length: number, chars?: string): string => {
  const characters = chars || 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
};

// 深拷贝
export const deepClone = <T>(obj: T): T => {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  if (obj instanceof Date) {
    return new Date(obj.getTime()) as T;
  }
  if (Array.isArray(obj)) {
    return (obj as unknown[]).map(item => deepClone(item)) as T;
  }
  if (typeof obj === 'object') {
    const copy: Record<string, unknown> = {};
    Object.keys(obj).forEach(key => {
      copy[key] = deepClone((obj as Record<string, unknown>)[key]);
    });
    return copy as T;
  }
  return obj;
};

// 深度合并对象
export const deepMerge = <T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T => {
  if (!sources.length) {
    return target;
  }
  const source = sources.shift();
  if (!source) {
    return target;
  }

  if (isObject(target) && isObject(source)) {
    const sourceRecord = source as Record<string, unknown>;
    const targetRecord = target as Record<string, unknown>;
    for (const key in sourceRecord) {
      const sourceValue = sourceRecord[key];
      if (isObject(sourceValue)) {
        if (!targetRecord[key]) {
          Object.assign(targetRecord, { [key]: {} });
        }
        deepMerge(
          targetRecord[key] as Record<string, unknown>,
          sourceValue as Record<string, unknown>
        );
      } else {
        Object.assign(targetRecord, { [key]: sourceValue });
      }
    }
  }

  return deepMerge(target, ...sources);
};

// 检查是否为对象
export const isObject = (item: unknown): item is Record<string, unknown> => {
  return Boolean(item) && typeof item === 'object' && !Array.isArray(item);
};

// 节流函数
export const throttle = <T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void => {
  let inThrottle = false;
  return function(this: unknown, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
};

// 防抖函数
export const debounce = <T extends (...args: unknown[]) => unknown>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void => {
  let timeoutId: NodeJS.Timeout;
  return function(this: unknown, ...args: Parameters<T>) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
};

// 睡眠函数
export const sleep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

// 重试函数
export const retry = async <T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  let lastError: Error | undefined;

  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < maxRetries) {
        await sleep(delay * Math.pow(2, i)); // 指数退避
      }
    }
  }

  throw lastError ?? new Error('重试失败');
};

// 数组去重
export const uniqueArray = <T>(array: T[], key?: keyof T): T[] => {
  if (!key) {
    return [...new Set(array)];
  }

  const seen = new Set<unknown>();
  return array.filter(item => {
    const value = item[key];
    if (seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
};

// 数组分组
export const groupBy = <T, K extends keyof T>(array: T[], key: K): Record<string, T[]> => {
  return array.reduce((result, item) => {
    const groupKey = String(item[key]);
    if (!result[groupKey]) {
      result[groupKey] = [];
    }
    result[groupKey].push(item);
    return result;
  }, {} as Record<string, T[]>);
};

// 数组分块
export const chunk = <T>(array: T[], size: number): T[][] => {
  const result: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    result.push(array.slice(i, i + size));
  }
  return result;
};

// 扁平化数组
export const flatten = <T>(array: (T | T[])[]): T[] => {
  return array.reduce<T[]>((acc, val) => {
    return acc.concat(Array.isArray(val) ? flatten(val) : val);
  }, []);
};

// 获取嵌套对象属性
export const getNestedValue = (
  obj: Record<string, unknown> | null | undefined,
  path: string,
  defaultValue?: unknown
): unknown => {
  return path.split('.').reduce<unknown>((current, key) => {
    if (current && typeof current === 'object') {
      const record = current as Record<string, unknown>;
      if (record[key] !== undefined) {
        return record[key];
      }
    }
    return defaultValue;
  }, obj ?? undefined);
};

// 设置嵌套对象属性
export const setNestedValue = (
  obj: Record<string, unknown>,
  path: string,
  value: unknown
): void => {
  const keys = path.split('.');
  const target = keys.slice(0, -1).reduce<Record<string, unknown>>((current, key) => {
    if (!current[key] || typeof current[key] !== 'object') {
      current[key] = {};
    }
    return current[key] as Record<string, unknown>;
  }, obj);

  target[keys[keys.length - 1]] = value;
};

// 比较两个值是否深度相等
export const isEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) {
    return true;
  }

  if (a === null || a === undefined || b === null || b === undefined) {
    return false;
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      return false;
    }
    return a.every((item, index) => isEqual(item, b[index]));
  }

  if (typeof a === 'object' && typeof b === 'object') {
    const recordA = a as Record<string, unknown>;
    const recordB = b as Record<string, unknown>;
    const keysA = Object.keys(recordA);
    const keysB = Object.keys(recordB);

    if (keysA.length !== keysB.length) {
      return false;
    }

    return keysA.every(key => isEqual(recordA[key], recordB[key]));
  }

  return false;
};
