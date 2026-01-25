import { describe, it, expect } from 'vitest'
import { normalizeApiBaseUrl } from '@/utils/apiBase'

describe('normalizeApiBaseUrl', () => {
  it('处理空值', () => {
    expect(normalizeApiBaseUrl('')).toBe('')
    expect(normalizeApiBaseUrl('   ')).toBe('')
  })

  it('移除末尾斜杠', () => {
    expect(normalizeApiBaseUrl('http://localhost:8000/')).toBe(
      'http://localhost:8000'
    )
  })

  it('去除末尾 api/v1 前缀', () => {
    expect(normalizeApiBaseUrl('http://localhost:8000/api/v1')).toBe(
      'http://localhost:8000'
    )
    expect(normalizeApiBaseUrl('http://localhost:8000/api/v1/')).toBe(
      'http://localhost:8000'
    )
    expect(normalizeApiBaseUrl('/api/v1')).toBe('')
  })

  it('保留非末尾前缀路径', () => {
    expect(normalizeApiBaseUrl('http://localhost:8000/api')).toBe(
      'http://localhost:8000/api'
    )
  })
})
