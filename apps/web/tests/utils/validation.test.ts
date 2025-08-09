import { describe, it, expect } from 'vitest'
import {
  validateMessage,
  validateFileName,
  validateUrl,
  sanitizeInput,
  checkContentLength,
  isEmptyContent
} from '../../src/utils/validation'

describe('Validation Utils', () => {
  describe('validateMessage', () => {
    it('validates normal messages', () => {
      const result = validateMessage('Hello, this is a normal message')
      expect(result.isValid).toBe(true)
      expect(result.error).toBeUndefined()
    })

    it('rejects empty messages', () => {
      const result = validateMessage('')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('消息不能为空')
    })

    it('rejects whitespace-only messages', () => {
      const result = validateMessage('   \n\t   ')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('消息不能为空')
    })

    it('rejects messages that are too long', () => {
      const longMessage = 'a'.repeat(2001)
      const result = validateMessage(longMessage)
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('消息长度不能超过2000个字符')
    })

    it('rejects messages with script tags', () => {
      const result = validateMessage('<script>alert("xss")</script>')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('消息包含不允许的内容')
    })

    it('rejects messages with javascript: protocol', () => {
      const result = validateMessage('javascript:alert("xss")')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('消息包含不允许的内容')
    })

    it('rejects messages with event handlers', () => {
      const result = validateMessage('onclick=alert("xss")')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('消息包含不允许的内容')
    })

    it('accepts messages at the length limit', () => {
      const maxLengthMessage = 'a'.repeat(2000)
      const result = validateMessage(maxLengthMessage)
      expect(result.isValid).toBe(true)
    })
  })

  describe('validateFileName', () => {
    it('validates normal file names', () => {
      const result = validateFileName('document.pdf')
      expect(result.isValid).toBe(true)
    })

    it('rejects empty file names', () => {
      const result = validateFileName('')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('文件名不能为空')
    })

    it('rejects file names with invalid characters', () => {
      const result = validateFileName('file<name>.txt')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('文件名包含非法字符')
    })

    it('rejects file names that are too long', () => {
      const longFileName = 'a'.repeat(256)
      const result = validateFileName(longFileName)
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('文件名过长')
    })
  })

  describe('validateUrl', () => {
    it('validates valid HTTP URLs', () => {
      const result = validateUrl('http://example.com')
      expect(result.isValid).toBe(true)
    })

    it('validates valid HTTPS URLs', () => {
      const result = validateUrl('https://example.com/path')
      expect(result.isValid).toBe(true)
    })

    it('rejects invalid URLs', () => {
      const result = validateUrl('not-a-url')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('无效的URL格式')
    })

    it('rejects empty URLs', () => {
      const result = validateUrl('')
      expect(result.isValid).toBe(false)
      expect(result.error).toBe('无效的URL格式')
    })
  })

  describe('sanitizeInput', () => {
    it('escapes HTML characters', () => {
      const result = sanitizeInput('<script>alert("xss")</script>')
      expect(result).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;')
    })

    it('trims whitespace', () => {
      const result = sanitizeInput('  hello world  ')
      expect(result).toBe('hello world')
    })

    it('handles empty input', () => {
      const result = sanitizeInput('')
      expect(result).toBe('')
    })
  })

  describe('checkContentLength', () => {
    it('returns true for content within limit', () => {
      expect(checkContentLength('Hello', 10)).toBe(true)
    })

    it('returns false for content exceeding limit', () => {
      expect(checkContentLength('Hello World', 5)).toBe(false)
    })

    it('uses default limit of 2000', () => {
      const content = 'a'.repeat(2000)
      expect(checkContentLength(content)).toBe(true)
      
      const longContent = 'a'.repeat(2001)
      expect(checkContentLength(longContent)).toBe(false)
    })
  })

  describe('isEmptyContent', () => {
    it('returns true for empty string', () => {
      expect(isEmptyContent('')).toBe(true)
    })

    it('returns true for whitespace-only string', () => {
      expect(isEmptyContent('   \n\t   ')).toBe(true)
    })

    it('returns false for non-empty content', () => {
      expect(isEmptyContent('Hello')).toBe(false)
    })

    it('returns false for content with non-whitespace characters', () => {
      expect(isEmptyContent('  Hello  ')).toBe(false)
    })
  })
})