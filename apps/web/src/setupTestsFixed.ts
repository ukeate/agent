import { beforeAll, vi } from 'vitest'

// 模拟窗口和DOM API
beforeAll(() => {
  // 模拟 window.matchMedia
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(), // deprecated
      removeListener: vi.fn(), // deprecated
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })

  // 模拟 ResizeObserver
  global.ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

  // 模拟 IntersectionObserver
  global.IntersectionObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

  // 模拟 getComputedStyle
  Object.defineProperty(window, 'getComputedStyle', {
    value: () => ({
      scrollbarColor: 'auto',
      scrollbarWidth: 'auto',
      getPropertyValue: () => '',
    }),
  })

  // 模拟 document.elementFromPoint
  Object.defineProperty(document, 'elementFromPoint', {
    value: () => null,
  })

  // 模拟滚动相关方法
  Element.prototype.scrollIntoView = vi.fn()

  // 确保console方法存在
  global.console = {
    ...console,
    error: vi.fn(),
    warn: vi.fn(),
    log: vi.fn(),
  }
})