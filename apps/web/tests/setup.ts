import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock window.matchMedia with proper matches property
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: query === '(max-width: 768px)' ? false : true, // 默认为桌面端
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock antd responsive observer with proper implementation
vi.mock('antd/lib/_util/responsiveObserver', () => ({
  default: {
    subscribe: vi.fn().mockImplementation(callback => {
      // 立即调用回调，模拟屏幕尺寸检测
      callback({ matches: true })
      return vi.fn() // 返回 unsubscribe 函数
    }),
    unsubscribe: vi.fn(),
    register: vi.fn().mockReturnValue(null),
    responsiveMap: {
      xs: '(max-width: 575px)',
      sm: '(min-width: 576px)',
      md: '(min-width: 768px)',
      lg: '(min-width: 992px)',
      xl: '(min-width: 1200px)',
      xxl: '(min-width: 1600px)',
    },
  },
}))

// Mock rc-util getScrollBarSize to avoid JSdom computedStyle issues
vi.mock('rc-util/lib/getScrollBarSize', () => ({
  default: vi.fn().mockReturnValue(17),
  __esModule: true,
}))

// Mock intersection observer
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockResolvedValue(undefined),
    readText: vi.fn().mockResolvedValue(''),
  },
})

// Mock window.getComputedStyle and computedStyle for JSdom compatibility
const mockComputedStyle = {
  getPropertyValue: vi.fn().mockReturnValue(''),
  setProperty: vi.fn(),
  removeProperty: vi.fn(),
  width: '100px',
  height: '100px',
  paddingLeft: '0px',
  paddingRight: '0px',
  marginLeft: '0px',
  marginRight: '0px',
  borderLeftWidth: '0px',
  borderRightWidth: '0px',
  scrollbarWidth: '17px',
  display: 'block',
  position: 'static',
  visibility: 'visible',
  overflow: 'visible',
}

// Mock both getComputedStyle and computedStyle
window.getComputedStyle = vi.fn().mockImplementation(() => mockComputedStyle)
window.computedStyle = vi.fn().mockImplementation(() => mockComputedStyle)

// Mock requestAnimationFrame for smooth animations in tests
global.requestAnimationFrame = vi.fn().mockImplementation(cb => {
  return setTimeout(cb, 16)
})

global.cancelAnimationFrame = vi.fn().mockImplementation(id => {
  return clearTimeout(id)
})
