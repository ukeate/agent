import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UIState {
  // 侧边栏状态
  sidebarCollapsed: boolean

  // 主题配置
  theme: 'light' | 'dark'

  // 语言设置
  locale: 'zh-CN' | 'en-US'

  // 显示选项
  showReasoningSteps: boolean
  showToolDetails: boolean

  // 通知设置
  notifications: {
    sound: boolean
    desktop: boolean
  }

  // Actions
  setSidebarCollapsed: (collapsed: boolean) => void
  setTheme: (theme: 'light' | 'dark') => void
  setLocale: (locale: 'zh-CN' | 'en-US') => void
  setShowReasoningSteps: (show: boolean) => void
  setShowToolDetails: (show: boolean) => void
  updateNotifications: (updates: Partial<UIState['notifications']>) => void
  toggleSidebar: () => void
  resetUI: () => void
}

export const useUIStore = create<UIState>()(
  persist(
    set => ({
      // 初始状态
      sidebarCollapsed: false,
      theme: 'light',
      locale: 'zh-CN',
      showReasoningSteps: true,
      showToolDetails: false,
      notifications: {
        sound: false,
        desktop: false,
      },

      // Actions
      setSidebarCollapsed: collapsed => set({ sidebarCollapsed: collapsed }),

      setTheme: theme => set({ theme }),

      setLocale: locale => set({ locale }),

      setShowReasoningSteps: show => set({ showReasoningSteps: show }),

      setShowToolDetails: show => set({ showToolDetails: show }),

      updateNotifications: updates =>
        set(state => ({
          notifications: { ...state.notifications, ...updates },
        })),

      toggleSidebar: () =>
        set(state => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      resetUI: () =>
        set({
          sidebarCollapsed: false,
          theme: 'light',
          locale: 'zh-CN',
          showReasoningSteps: true,
          showToolDetails: false,
          notifications: {
            sound: false,
            desktop: false,
          },
        }),
    }),
    {
      name: 'ui-store',
    }
  )
)
