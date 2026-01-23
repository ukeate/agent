export const PALETTE_OPEN_EVENT = 'ai-agent-open-palette'

export const dispatchPaletteOpen = () => {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new Event(PALETTE_OPEN_EVENT))
}
