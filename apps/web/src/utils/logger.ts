type LogMethod = (...args: unknown[]) => void

const bindConsole = (method: LogMethod, enabled: boolean): LogMethod => {
  if (!enabled) return () => {}
  return method.bind(console)
}

const isProd = import.meta.env.PROD

export const logger = {
  debug: bindConsole(console.debug, !isProd),
  info: bindConsole(console.info, !isProd),
  log: bindConsole(console.log, !isProd),
  warn: bindConsole(console.warn, true),
  error: bindConsole(console.error, true),
}
