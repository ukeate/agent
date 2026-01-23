export const clampIndex = (value: number, length: number) => {
  if (length <= 0) return 0
  if (value < 0) return 0
  if (value >= length) return length - 1
  return value
}

export const wrapIndex = (value: number, length: number) => {
  if (length <= 0) return 0
  const wrapped = value % length
  return wrapped < 0 ? wrapped + length : wrapped
}
