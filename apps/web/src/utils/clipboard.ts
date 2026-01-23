export const copyToClipboard = async (text: string): Promise<void> => {
  if (typeof navigator === 'undefined' || !navigator.clipboard?.writeText) {
    throw new Error('当前环境不支持复制')
  }
  await navigator.clipboard.writeText(text)
}
