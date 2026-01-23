import { test, expect } from '@playwright/test'

test.describe('简单应用检查', () => {
  test('访问首页并截图', async ({ page }) => {
    await page.goto('http://localhost:3000')

    // 等待页面加载
    await page.waitForTimeout(3000)

    // 截图查看实际页面情况
    await page.screenshot({
      path: 'test-results/actual-homepage.png',
      fullPage: true,
    })

    // 检查页面是否完全白屏
    const bodyText = await page.locator('body').textContent()
    console.log('页面文本内容长度:', bodyText?.length || 0)

    // 检查是否有任何可见元素
    const visibleElements = await page
      .locator('*')
      .filter({ hasText: /\S/ })
      .count()
    console.log('可见元素数量:', visibleElements)

    // 检查DOM结构
    const rootElement = page.locator('#root')
    const rootExists = await rootElement.count()
    console.log('Root元素是否存在:', rootExists > 0)

    if (rootExists > 0) {
      const rootContent = await rootElement.innerHTML()
      console.log('Root元素内容长度:', rootContent.length)
      if (rootContent.length < 500) {
        console.log('Root元素内容:', rootContent.substring(0, 200))
      }
    }

    // 检查控制台错误
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log('控制台错误:', msg.text())
      }
    })

    console.log('✅ 简单页面检查完成')
  })

  test('尝试访问具体路由', async ({ page }) => {
    const routes = ['/chat', '/multi-agent', '/rag']

    for (const route of routes) {
      console.log(`测试路由: ${route}`)
      await page.goto(`http://localhost:3000${route}`)
      await page.waitForTimeout(2000)

      const pageContent = await page.textContent('body')
      console.log(`路由 ${route} 内容长度:`, pageContent?.length || 0)

      await page.screenshot({
        path: `test-results/route${route.replace('/', '-')}.png`,
      })
    }

    console.log('✅ 路由测试完成')
  })
})
