import { test, expect } from '@playwright/test'

test.describe('Multi-Agent Flow Control', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/multi-agent')
    await page.waitForLoadState('domcontentloaded')
    
    // 等待页面完全加载
    await expect(page.locator('h1:has-text("多智能体协作")')).toBeVisible({ timeout: 15000 })
  })

  test('pause functionality stops token generation', async ({ page }) => {
    console.log('=== 测试暂停功能 ===')
    
    // 选择两个智能体
    const agentCheckboxes = page.locator('input[type="checkbox"]')
    await agentCheckboxes.first().check()
    await agentCheckboxes.nth(1).check()
    
    // 输入测试消息
    await page.fill('textarea', '请进行技术讨论，每个智能体详细回应，这是流控制测试')
    
    // 开始对话
    await page.click('button:has-text("开始多智能体讨论")')
    
    // 等待对话状态变为进行中
    await expect(page.locator('.bg-green-100:has-text("进行中")')).toBeVisible({ timeout: 10000 })
    console.log('对话已创建并开始')
    
    // 等待一些消息生成
    await page.waitForTimeout(8000)
    
    // 记录暂停前的消息数量
    const messagesBeforePause = await page.locator('.flex.flex-col.gap-4 > div').count()
    console.log(`暂停前消息数量: ${messagesBeforePause}`)
    
    // 点击暂停按钮
    const pauseButton = page.locator('button:has-text("暂停")')
    await expect(pauseButton).toBeVisible()
    await pauseButton.click()
    
    // 验证状态变为已暂停
    await expect(page.locator('.bg-yellow-100:has-text("已暂停")')).toBeVisible({ timeout: 5000 })
    console.log('对话状态已变为暂停')
    
    // 等待足够长的时间来验证没有新消息生成
    await page.waitForTimeout(10000)
    
    const messagesAfterPause = await page.locator('.flex.flex-col.gap-4 > div').count()
    console.log(`暂停后消息数量: ${messagesAfterPause}`)
    
    // 验证暂停期间没有新消息生成（允许1条正在完成的消息）
    expect(messagesAfterPause).toBeLessThanOrEqual(messagesBeforePause + 1)
    console.log('✅ 暂停功能验证通过 - token生成已停止')
  })

  test('resume functionality continues token generation', async ({ page }) => {
    console.log('=== 测试恢复功能 ===')
    
    // 先创建并暂停对话
    const agentCheckboxes = page.locator('input[type="checkbox"]')
    await agentCheckboxes.first().check()
    await agentCheckboxes.nth(1).check()
    
    await page.fill('textarea', '请进行技术讨论，每个智能体详细回应，这是恢复功能测试')
    await page.click('button:has-text("开始多智能体讨论")')
    
    await expect(page.locator('.bg-green-100:has-text("进行中")')).toBeVisible({ timeout: 10000 })
    await page.waitForTimeout(5000)
    
    // 暂停对话
    await page.click('button:has-text("暂停")')
    await expect(page.locator('.bg-yellow-100:has-text("已暂停")')).toBeVisible()
    
    // 记录恢复前的消息数量
    const messagesBeforeResume = await page.locator('.flex.flex-col.gap-4 > div').count()
    console.log(`恢复前消息数量: ${messagesBeforeResume}`)
    
    // 点击恢复按钮
    const resumeButton = page.locator('button:has-text("恢复")')
    await expect(resumeButton).toBeVisible()
    await resumeButton.click()
    
    // 验证状态变为进行中
    await expect(page.locator('.bg-green-100:has-text("进行中")')).toBeVisible({ timeout: 5000 })
    console.log('对话状态已变为进行中')
    
    // 等待新消息生成
    await page.waitForTimeout(12000)
    
    const messagesAfterResume = await page.locator('.flex.flex-col.gap-4 > div').count()
    console.log(`恢复后消息数量: ${messagesAfterResume}`)
    
    // 验证恢复后有新消息生成
    expect(messagesAfterResume).toBeGreaterThan(messagesBeforeResume)
    console.log('✅ 恢复功能验证通过 - token生成已恢复')
  })

  test('terminate functionality completely stops conversation', async ({ page }) => {
    console.log('=== 测试中止功能 ===')
    
    // 创建对话
    const agentCheckboxes = page.locator('input[type="checkbox"]')
    await agentCheckboxes.first().check()
    await agentCheckboxes.nth(1).check()
    
    await page.fill('textarea', '请进行技术讨论，这是中止功能测试')
    await page.click('button:has-text("开始多智能体讨论")')
    
    await expect(page.locator('.bg-green-100:has-text("进行中")')).toBeVisible({ timeout: 10000 })
    await page.waitForTimeout(5000)
    
    // 点击终止按钮
    const terminateButton = page.locator('button:has-text("终止")')
    await expect(terminateButton).toBeVisible()
    await terminateButton.click()
    
    // 在弹窗中输入终止原因并确认
    await page.fill('input[placeholder="请输入终止原因..."]', 'Playwright流控制测试完成')
    await page.click('button:has-text("确认终止")')
    
    // 等待对话被清理
    await page.waitForTimeout(3000)
    
    // 验证UI返回到创建对话状态
    await expect(page.locator('h3:has-text("创建多智能体对话")')).toBeVisible({ timeout: 5000 })
    console.log('✅ 中止功能验证通过 - 对话已完全停止并返回初始界面')
  })

  test('smart auto-scroll when user is at bottom', async ({ page }) => {
    console.log('=== 测试智能自动滚动（用户在底部）===')
    
    // 创建对话
    const agentCheckboxes = page.locator('input[type="checkbox"]')
    await agentCheckboxes.first().check()
    await agentCheckboxes.nth(1).check()
    
    await page.fill('textarea', '请进行详细的技术讨论，每个智能体都要详细回应多段内容，这是自动滚动测试')
    await page.click('button:has-text("开始多智能体讨论")')
    
    await expect(page.locator('.bg-green-100:has-text("进行中")')).toBeVisible({ timeout: 10000 })
    
    // 等待一些消息生成
    await page.waitForTimeout(10000)
    
    // 获取聊天容器
    const chatContainer = page.locator('.overflow-y-auto.p-4.bg-gray-50')
    await expect(chatContainer).toBeVisible()
    
    // 滚动到底部
    await chatContainer.evaluate(el => el.scrollTop = el.scrollHeight)
    
    // 记录滚动位置
    const scrollBefore = await chatContainer.evaluate(el => el.scrollTop)
    const scrollHeightBefore = await chatContainer.evaluate(el => el.scrollHeight)
    console.log(`滚动到底部 - 位置: ${scrollBefore}, 总高度: ${scrollHeightBefore}`)
    
    // 等待新消息出现
    await page.waitForTimeout(8000)
    
    // 检查是否自动滚动到底部
    const scrollAfter = await chatContainer.evaluate(el => el.scrollTop)
    const scrollHeightAfter = await chatContainer.evaluate(el => el.scrollHeight)
    const clientHeight = await chatContainer.evaluate(el => el.clientHeight)
    
    console.log(`新消息后 - 位置: ${scrollAfter}, 总高度: ${scrollHeightAfter}, 可见高度: ${clientHeight}`)
    
    // 验证自动滚动到底部（允许100px误差）
    const distanceFromBottom = scrollHeightAfter - scrollAfter - clientHeight
    expect(distanceFromBottom).toBeLessThanOrEqual(100)
    console.log('✅ 智能自动滚动（用户在底部）验证通过')
  })

  test('smart auto-scroll does not scroll when user is not at bottom', async ({ page }) => {
    console.log('=== 测试智能自动滚动（用户不在底部）===')
    
    // 创建对话并等待消息累积
    const agentCheckboxes = page.locator('input[type="checkbox"]')
    await agentCheckboxes.first().check()
    await agentCheckboxes.nth(1).check()
    
    await page.fill('textarea', '请进行详细的技术讨论，每个智能体都要详细回应多段内容，这是非底部滚动测试')
    await page.click('button:has-text("开始多智能体讨论")')
    
    await expect(page.locator('.bg-green-100:has-text("进行中")')).toBeVisible({ timeout: 10000 })
    
    // 等待足够的消息生成
    await page.waitForTimeout(15000)
    
    // 获取聊天容器
    const chatContainer = page.locator('.overflow-y-auto.p-4.bg-gray-50')
    await expect(chatContainer).toBeVisible()
    
    // 滚动到中间位置
    await chatContainer.evaluate(el => el.scrollTop = el.scrollHeight * 0.5)
    
    // 记录滚动位置
    const scrollBefore = await chatContainer.evaluate(el => el.scrollTop)
    console.log(`滚动到中间位置: ${scrollBefore}`)
    
    // 等待新消息出现
    await page.waitForTimeout(8000)
    
    // 检查滚动位置是否保持相对不变
    const scrollAfter = await chatContainer.evaluate(el => el.scrollTop)
    console.log(`新消息后滚动位置: ${scrollAfter}`)
    
    // 验证没有大幅度自动滚动（允许50px的小幅变化）
    const scrollChange = Math.abs(scrollAfter - scrollBefore)
    expect(scrollChange).toBeLessThanOrEqual(50)
    console.log('✅ 智能自动滚动（用户不在底部）验证通过 - 没有自动滚动')
  })
})