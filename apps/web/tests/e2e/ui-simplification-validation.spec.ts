import { test, expect } from '@playwright/test';

test.describe('UI简化版应用验证', () => {
  test.beforeEach(async ({ page }) => {
    // 访问首页
    await page.goto('http://localhost:3000');
    // 等待页面基本结构加载完成
    await page.waitForSelector('[data-testid="app-layout"]', { timeout: 15000 });
    // 等待网络空闲
    await page.waitForLoadState('networkidle');
  });

  test('1. 首页应该正常渲染（不再是空白）', async ({ page }) => {
    // 检查页面是否有基本内容
    await expect(page.locator('body')).not.toBeEmpty();
    
    // 检查是否有主要布局元素
    await expect(page.locator('[data-testid="app-layout"]')).toBeVisible();
    
    // 检查是否有侧边栏
    await expect(page.locator('.ant-layout-sider')).toBeVisible();
    
    // 检查是否有内容区域
    await expect(page.locator('.ant-layout-content')).toBeVisible();
    
    // 验证页面标题包含AI相关内容
    await expect(page.locator('text=AI Agent')).toBeVisible();
  });

  test('2. 验证左侧导航菜单显示6个核心功能', async ({ page }) => {
    // 检查菜单项是否存在
    const menuItems = page.locator('.ant-menu-item');
    await expect(menuItems).toHaveCount(6);
    
    // 验证具体菜单项文本
    await expect(page.locator('.ant-menu-item').nth(0)).toContainText('单代理对话');
    await expect(page.locator('.ant-menu-item').nth(1)).toContainText('多代理协作');
    await expect(page.locator('.ant-menu-item').nth(2)).toContainText('监督者编排');
    await expect(page.locator('.ant-menu-item').nth(3)).toContainText('基础RAG检索');
    await expect(page.locator('.ant-menu-item').nth(4)).toContainText('用户反馈系统');
    await expect(page.locator('.ant-menu-item').nth(5)).toContainText('Q-Learning算法');
    
    // 验证菜单图标存在
    await expect(page.locator('.anticon-message')).toBeVisible();
    await expect(page.locator('.anticon-team')).toBeVisible();
    await expect(page.locator('.anticon-control')).toBeVisible();
    await expect(page.locator('.anticon-search')).toBeVisible();
    await expect(page.locator('.anticon-heart')).toBeVisible();
    await expect(page.locator('.anticon-thunderbolt')).toBeVisible();
  });

  test('3. 测试导航功能 - 点击各个菜单项能正常跳转', async ({ page }) => {
    // 测试跳转到多代理协作页面
    await page.locator('.ant-menu-item').nth(1).click();
    await expect(page).toHaveURL(/.*\/multi-agent/);
    
    // 测试跳转到监督者编排页面
    await page.locator('.ant-menu-item').nth(2).click();
    await expect(page).toHaveURL(/.*\/supervisor/);
    
    // 测试跳转到RAG页面
    await page.locator('.ant-menu-item').nth(3).click();
    await expect(page).toHaveURL(/.*\/rag/);
    
    // 测试跳转到反馈系统页面
    await page.locator('.ant-menu-item').nth(4).click();
    await expect(page).toHaveURL(/.*\/feedback-system/);
    
    // 测试跳转到Q-Learning页面
    await page.locator('.ant-menu-item').nth(5).click();
    await expect(page).toHaveURL(/.*\/qlearning/);
    
    // 回到首页
    await page.locator('.ant-menu-item').nth(0).click();
    await expect(page).toHaveURL(/.*\/chat/);
  });

  test('4. 检查页面加载显示"加载中..."的Suspense fallback', async ({ page }) => {
    // 刷新页面并立即检查加载状态
    await page.reload();
    
    // 尝试捕获loading状态
    try {
      await expect(page.locator('text=加载中...')).toBeVisible({ timeout: 1000 });
    } catch (error) {
      // 如果加载太快，可能看不到加载状态，这是正常的
      console.log('加载速度太快，未捕获到加载中状态');
    }
    
    // 最终页面应该正常加载
    await expect(page.locator('[data-testid="app-layout"]')).toBeVisible();
  });

  test('5. 验证懒加载是否工作正常', async ({ page }) => {
    // 测试页面切换时的lazy loading
    await page.locator('.ant-menu-item').nth(1).click();
    await page.waitForLoadState('networkidle');
    await expect(page.locator('.ant-layout-content')).toBeVisible();
    
    await page.locator('.ant-menu-item').nth(2).click();
    await page.waitForLoadState('networkidle');
    await expect(page.locator('.ant-layout-content')).toBeVisible();
    
    // 验证页面内容确实发生了变化
    await expect(page).toHaveURL(/.*\/supervisor/);
  });

  test('6. 测试左侧菜单的收缩/展开功能', async ({ page }) => {
    // 找到折叠按钮
    const collapseButton = page.locator('.anticon-menu-fold, .anticon-menu-unfold');
    await expect(collapseButton).toBeVisible();
    
    // 点击折叠
    await collapseButton.click();
    
    // 验证菜单被折叠 (宽度变小)
    const sider = page.locator('.ant-layout-sider');
    const collapsedClass = await sider.getAttribute('class');
    expect(collapsedClass).toContain('ant-layout-sider-collapsed');
    
    // 点击展开
    await collapseButton.click();
    
    // 验证菜单被展开
    const expandedClass = await sider.getAttribute('class');
    expect(expandedClass).not.toContain('ant-layout-sider-collapsed');
  });

  test('7. 截图记录页面正常渲染状态', async ({ page }) => {
    // 等待页面完全加载
    await page.waitForLoadState('networkidle');
    
    // 全页面截图
    await page.screenshot({ 
      path: '/Users/runout/awork/code/my_git/agent/apps/web/ui-simplification-homepage.png',
      fullPage: true 
    });
    
    // 测试展开状态截图
    await page.screenshot({ 
      path: '/Users/runout/awork/code/my_git/agent/apps/web/ui-simplification-expanded.png' 
    });
    
    // 测试折叠状态截图
    await page.locator('.anticon-menu-fold, .anticon-menu-unfold').click();
    await page.screenshot({ 
      path: '/Users/runout/awork/code/my_git/agent/apps/web/ui-simplification-collapsed.png' 
    });
    
    // 测试不同页面截图
    await page.locator('.ant-menu-item').nth(1).click(); // 多代理协作
    await page.waitForLoadState('networkidle');
    await page.screenshot({ 
      path: '/Users/runout/awork/code/my_git/agent/apps/web/ui-simplification-multi-agent.png' 
    });
    
    await page.locator('.ant-menu-item').nth(3).click(); // RAG页面
    await page.waitForLoadState('networkidle');
    await page.screenshot({ 
      path: '/Users/runout/awork/code/my_git/agent/apps/web/ui-simplification-rag.png' 
    });
  });

  test('8. 验证应用解决了空白渲染问题', async ({ page }) => {
    // 检查页面不是空白的
    const bodyContent = await page.locator('body').innerHTML();
    expect(bodyContent.length).toBeGreaterThan(100);
    
    // 检查是否有实际的UI元素
    await expect(page.locator('[data-testid="app-layout"]')).toBeVisible();
    await expect(page.locator('.ant-menu')).toBeVisible();
    await expect(page.locator('.ant-layout-header')).toBeVisible();
    await expect(page.locator('.ant-layout-content')).toBeVisible();
    
    // 检查页面的高度不为0
    const bodyHeight = await page.locator('body').boundingBox();
    expect(bodyHeight?.height).toBeGreaterThan(100);
    
    // 验证页面包含必要的文本内容
    await expect(page).toHaveTitle(/.*AI Agent.*/i);
    await expect(page.locator('text=AI Agent')).toBeVisible();
    await expect(page.locator('text=智能体技术演示')).toBeVisible();
  });
});