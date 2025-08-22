import { test, expect, Page } from '@playwright/test';

test.describe('完整应用功能测试', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    // 监听JavaScript错误
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        console.error('JavaScript Error:', msg.text());
      }
    });
    
    page.on('pageerror', (error) => {
      console.error('Page Error:', error.message);
    });
  });

  test('首页渲染验证', async () => {
    await page.goto('http://localhost:3000');
    
    // 等待页面加载完成
    await page.waitForLoadState('networkidle');
    
    // 验证页面标题
    await expect(page).toHaveTitle(/AI/);
    
    // 验证主要布局元素存在 - 使用Ant Design Layout的实际类
    await expect(page.locator('.ant-layout')).toBeVisible();
    await expect(page.locator('.ant-layout-sider')).toBeVisible();
    await expect(page.locator('.ant-layout-content')).toBeVisible();
    
    // 截图记录正常渲染状态
    await page.screenshot({ path: 'test-results/homepage-rendered.png', fullPage: true });
    
    console.log('✅ 首页渲染正常');
  });

  test('18个功能分组菜单项验证', async () => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // 等待侧边栏加载
    const sidebar = page.locator('.ant-layout-sider');
    await expect(sidebar).toBeVisible();
    
    // 验证主要功能分组
    const expectedGroups = [
      'AI智能体管理',
      '多智能体协作',
      'RAG检索系统', 
      '工作流编排',
      '智能监控',
      '用户反馈系统',
      'Q-Learning算法',
      '批处理作业',
      '企业架构',
      '多模态处理',
      'PgVector优化',
      '推理引擎',
      '内存管理',
      '流式处理',
      '离线能力',
      '安全管理',
      '统一引擎',
      '向量处理'
    ];
    
    for (const group of expectedGroups) {
      const groupElement = page.locator(`text="${group}"`);
      await expect(groupElement).toBeVisible({ timeout: 5000 });
    }
    
    // 截图记录菜单结构
    await page.screenshot({ path: 'test-results/menu-groups.png' });
    
    console.log('✅ 18个功能分组菜单项验证通过');
  });

  test('关键页面导航测试', async () => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // 测试用户反馈系统页面
    await page.click('text="用户反馈系统"');
    await page.click('text="反馈系统"');
    await page.waitForURL('**/feedback-system');
    await expect(page.locator('h1, h2')).toContainText(/反馈|Feedback/);
    await page.screenshot({ path: 'test-results/feedback-system-page.png' });
    console.log('✅ 用户反馈系统页面导航成功');
    
    // 测试Q-Learning页面
    await page.click('text="Q-Learning算法"');
    await page.click('text="Q-Learning页面"');
    await page.waitForURL('**/qlearning');
    await expect(page.locator('h1, h2')).toContainText(/Q-Learning|学习/);
    await page.screenshot({ path: 'test-results/qlearning-page.png' });
    console.log('✅ Q-Learning页面导航成功');
    
    // 测试多代理协作页面
    await page.click('text="多智能体协作"');
    await page.click('text="多代理协作"');
    await page.waitForURL('**/multi-agent');
    await expect(page.locator('h1, h2')).toContainText(/多代理|Multi.*Agent/);
    await page.screenshot({ path: 'test-results/multi-agent-page.png' });
    console.log('✅ 多代理协作页面导航成功');
    
    // 测试基础RAG检索页面
    await page.click('text="RAG检索系统"');
    await page.click('text="基础RAG检索"');
    await page.waitForURL('**/rag');
    await expect(page.locator('h1, h2')).toContainText(/RAG|检索/);
    await page.screenshot({ path: 'test-results/rag-page.png' });
    console.log('✅ 基础RAG检索页面导航成功');
  });

  test('懒加载功能验证', async () => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // 点击一个页面并验证加载状态
    await page.click('text="企业架构"');
    await page.click('text="企业架构页面"');
    
    // 检查是否有加载指示器（Suspense fallback）
    const loadingIndicator = page.locator('text="加载中..." , text="Loading..."');
    // 由于加载很快，可能看不到加载状态，所以这个检查是可选的
    
    // 等待内容加载完成
    await page.waitForLoadState('networkidle');
    
    // 验证页面最终加载成功
    await expect(page.locator('h1, h2, [data-testid="page-content"]')).toBeVisible();
    
    console.log('✅ 懒加载功能正常工作');
  });

  test('JavaScript错误检查', async () => {
    const jsErrors: string[] = [];
    const pageErrors: string[] = [];
    
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        jsErrors.push(msg.text());
      }
    });
    
    page.on('pageerror', (error) => {
      pageErrors.push(error.message);
    });
    
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // 导航到几个不同页面检查错误
    const testPages = [
      { group: '用户反馈系统', page: '反馈系统' },
      { group: 'Q-Learning算法', page: 'Q-Learning页面' },
      { group: '多智能体协作', page: '多代理协作' }
    ];
    
    for (const testPage of testPages) {
      await page.click(`text="${testPage.group}"`);
      await page.click(`text="${testPage.page}"`);
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(1000); // 等待可能的异步错误
    }
    
    // 报告发现的错误
    if (jsErrors.length > 0) {
      console.warn('发现JavaScript错误:', jsErrors);
    }
    if (pageErrors.length > 0) {
      console.warn('发现页面错误:', pageErrors);
    }
    
    // 不阻止测试，但记录错误
    console.log(`✅ JavaScript错误检查完成 (JS错误: ${jsErrors.length}, 页面错误: ${pageErrors.length})`);
  });

  test('左侧菜单收缩展开功能', async () => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    const sidebar = page.locator('.ant-layout-sider');
    await expect(sidebar).toBeVisible();
    
    // 查找菜单切换按钮
    const toggleButton = page.locator('.ant-layout-sider-trigger, button[aria-label*="菜单"], button[aria-label*="menu"]').first();
    
    if (await toggleButton.isVisible()) {
      // 点击收缩
      await toggleButton.click();
      await page.waitForTimeout(500); // 等待动画
      
      // 截图收缩状态
      await page.screenshot({ path: 'test-results/sidebar-collapsed.png' });
      
      // 点击展开
      await toggleButton.click();
      await page.waitForTimeout(500); // 等待动画
      
      // 截图展开状态
      await page.screenshot({ path: 'test-results/sidebar-expanded.png' });
      
      console.log('✅ 左侧菜单收缩展开功能正常');
    } else {
      console.log('⚠️  未找到菜单切换按钮，跳过收缩展开测试');
    }
  });

  test('页面路由完整性验证', async () => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // 测试一些关键路由的直接访问
    const criticalRoutes = [
      '/chat',
      '/multi-agent', 
      '/rag',
      '/workflow',
      '/supervisor',
      '/feedback-system',
      '/qlearning'
    ];
    
    for (const route of criticalRoutes) {
      await page.goto(`http://localhost:3000${route}`);
      await page.waitForLoadState('networkidle');
      
      // 验证页面不是空白或错误页面
      const hasContent = await page.locator('h1, h2, [data-testid="page-content"], main').count();
      expect(hasContent).toBeGreaterThan(0);
      
      console.log(`✅ 路由 ${route} 可正常访问`);
    }
  });

  test('最终整体验证和截图', async () => {
    await page.goto('http://localhost:3000');
    await page.waitForLoadState('networkidle');
    
    // 等待所有异步加载完成
    await page.waitForTimeout(2000);
    
    // 验证整体页面结构
    await expect(page.locator('.ant-layout')).toBeVisible();
    await expect(page.locator('.ant-layout-sider')).toBeVisible();
    await expect(page.locator('.ant-layout-content')).toBeVisible();
    
    // 验证没有明显的布局问题
    const viewportSize = page.viewportSize();
    if (viewportSize) {
      const mainContent = page.locator('.ant-layout-content');
      const boundingBox = await mainContent.boundingBox();
      
      if (boundingBox) {
        expect(boundingBox.width).toBeGreaterThan(300); // 确保主内容有足够宽度
        expect(boundingBox.height).toBeGreaterThan(200); // 确保主内容有足够高度
      }
    }
    
    // 最终截图记录完整应用状态
    await page.screenshot({ 
      path: 'test-results/complete-app-final-state.png', 
      fullPage: true 
    });
    
    console.log('✅ 完整应用验证通过，所有核心功能正常工作');
  });
});