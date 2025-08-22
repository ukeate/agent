import { test, expect } from '@playwright/test';

test.describe('工作流可视化系统', () => {
  test.beforeEach(async ({ page }) => {
    // 导航到工作流页面
    await page.goto('/workflows');
  });

  test('工作流页面基本功能', async ({ page }) => {
    // 等待页面加载
    await expect(page.getByText('LangGraph 工作流可视化')).toBeVisible();
    
    // 检查工作流控制面板
    await expect(page.getByText('工作流控制')).toBeVisible();
    await expect(page.getByRole('button', { name: '启动工作流' })).toBeVisible();
    
    // 检查提示信息
    await expect(page.getByText('🚀 启动工作流以查看可视化图形')).toBeVisible();
    await expect(page.getByText('点击上方"启动工作流"按钮开始')).toBeVisible();
  });

  test('启动工作流并验证可视化', async ({ page }) => {
    // Mock API 响应以避免后端依赖
    await page.route('**/api/v1/workflows', async route => {
      await route.fulfill({
        json: { id: 'test-workflow-123', name: '演示工作流', status: 'running' }
      });
    });

    await page.route('**/api/v1/workflows/*/start', async route => {
      await route.fulfill({
        json: { id: 'test-workflow-123', status: 'started' }
      });
    });

    // 点击启动工作流按钮
    await page.getByRole('button', { name: '启动工作流' }).click();
    
    // 等待一下让组件加载
    await page.waitForTimeout(1000);
    
    // 检查工作流信息是否显示
    await expect(page.getByText(/当前工作流:/)).toBeVisible();
    await expect(page.getByText(/演示工作流/)).toBeVisible();
    
    // 检查状态显示
    await expect(page.getByText(/状态:/)).toBeVisible();
  });

  test('节点操作功能', async ({ page }) => {
    // 点击运行中的节点
    await page.waitForSelector('.react-flow__node');
    const runningNode = page.locator('.react-flow__node').filter({ hasText: '数据处理' });
    await runningNode.click();
    
    // 等待详情面板打开
    await expect(page.getByText('节点详情')).toBeVisible();
    
    // 测试暂停操作
    await page.getByRole('button', { name: '暂停' }).click();
    
    // 检查成功消息
    await expect(page.getByText(/操作执行成功/)).toBeVisible();
    
    // 验证按钮状态变化 - 应该显示恢复按钮
    await expect(page.getByRole('button', { name: '恢复' })).toBeVisible();
    
    // 测试恢复操作
    await page.getByRole('button', { name: '恢复' }).click();
    await expect(page.getByText(/操作执行成功/)).toBeVisible();
  });

  test('调试面板功能', async ({ page }) => {
    // 等待页面加载
    await page.waitForSelector('[data-testid="workflow-page"]');
    
    // 点击调试按钮
    await page.getByText('🐛 调试').click();
    
    // 检查调试面板是否打开
    await expect(page.getByText('工作流调试')).toBeVisible();
    
    // 检查调试面板的标签页
    await expect(page.getByText('状态历史')).toBeVisible();
    await expect(page.getByText('执行日志')).toBeVisible();
    await expect(page.getByText('当前状态')).toBeVisible();
    
    // 测试状态历史标签页
    await page.getByText('状态历史').click();
    await expect(page.getByText('时间轴视图')).toBeVisible();
    await expect(page.getByText('表格视图')).toBeVisible();
    
    // 测试执行日志标签页
    await page.getByText('执行日志').click();
    await expect(page.locator('table')).toBeVisible();
    
    // 测试当前状态标签页
    await page.getByText('当前状态').click();
    await expect(page.getByText('工作流状态')).toBeVisible();
    
    // 关闭调试面板
    await page.getByRole('button', { name: '关闭' }).click();
    await expect(page.getByText('工作流调试')).not.toBeVisible();
  });

  test('工作流状态实时更新', async ({ page }) => {
    // 等待工作流图形加载
    await page.waitForSelector('.react-flow__node');
    
    // 模拟工作流状态变化（通过WebSocket）
    // 注意：在实际测试中，这需要后端支持或使用mock
    
    // 检查初始状态
    const startNode = page.locator('.react-flow__node').filter({ hasText: '开始' });
    await expect(startNode).toHaveClass(/completed/);
    
    const processNode = page.locator('.react-flow__node').filter({ hasText: '数据处理' });
    await expect(processNode).toHaveClass(/running/);
  });

  test('工作流可视化响应式设计', async ({ page }) => {
    // 测试桌面视图
    await page.setViewportSize({ width: 1200, height: 800 });
    await page.waitForSelector('.react-flow');
    
    // 检查所有元素是否正确显示
    await expect(page.locator('.react-flow__controls')).toBeVisible();
    await expect(page.locator('.react-flow__minimap')).toBeVisible();
    await expect(page.getByText('🐛 调试')).toBeVisible();
    
    // 测试平板视图
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('.react-flow')).toBeVisible();
    
    // 测试移动设备视图
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('.react-flow')).toBeVisible();
  });

  test('工作流错误处理', async ({ page }) => {
    // 模拟网络错误
    await page.route('**/api/v1/workflows/**', route => {
      route.abort('failed');
    });
    
    await page.goto('/workflows');
    
    // 检查错误信息是否显示
    await expect(page.getByText('加载工作流失败')).toBeVisible();
    
    // 取消路由拦截
    await page.unroute('**/api/v1/workflows/**');
  });

  test('工作流图形操作', async ({ page }) => {
    // 等待图形加载
    await page.waitForSelector('.react-flow');
    
    // 测试缩放功能
    const zoomInButton = page.locator('.react-flow__controls-zoomin');
    await zoomInButton.click();
    
    const zoomOutButton = page.locator('.react-flow__controls-zoomout');
    await zoomOutButton.click();
    
    // 测试适应视图功能
    const fitViewButton = page.locator('.react-flow__controls-fitview');
    await fitViewButton.click();
    
    // 测试拖拽功能（如果启用）
    const flowContainer = page.locator('.react-flow__pane');
    await flowContainer.hover();
    
    // 测试小地图交互
    const minimap = page.locator('.react-flow__minimap');
    await minimap.click();
  });

  test('工作流数据持久化', async ({ page }) => {
    // 打开节点详情面板
    await page.waitForSelector('.react-flow__node');
    const firstNode = page.locator('.react-flow__node').first();
    await firstNode.click();
    
    // 执行节点操作
    await page.getByRole('button', { name: '暂停' }).click();
    await expect(page.getByText(/操作执行成功/)).toBeVisible();
    
    // 刷新页面
    await page.reload();
    
    // 验证状态是否保持
    await page.waitForSelector('.react-flow__node');
    // 注意：实际测试需要后端支持状态持久化
  });
});