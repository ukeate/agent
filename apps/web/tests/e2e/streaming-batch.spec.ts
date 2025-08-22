/**
 * 流式处理和批处理E2E测试
 */

import { test, expect } from '@playwright/test';

test.describe('流式处理和批处理系统', () => {
  test.beforeEach(async ({ page }) => {
    // 访问统一监控页面
    await page.goto('http://localhost:3000/monitor');
  });

  test('统一监控页面正确加载', async ({ page }) => {
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('统一监控中心');
    
    // 检查标签页存在
    await expect(page.locator('text=流式处理')).toBeVisible();
    await expect(page.locator('text=批处理')).toBeVisible();
    await expect(page.locator('text=会话管理')).toBeVisible();
    await expect(page.locator('text=性能分析')).toBeVisible();
    
    // 检查系统运行状态
    await expect(page.locator('text=系统运行中')).toBeVisible();
  });

  test('流式处理监控面板功能', async ({ page }) => {
    // 点击流式处理标签
    await page.click('text=流式处理');
    
    // 等待数据加载
    await page.waitForSelector('text=系统指标', { timeout: 10000 });
    
    // 检查关键指标显示
    await expect(page.locator('text=活跃会话')).toBeVisible();
    await expect(page.locator('text=处理速率')).toBeVisible();
    await expect(page.locator('text=平均延迟')).toBeVisible();
    await expect(page.locator('text=错误率')).toBeVisible();
    
    // 检查背压控制部分
    await expect(page.locator('text=背压控制')).toBeVisible();
    
    // 检查流量控制部分
    await expect(page.locator('text=流量控制')).toBeVisible();
    
    // 检查队列健康度部分
    await expect(page.locator('text=队列健康度')).toBeVisible();
    
    // 测试自动刷新切换
    const autoRefreshCheckbox = page.locator('input[type="checkbox"]').first();
    await expect(autoRefreshCheckbox).toBeChecked();
    await autoRefreshCheckbox.uncheck();
    await expect(autoRefreshCheckbox).not.toBeChecked();
  });

  test('批处理监控面板功能', async ({ page }) => {
    // 点击批处理标签
    await page.click('text=批处理');
    
    // 等待数据加载
    await page.waitForSelector('text=批处理系统指标', { timeout: 10000 });
    
    // 检查批处理指标
    await expect(page.locator('text=活跃作业')).toBeVisible();
    await expect(page.locator('text=处理速率')).toBeVisible();
    await expect(page.locator('text=工作线程')).toBeVisible();
    await expect(page.locator('text=队列深度')).toBeVisible();
    
    // 检查作业列表部分
    await expect(page.locator('text=批处理作业')).toBeVisible();
  });

  test('会话管理功能', async ({ page }) => {
    // 点击会话管理标签
    await page.click('text=会话管理');
    
    // 等待页面加载
    await page.waitForSelector('text=创建流式会话', { timeout: 10000 });
    
    // 填写创建会话表单
    await page.fill('input[placeholder="智能体ID"]', 'test-agent-e2e');
    await page.fill('input[placeholder="消息内容"]', 'E2E测试消息');
    await page.fill('input[type="number"]', '200');
    
    // 点击创建会话按钮
    await page.click('text=创建会话');
    
    // 等待会话创建成功（实际环境中）
    // 注意：由于需要后端支持，这里只测试UI交互
    
    // 检查会话管理部分
    await expect(page.locator('text=会话管理')).toBeVisible();
  });

  test('性能分析功能', async ({ page }) => {
    // 点击性能分析标签
    await page.click('text=性能分析');
    
    // 等待性能分析加载
    await page.waitForSelector('text=性能评分', { timeout: 10000 });
    
    // 检查性能分析组件
    await expect(page.locator('text=性能瓶颈')).toBeVisible();
    await expect(page.locator('text=优化建议')).toBeVisible();
    
    // 检查时间范围选择器
    const timeRangeSelect = page.locator('select').first();
    await expect(timeRangeSelect).toBeVisible();
    await timeRangeSelect.selectOption('6h');
    await expect(timeRangeSelect).toHaveValue('6h');
    
    // 检查流式处理性能指标
    await expect(page.locator('text=流式处理性能')).toBeVisible();
    await expect(page.locator('text=P50延迟')).toBeVisible();
    await expect(page.locator('text=P95延迟')).toBeVisible();
    await expect(page.locator('text=P99延迟')).toBeVisible();
    
    // 检查批处理性能指标
    await expect(page.locator('text=批处理性能')).toBeVisible();
  });

  test('标签页切换功能', async ({ page }) => {
    // 测试所有标签页切换
    const tabs = ['流式处理', '批处理', '会话管理', '性能分析'];
    
    for (const tab of tabs) {
      await page.click(`text=${tab}`);
      
      // 等待内容加载
      await page.waitForTimeout(500);
      
      // 检查标签是否激活（通过类名判断）
      const tabButton = page.locator(`button:has-text("${tab}")`);
      const className = await tabButton.getAttribute('class');
      expect(className).toContain('border-blue-500');
    }
  });

  test('响应式布局', async ({ page }) => {
    // 测试桌面视图
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(page.locator('.max-w-7xl')).toBeVisible();
    
    // 测试平板视图
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('.max-w-7xl')).toBeVisible();
    
    // 测试移动视图
    await page.setViewportSize({ width: 375, height: 812 });
    await expect(page.locator('.max-w-7xl')).toBeVisible();
  });

  test('错误处理', async ({ page }) => {
    // 模拟网络错误
    await page.route('**/api/v1/streaming/**', route => {
      route.abort();
    });
    
    await page.goto('http://localhost:3000/monitor');
    await page.click('text=流式处理');
    
    // 应该显示错误信息或空状态
    await page.waitForTimeout(2000);
    
    // 恢复路由
    await page.unroute('**/api/v1/streaming/**');
  });

  test('数据自动刷新', async ({ page }) => {
    await page.click('text=流式处理');
    await page.waitForSelector('text=系统指标', { timeout: 10000 });
    
    // 确保自动刷新开启
    const autoRefreshCheckbox = page.locator('input[type="checkbox"]').first();
    if (!(await autoRefreshCheckbox.isChecked())) {
      await autoRefreshCheckbox.check();
    }
    
    // 记录初始请求数
    let requestCount = 0;
    page.on('request', request => {
      if (request.url().includes('/api/v1/streaming')) {
        requestCount++;
      }
    });
    
    // 等待自动刷新触发（默认5秒）
    await page.waitForTimeout(6000);
    
    // 应该有多次请求
    expect(requestCount).toBeGreaterThan(1);
  });

  test('页面底部信息显示', async ({ page }) => {
    // 滚动到页面底部
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    
    // 检查特性说明
    await expect(page.locator('text=流式处理特性')).toBeVisible();
    await expect(page.locator('text=批处理特性')).toBeVisible();
    await expect(page.locator('text=性能优化')).toBeVisible();
    
    // 检查具体特性项
    await expect(page.locator('text=SSE/WebSocket实时流')).toBeVisible();
    await expect(page.locator('text=任务调度和分片')).toBeVisible();
    await expect(page.locator('text=实时性能分析')).toBeVisible();
  });
});

test.describe('流式处理详细功能', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/streaming');
  });

  test('流式处理专用页面加载', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('流式处理监控');
    await expect(page.locator('text=监控面板')).toBeVisible();
    await expect(page.locator('text=会话管理')).toBeVisible();
  });

  test('创建和管理流式会话', async ({ page }) => {
    // 切换到会话管理标签
    await page.click('text=会话管理');
    
    // 填写表单
    await page.fill('input[placeholder="智能体ID"]', 'streaming-test');
    await page.fill('input[placeholder="消息内容"]', '测试流式消息');
    
    // 创建会话
    await page.click('button:has-text("创建会话")');
    
    // 等待响应（可能需要模拟后端）
    await page.waitForTimeout(1000);
  });

  test('SSE和WebSocket连接测试', async ({ page }) => {
    await page.click('text=会话管理');
    
    // 检查SSE和WebSocket按钮存在
    await page.waitForSelector('text=启动SSE流', { timeout: 10000 });
    await page.waitForSelector('text=启动WebSocket流', { timeout: 10000 });
  });
});

test.describe('性能和稳定性', () => {
  test('长时间运行稳定性', async ({ page }) => {
    await page.goto('http://localhost:3000/monitor');
    
    // 保持页面打开30秒
    await page.waitForTimeout(30000);
    
    // 检查页面仍然响应
    await page.click('text=批处理');
    await expect(page.locator('text=批处理系统指标')).toBeVisible();
  });

  test('大量数据渲染性能', async ({ page }) => {
    // 模拟返回大量数据
    await page.route('**/api/v1/batch/jobs', route => {
      const jobs = Array.from({ length: 100 }, (_, i) => ({
        id: `job-${i}`,
        status: i % 2 === 0 ? 'running' : 'completed',
        total_tasks: 100,
        completed_tasks: 50 + i,
        failed_tasks: i,
        created_at: new Date().toISOString()
      }));
      
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ jobs })
      });
    });
    
    await page.goto('http://localhost:3000/monitor');
    await page.click('text=批处理');
    
    // 页面应该能够处理大量数据
    await page.waitForSelector('text=批处理作业', { timeout: 10000 });
    
    // 检查性能（渲染时间）
    const startTime = Date.now();
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    const endTime = Date.now();
    
    // 渲染应该在合理时间内完成
    expect(endTime - startTime).toBeLessThan(2000);
  });
});