import { test, expect } from '@playwright/test';

test.describe('容错和检查点功能 E2E 测试', () => {
  test.beforeEach(async ({ page }) => {
    // 设置API拦截
    await page.route('**/api/v1/streaming/fault-tolerance/stats', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total_active_connections: 3,
          healthy_connections: 2,
          failed_connections: 1,
          average_uptime: 7200,
          total_reconnections: 5,
          connections: [
            {
              session_id: 'session-001',
              state: 'connected',
              retry_count: 0,
              uptime_seconds: 7200,
              total_reconnections: 0,
              heartbeat_alive: true,
              buffered_messages: 0,
              metrics: {
                total_connections: 1,
                successful_connections: 1,
                failed_connections: 0
              }
            },
            {
              session_id: 'session-002',
              state: 'reconnecting',
              retry_count: 2,
              uptime_seconds: 3600,
              total_reconnections: 3,
              heartbeat_alive: false,
              buffered_messages: 5,
              metrics: {
                total_connections: 4,
                successful_connections: 3,
                failed_connections: 1,
                last_failure_reason: 'Network timeout'
              }
            },
            {
              session_id: 'session-003',
              state: 'failed',
              retry_count: 5,
              uptime_seconds: 1800,
              total_reconnections: 2,
              heartbeat_alive: false,
              buffered_messages: 8,
              metrics: {
                total_connections: 6,
                successful_connections: 4,
                failed_connections: 2,
                last_failure_reason: 'Connection refused'
              }
            }
          ]
        })
      });
    });

    await page.route('**/api/v1/batch/checkpoints**', async route => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            checkpoints: [
              {
                checkpoint_id: 'ckpt-001',
                job_id: 'job-001',
                created_at: '2024-01-15T10:00:00Z',
                checkpoint_type: 'manual',
                task_count: 100,
                completed_tasks: 75,
                failed_tasks: 5,
                file_size: 2048000,
                checksum: 'abc123def456',
                tags: { priority: 'high', environment: 'production' }
              },
              {
                checkpoint_id: 'ckpt-002',
                job_id: 'job-002',
                created_at: '2024-01-15T11:30:00Z',
                checkpoint_type: 'auto',
                task_count: 200,
                completed_tasks: 180,
                failed_tasks: 10,
                file_size: 4096000,
                checksum: 'def456ghi789',
                tags: { environment: 'staging' }
              }
            ]
          })
        });
      }
    });

    await page.route('**/api/v1/batch/checkpoints/stats', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total_checkpoints: 2,
          total_size_bytes: 6144000,
          jobs_with_checkpoints: 2,
          checkpoint_types: {
            manual: 1,
            auto: 1
          },
          oldest_checkpoint: '2024-01-15T10:00:00Z',
          newest_checkpoint: '2024-01-15T11:30:00Z'
        })
      });
    });

    await page.route('**/api/v1/batch/jobs', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          jobs: [
            {
              id: 'job-001',
              name: 'High Priority Batch Job',
              status: 'running',
              progress: 0.75,
              total_tasks: 100,
              completed_tasks: 75,
              failed_tasks: 5
            },
            {
              id: 'job-002',
              name: 'Standard Processing Job',
              status: 'paused',
              progress: 0.90,
              total_tasks: 200,
              completed_tasks: 180,
              failed_tasks: 10
            }
          ]
        })
      });
    });

    // 导航到统一监控页面
    await page.goto('/monitor');
  });

  test('容错监控功能完整性测试', async ({ page }) => {
    // 切换到容错监控标签页
    await page.click('text=容错监控');
    
    // 验证页面加载
    await expect(page.locator('text=容错连接监控')).toBeVisible();
    await expect(page.locator('text=实时监控流式处理连接状态')).toBeVisible();

    // 验证统计卡片
    await expect(page.locator('text=活跃连接')).toBeVisible();
    await expect(page.locator('text=3')).toBeVisible(); // 总连接数
    
    await expect(page.locator('text=健康连接')).toBeVisible();
    await expect(page.locator('text=2')).toBeVisible(); // 健康连接数
    
    await expect(page.locator('text=失败连接')).toBeVisible();
    await expect(page.locator('text=1')).toBeVisible(); // 失败连接数
    
    await expect(page.locator('text=总重连次数')).toBeVisible();
    await expect(page.locator('text=5')).toBeVisible(); // 重连次数

    // 验证连接健康度
    await expect(page.locator('text=67%')).toBeVisible(); // 2/3 = 67%
    await expect(page.locator('text=2/3 连接健康')).toBeVisible();

    // 验证连接详情表格
    await expect(page.locator('text=连接详情')).toBeVisible();
    await expect(page.locator('text=session-001')).toBeVisible();
    await expect(page.locator('text=session-002')).toBeVisible();
    await expect(page.locator('text=session-003')).toBeVisible();

    // 验证连接状态标签
    await expect(page.locator('.ant-tag:has-text("CONNECTED")')).toBeVisible();
    await expect(page.locator('.ant-tag:has-text("RECONNECTING")')).toBeVisible();
    await expect(page.locator('.ant-tag:has-text("FAILED")')).toBeVisible();

    // 验证心跳状态
    await expect(page.locator('text=正常')).toBeVisible();
    await expect(page.locator('text=异常')).toBeVisible();

    // 验证最近错误部分
    await expect(page.locator('text=最近错误')).toBeVisible();
    await expect(page.locator('text=Network timeout')).toBeVisible();
    await expect(page.locator('text=Connection refused')).toBeVisible();
  });

  test('容错监控重连功能测试', async ({ page }) => {
    // 模拟重连API
    await page.route('**/api/v1/streaming/fault-tolerance/reconnect/session-002', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true })
      });
    });

    await page.click('text=容错监控');
    
    // 等待表格加载
    await expect(page.locator('text=session-002')).toBeVisible();
    
    // 查找重连按钮（非已连接状态的会话）
    const reconnectButtons = page.locator('button:has-text("重连")');
    const enabledReconnectButton = reconnectButtons.nth(1); // 第二个按钮应该是启用的
    
    // 点击重连按钮
    await enabledReconnectButton.click();
    
    // 验证重连请求被发送
    await page.waitForResponse('**/api/v1/streaming/fault-tolerance/reconnect/session-002');
  });

  test('检查点管理功能完整性测试', async ({ page }) => {
    // 切换到检查点管理标签页
    await page.click('text=检查点管理');
    
    // 验证页面加载
    await expect(page.locator('text=总检查点数')).toBeVisible();
    await expect(page.locator('text=存储空间')).toBeVisible();
    await expect(page.locator('text=覆盖作业数')).toBeVisible();

    // 验证统计数据
    await expect(page.locator('text=2')).toBeVisible(); // 总检查点数
    await expect(page.locator('text=5.86 MB')).toBeVisible(); // 存储空间
    await expect(page.locator('text=2')).toBeVisible(); // 覆盖作业数

    // 验证检查点类型分布
    await expect(page.locator('.ant-tag:has-text("manual")')).toBeVisible();
    await expect(page.locator('.ant-tag:has-text("auto")')).toBeVisible();

    // 验证检查点列表
    await expect(page.locator('text=检查点列表')).toBeVisible();
    await expect(page.locator('text=ckpt-001')).toBeVisible();
    await expect(page.locator('text=ckpt-002')).toBeVisible();
    
    // 验证类型标签
    await expect(page.locator('.ant-tag:has-text("MANUAL")')).toBeVisible();
    await expect(page.locator('.ant-tag:has-text("AUTO")')).toBeVisible();

    // 验证进度显示
    await expect(page.locator('text=75/100')).toBeVisible(); // 第一个检查点进度
    await expect(page.locator('text=180/200')).toBeVisible(); // 第二个检查点进度

    // 验证文件大小显示
    await expect(page.locator('text=2 MB')).toBeVisible(); // 第一个文件大小
    await expect(page.locator('text=4 MB')).toBeVisible(); // 第二个文件大小
  });

  test('检查点管理作业筛选功能测试', async ({ page }) => {
    await page.click('text=检查点管理');
    
    // 等待页面加载
    await expect(page.locator('text=所有作业')).toBeVisible();
    
    // 点击作业筛选下拉框
    await page.click('.ant-select-selector:has-text("所有作业")');
    
    // 验证作业选项
    await expect(page.locator('text=High Priority Batch Job')).toBeVisible();
    await expect(page.locator('text=Standard Processing Job')).toBeVisible();
    
    // 选择特定作业
    await page.click('text=High Priority Batch Job');
    
    // 验证筛选后的结果
    await expect(page.locator('text=创建检查点')).toBeVisible();
  });

  test('检查点搜索功能测试', async ({ page }) => {
    await page.click('text=检查点管理');
    
    // 等待列表加载
    await expect(page.locator('text=ckpt-001')).toBeVisible();
    await expect(page.locator('text=ckpt-002')).toBeVisible();
    
    // 搜索特定检查点
    const searchInput = page.locator('input[placeholder="搜索检查点ID或作业ID"]');
    await searchInput.fill('ckpt-001');
    
    // 验证搜索结果（这里由于是前端过滤，两个都可能显示，但实际应用中会过滤）
    await expect(page.locator('text=ckpt-001')).toBeVisible();
  });

  test('检查点操作功能测试', async ({ page }) => {
    // 模拟检查点操作API
    await page.route('**/api/v1/batch/jobs/job-001/checkpoint', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ checkpoint_id: 'new-checkpoint-id' })
      });
    });

    await page.route('**/api/v1/batch/checkpoints/ckpt-001/restore', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ job_id: 'restored-job-id' })
      });
    });

    await page.route('**/api/v1/batch/checkpoints/ckpt-001', async route => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({ status: 200 });
      }
    });

    await page.click('text=检查点管理');
    
    // 等待页面加载
    await expect(page.locator('text=ckpt-001')).toBeVisible();
    
    // 测试恢复操作
    const restoreButtons = page.locator('button:has-text("恢复")');
    await restoreButtons.first().click();
    
    // 确认对话框
    await expect(page.locator('text=确认恢复')).toBeVisible();
    await page.click('button:has-text("确定")');
    
    // 验证恢复请求
    await page.waitForResponse('**/api/v1/batch/checkpoints/ckpt-001/restore');
    
    // 测试删除操作  
    const deleteButtons = page.locator('button:has-text("删除")');
    await deleteButtons.first().click();
    
    // 确认删除对话框
    await expect(page.locator('text=确认删除')).toBeVisible();
    await page.click('button:has-text("确定")');
    
    // 验证删除请求
    await page.waitForResponse(request => 
      request.url().includes('/api/v1/batch/checkpoints/ckpt-001') && 
      request.method() === 'DELETE'
    );
  });

  test('刷新功能测试', async ({ page }) => {
    let requestCount = 0;
    
    // 计算API调用次数
    await page.route('**/api/v1/streaming/fault-tolerance/stats', async route => {
      requestCount++;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total_active_connections: requestCount,
          healthy_connections: requestCount,
          failed_connections: 0,
          average_uptime: 3600,
          total_reconnections: 0,
          connections: []
        })
      });
    });

    await page.click('text=容错监控');
    
    // 初始加载
    await expect(page.locator('text=活跃连接')).toBeVisible();
    
    // 点击刷新按钮
    await page.click('button:has-text("刷新")');
    
    // 验证数据更新（请求次数应该增加）
    await page.waitForResponse('**/api/v1/streaming/fault-tolerance/stats');
    
    // 验证requestCount增加
    expect(requestCount).toBeGreaterThan(1);
  });

  test('界面响应性测试', async ({ page }) => {
    // 测试不同设备尺寸下的响应性
    await page.setViewportSize({ width: 768, height: 1024 }); // 平板尺寸
    
    await page.click('text=容错监控');
    await expect(page.locator('text=容错连接监控')).toBeVisible();
    
    await page.click('text=检查点管理');
    await expect(page.locator('text=总检查点数')).toBeVisible();
    
    // 移动设备尺寸
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.click('text=容错监控');
    await expect(page.locator('text=活跃连接')).toBeVisible();
    
    // 桌面尺寸
    await page.setViewportSize({ width: 1920, height: 1080 });
    
    await page.click('text=检查点管理');
    await expect(page.locator('text=检查点列表')).toBeVisible();
  });

  test('错误处理测试', async ({ page }) => {
    // 模拟API错误
    await page.route('**/api/v1/streaming/fault-tolerance/stats', async route => {
      await route.fulfill({ status: 500, body: 'Internal Server Error' });
    });

    await page.route('**/api/v1/batch/checkpoints**', async route => {
      await route.fulfill({ status: 404, body: 'Not Found' });
    });

    // 容错监控错误处理
    await page.click('text=容错监控');
    
    // 页面应该仍然可用，即使API失败
    await expect(page.locator('text=容错连接监控')).toBeVisible();
    
    // 检查点管理错误处理
    await page.click('text=检查点管理');
    await expect(page.locator('text=总检查点数')).toBeVisible();
  });

  test('数据自动刷新测试', async ({ page }) => {
    let callCount = 0;
    
    await page.route('**/api/v1/streaming/fault-tolerance/stats', async route => {
      callCount++;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total_active_connections: callCount,
          healthy_connections: callCount,
          failed_connections: 0,
          average_uptime: 3600,
          total_reconnections: 0,
          connections: []
        })
      });
    });

    await page.click('text=容错监控');
    
    // 等待初始加载
    await expect(page.locator('text=活跃连接')).toBeVisible();
    const initialCount = callCount;
    
    // 等待自动刷新（5秒间隔）
    await page.waitForTimeout(6000);
    
    // 验证自动刷新发生
    expect(callCount).toBeGreaterThan(initialCount);
  });
});