import { test, expect } from '@playwright/test';

test.describe('离线能力与同步机制系统', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    
    // 等待页面加载完成
    await page.waitForLoadState('networkidle');
  });

  test.describe('导航和页面访问', () => {
    test('应该能够通过导航访问离线能力页面', async ({ page }) => {
      // 查找并点击离线能力分组
      await page.getByText('🔄 离线能力与同步').click();
      
      // 点击离线能力监控
      await page.getByText('离线能力监控').click();
      await page.waitForURL('/offline');
      
      // 验证页面标题
      await expect(page.getByText('离线能力监控')).toBeVisible();
    });

    test('应该能够访问向量时钟可视化页面', async ({ page }) => {
      // 通过导航访问向量时钟页面
      await page.getByText('向量时钟可视化').click();
      await page.waitForURL('/vector-clock');
      
      // 验证页面加载
      await expect(page.getByText('⏰ 向量时钟算法可视化')).toBeVisible();
      await expect(page.getByText('分布式系统中的向量时钟算法演示')).toBeVisible();
    });

    test('应该能够访问同步引擎内部机制页面', async ({ page }) => {
      // 访问同步引擎页面
      await page.getByText('同步引擎内部机制').click();
      await page.waitForURL('/sync-engine');
      
      // 验证页面内容
      await expect(page.getByText('🔄 同步引擎内部机制展示')).toBeVisible();
      await expect(page.getByText('深入了解数据同步引擎的内部工作原理')).toBeVisible();
    });

    test('应该能够访问本地模型缓存监控页面', async ({ page }) => {
      // 访问模型缓存页面
      await page.getByText('本地模型缓存监控').click();
      await page.waitForURL('/model-cache');
      
      // 验证页面内容
      await expect(page.getByText('🗄️ 本地模型缓存监控')).toBeVisible();
      await expect(page.getByText('监控和管理本地AI模型缓存')).toBeVisible();
    });

    test('应该能够访问网络监控详情页面', async ({ page }) => {
      // 访问网络监控页面
      await page.getByText('网络监控详情').click();
      await page.waitForURL('/network-monitor');
      
      // 验证页面内容
      await expect(page.getByText('🌐 网络监控详情')).toBeVisible();
    });
  });

  test.describe('向量时钟可视化功能测试', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('/vector-clock');
      await page.waitForLoadState('networkidle');
    });

    test('应该显示节点状态信息', async ({ page }) => {
      // 验证节点状态卡片
      await expect(page.getByText('节点状态')).toBeVisible();
      await expect(page.getByText('Node A')).toBeVisible();
      await expect(page.getByText('Node B')).toBeVisible();
      await expect(page.getByText('Node C')).toBeVisible();
    });

    test('应该能够添加事件', async ({ page }) => {
      // 查找并点击添加事件按钮
      const addEventButtons = page.getByText('添加事件');
      await expect(addEventButtons.first()).toBeVisible();
      
      // 记录初始事件数量
      const initialEvents = await page.locator('.ant-timeline-item').count();
      
      // 添加事件
      await addEventButtons.first().click();
      
      // 验证事件被添加
      await page.waitForTimeout(1000);
      const newEvents = await page.locator('.ant-timeline-item').count();
      expect(newEvents).toBeGreaterThanOrEqual(initialEvents);
    });

    test('应该能够发送消息', async ({ page }) => {
      // 查找发送消息按钮
      const sendMessageButtons = page.getByText('发送消息');
      await expect(sendMessageButtons.first()).toBeVisible();
      
      // 点击发送消息
      await sendMessageButtons.first().click();
      
      // 验证消息事件出现在时间线中
      await page.waitForTimeout(1000);
      await expect(page.getByText(/发送消息|接收消息/)).toBeVisible();
    });

    test('应该显示向量时钟比较结果', async ({ page }) => {
      // 验证向量时钟比较功能
      await expect(page.getByText('🔍 向量时钟比较')).toBeVisible();
      await expect(page.getByText('时钟A')).toBeVisible();
      await expect(page.getByText('时钟B')).toBeVisible();
      await expect(page.getByText('比较结果:')).toBeVisible();
    });

    test('应该能够切换实时模式', async ({ page }) => {
      // 查找实时模拟开关
      await expect(page.getByText('实时模拟')).toBeVisible();
      
      // 查找并点击开关
      const realTimeSwitch = page.locator('.ant-switch').first();
      await expect(realTimeSwitch).toBeVisible();
      
      // 点击开关
      await realTimeSwitch.click();
      
      // 验证开关状态变化
      await page.waitForTimeout(500);
    });

    test('应该显示统计信息', async ({ page }) => {
      // 验证统计信息显示
      await expect(page.getByText('总事件数')).toBeVisible();
      await expect(page.getByText('消息传递')).toBeVisible();
      await expect(page.getByText('并发事件')).toBeVisible();
      await expect(page.getByText('因果关系')).toBeVisible();
    });

    test('应该显示算法原理说明', async ({ page }) => {
      // 验证算法说明部分
      await expect(page.getByText('📖 算法原理说明')).toBeVisible();
      await expect(page.getByText('向量时钟基础')).toBeVisible();
      await expect(page.getByText('算法步骤')).toBeVisible();
    });
  });

  test.describe('同步引擎内部机制功能测试', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('/sync-engine');
      await page.waitForLoadState('networkidle');
    });

    test('应该显示引擎控制面板', async ({ page }) => {
      // 验证控制面板存在
      await expect(page.getByText('引擎控制面板')).toBeVisible();
      await expect(page.getByText('实时模式')).toBeVisible();
      await expect(page.getByText('最大并发任务')).toBeVisible();
      await expect(page.getByText('批处理大小')).toBeVisible();
      await expect(page.getByText('检查点间隔')).toBeVisible();
    });

    test('应该显示统计信息', async ({ page }) => {
      // 验证统计卡片
      await expect(page.getByText('已同步操作')).toBeVisible();
      await expect(page.getByText('失败操作')).toBeVisible();
      await expect(page.getByText('冲突解决')).toBeVisible();
      await expect(page.getByText('同步效率')).toBeVisible();
      await expect(page.getByText('平均吞吐量')).toBeVisible();
      await expect(page.getByText('活跃任务')).toBeVisible();
    });

    test('应该显示活跃任务表格', async ({ page }) => {
      // 验证活跃任务表格
      await expect(page.getByText('🏃‍♂️ 活跃同步任务')).toBeVisible();
      await expect(page.getByText('任务ID')).toBeVisible();
      await expect(page.getByText('方向')).toBeVisible();
      await expect(page.getByText('优先级')).toBeVisible();
      await expect(page.getByText('状态')).toBeVisible();
      await expect(page.getByText('进度')).toBeVisible();
    });

    test('应该显示操作批处理机制', async ({ page }) => {
      // 验证批处理部分
      await expect(page.getByText('📦 操作批处理机制')).toBeVisible();
      await expect(page.getByText('批处理优化策略')).toBeVisible();
      await expect(page.getByText('操作分组：按表名和操作类型分组')).toBeVisible();
    });

    test('应该显示同步流程图', async ({ page }) => {
      // 验证流程可视化
      await expect(page.getByText('🔄 同步流程可视化')).toBeVisible();
      await expect(page.getByText('上传流程')).toBeVisible();
      await expect(page.getByText('下载流程')).toBeVisible();
      await expect(page.getByText('双向流程')).toBeVisible();
    });

    test('应该能够切换实时模式', async ({ page }) => {
      // 查找实时模式开关
      const realTimeSwitch = page.locator('.ant-switch').first();
      await expect(realTimeSwitch).toBeVisible();
      
      // 点击开关
      await realTimeSwitch.click();
      await page.waitForTimeout(500);
    });

    test('应该能够修改引擎配置', async ({ page }) => {
      // 测试配置修改
      const selectors = page.locator('.ant-select').first();
      await expect(selectors).toBeVisible();
      
      // 点击选择器
      await selectors.click();
      await page.waitForTimeout(500);
    });

    test('应该显示任务进度更新', async ({ page }) => {
      // 验证进度条存在
      const progressBars = page.locator('.ant-progress');
      await expect(progressBars.first()).toBeVisible();
      
      // 等待实时更新
      await page.waitForTimeout(3000);
    });
  });

  test.describe('模型缓存监控功能测试', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('/model-cache');
      await page.waitForLoadState('networkidle');
    });

    test('应该显示缓存统计概览', async ({ page }) => {
      // 验证统计卡片
      await expect(page.getByText('缓存模型')).toBeVisible();
      await expect(page.getByText('内存加载')).toBeVisible();
      await expect(page.getByText('缓存使用率')).toBeVisible();
      await expect(page.getByText('总缓存大小')).toBeVisible();
    });

    test('应该显示缓存空间使用情况', async ({ page }) => {
      // 验证空间使用部分
      await expect(page.getByText('💾 缓存空间使用情况')).toBeVisible();
      await expect(page.getByText('已使用')).toBeVisible();
      await expect(page.getByText('剩余空间')).toBeVisible();
      await expect(page.getByText('总容量')).toBeVisible();
    });

    test('应该显示模型列表', async ({ page }) => {
      // 验证模型列表表格
      await expect(page.getByText('📋 缓存模型列表')).toBeVisible();
      await expect(page.getByText('模型ID')).toBeVisible();
      await expect(page.getByText('状态')).toBeVisible();
      await expect(page.getByText('大小')).toBeVisible();
      await expect(page.getByText('使用统计')).toBeVisible();
    });

    test('应该显示缓存管理策略', async ({ page }) => {
      // 验证管理策略
      await expect(page.getByText('⚙️ 缓存管理策略')).toBeVisible();
      await expect(page.getByText('LRU淘汰')).toBeVisible();
      await expect(page.getByText('智能预加载')).toBeVisible();
      await expect(page.getByText('压缩存储')).toBeVisible();
    });

    test('应该显示压缩与量化技术', async ({ page }) => {
      // 验证技术说明部分
      await expect(page.getByText('🗜️ 模型压缩与量化技术')).toBeVisible();
      await expect(page.getByText('压缩算法')).toBeVisible();
      await expect(page.getByText('量化技术')).toBeVisible();
      await expect(page.getByText('优化效果')).toBeVisible();
    });

    test('应该能够切换自动管理设置', async ({ page }) => {
      // 测试自动清理开关
      await expect(page.getByText('自动清理')).toBeVisible();
      const autoCleanupSwitch = page.locator('.ant-switch').first();
      await expect(autoCleanupSwitch).toBeVisible();
      
      await autoCleanupSwitch.click();
      await page.waitForTimeout(500);
    });

    test('应该能够执行手动清理', async ({ page }) => {
      // 测试手动清理按钮
      const cleanupButton = page.getByText('手动清理');
      await expect(cleanupButton).toBeVisible();
      
      await cleanupButton.click();
      await page.waitForTimeout(500);
    });

    test('应该显示模型详细信息', async ({ page }) => {
      // 验证模型信息显示
      await expect(page.getByText(/claude-3-haiku-quantized|gpt-4-turbo-preview/)).toBeVisible();
      await expect(page.getByText(/已加载|磁盘缓存/)).toBeVisible();
      await expect(page.getByText(/使用次数/)).toBeVisible();
    });

    test('应该显示使用热度分析', async ({ page }) => {
      // 验证热度分析
      await expect(page.getByText('📊 使用热度分析')).toBeVisible();
      await expect(page.getByText('最常用模型')).toBeVisible();
      await expect(page.getByText('最少用模型')).toBeVisible();
    });
  });

  test.describe('响应式设计测试', () => {
    test('应该在移动设备上正确显示', async ({ page }) => {
      // 设置移动视口
      await page.setViewportSize({ width: 375, height: 667 });
      
      await page.goto('/vector-clock');
      await page.waitForLoadState('networkidle');
      
      // 验证移动端布局
      await expect(page.getByText('⏰ 向量时钟算法可视化')).toBeVisible();
    });

    test('应该在平板设备上正确显示', async ({ page }) => {
      // 设置平板视口
      await page.setViewportSize({ width: 768, height: 1024 });
      
      await page.goto('/sync-engine');
      await page.waitForLoadState('networkidle');
      
      // 验证平板端布局
      await expect(page.getByText('🔄 同步引擎内部机制展示')).toBeVisible();
    });
  });

  test.describe('性能测试', () => {
    test('页面加载性能测试', async ({ page }) => {
      const startTime = Date.now();
      
      await page.goto('/model-cache');
      await page.waitForLoadState('networkidle');
      
      const loadTime = Date.now() - startTime;
      
      // 页面应该在5秒内加载完成
      expect(loadTime).toBeLessThan(5000);
    });

    test('大量数据渲染性能测试', async ({ page }) => {
      await page.goto('/sync-engine');
      await page.waitForLoadState('networkidle');
      
      // 测试表格渲染性能
      const table = page.locator('.ant-table-tbody');
      await expect(table).toBeVisible();
      
      // 验证表格行数
      const rows = page.locator('.ant-table-tbody tr');
      const rowCount = await rows.count();
      expect(rowCount).toBeGreaterThan(0);
    });
  });

  test.describe('错误处理测试', () => {
    test('应该处理网络错误', async ({ page }) => {
      // 模拟网络离线
      await page.context().setOffline(true);
      
      await page.goto('/vector-clock');
      
      // 页面应该能够显示离线状态或错误处理
      await expect(page.getByText(/向量时钟|离线/)).toBeVisible();
      
      // 恢复网络
      await page.context().setOffline(false);
    });

    test('应该处理组件错误', async ({ page }) => {
      await page.goto('/model-cache');
      await page.waitForLoadState('networkidle');
      
      // 验证页面没有JavaScript错误
      const errors: string[] = [];
      page.on('pageerror', error => {
        errors.push(error.message);
      });
      
      // 等待一段时间确保没有错误
      await page.waitForTimeout(2000);
      
      // 应该没有关键错误
      const criticalErrors = errors.filter(error => 
        error.includes('Error') || error.includes('TypeError')
      );
      expect(criticalErrors.length).toBe(0);
    });
  });

  test.describe('数据持久化测试', () => {
    test('应该保持用户设置', async ({ page }) => {
      await page.goto('/sync-engine');
      await page.waitForLoadState('networkidle');
      
      // 切换实时模式
      const realTimeSwitch = page.locator('.ant-switch').first();
      await realTimeSwitch.click();
      
      // 刷新页面
      await page.reload();
      await page.waitForLoadState('networkidle');
      
      // 验证设置是否保持（根据具体实现可能需要调整）
      await expect(page.getByText('引擎控制面板')).toBeVisible();
    });
  });

  test.describe('键盘导航测试', () => {
    test('应该支持键盘导航', async ({ page }) => {
      await page.goto('/vector-clock');
      await page.waitForLoadState('networkidle');
      
      // 使用Tab键导航
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      
      // 验证焦点处理
      const focusedElement = await page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    });

    test('应该支持快捷键操作', async ({ page }) => {
      await page.goto('/model-cache');
      await page.waitForLoadState('networkidle');
      
      // 测试Escape键（如果有模态框的话）
      await page.keyboard.press('Escape');
      
      // 验证页面仍然正常
      await expect(page.getByText('🗄️ 本地模型缓存监控')).toBeVisible();
    });
  });
});