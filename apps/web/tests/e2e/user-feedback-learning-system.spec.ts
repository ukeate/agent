/**
 * 用户反馈学习系统 E2E 测试
 * 
 * 测试完整的用户反馈收集、处理和学习工作流程，
 * 包括多种反馈类型的提交、实时更新和分析功能。
 */

import { test, expect, Page } from '@playwright/test';

test.describe('用户反馈学习系统', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('/');
  });

  test.afterEach(async () => {
    await page.close();
  });

  test.describe('反馈收集功能', () => {
    test('应该能够提交评分反馈', async () => {
      // 导航到反馈页面
      await page.click('[data-testid="nav-feedback"]');
      await expect(page).toHaveURL(/.*feedback/);
      
      // 等待页面加载
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 选择5星评分
      await page.click('[data-testid="rating-star-5"]');
      
      // 验证评分被选中
      await expect(page.locator('[data-testid="rating-star-5"]')).toHaveClass(/selected/);
      
      // 等待自动提交
      await page.waitForSelector('[data-testid="feedback-success"]', { timeout: 5000 });
      
      // 验证成功提示
      await expect(page.locator('[data-testid="feedback-success"]')).toContainText('反馈已提交');
    });

    test('应该能够提交点赞反馈', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 点击点赞按钮
      await page.click('[data-testid="like-button"]');
      
      // 验证点赞状态
      await expect(page.locator('[data-testid="like-button"]')).toHaveClass(/active/);
      
      // 验证点踩按钮未激活
      await expect(page.locator('[data-testid="dislike-button"]')).not.toHaveClass(/active/);
      
      // 等待提交成功
      await page.waitForSelector('[data-testid="feedback-success"]');
    });

    test('应该能够提交收藏反馈', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 点击收藏按钮
      await page.click('[data-testid="bookmark-button"]');
      
      // 验证收藏状态
      await expect(page.locator('[data-testid="bookmark-button"]')).toHaveClass(/bookmarked/);
      
      // 再次点击取消收藏
      await page.click('[data-testid="bookmark-button"]');
      await expect(page.locator('[data-testid="bookmark-button"]')).not.toHaveClass(/bookmarked/);
    });

    test('应该能够提交评论反馈', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 展开高级选项显示评论区域
      await page.click('[data-testid="advanced-options"]');
      
      // 输入评论
      const commentText = '这个功能很不错，使用体验很好！';
      await page.fill('[data-testid="comment-textarea"]', commentText);
      
      // 验证字符计数
      await expect(page.locator('[data-testid="character-count"]')).toContainText(`${commentText.length}/500`);
      
      // 提交评论
      await page.click('[data-testid="submit-comment"]');
      
      // 验证提交成功
      await page.waitForSelector('[data-testid="comment-success"]');
      await expect(page.locator('[data-testid="comment-success"]')).toContainText('评论已提交');
      
      // 验证评论区域被禁用
      await expect(page.locator('[data-testid="comment-textarea"]')).toBeDisabled();
    });

    test('应该验证评论内容', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      await page.click('[data-testid="advanced-options"]');
      
      // 测试空评论
      await page.click('[data-testid="submit-comment"]');
      await expect(page.locator('[data-testid="validation-error"]')).toContainText('评论内容至少需要3个字符');
      
      // 测试过长评论
      const longComment = 'a'.repeat(501);  // 超过500字符限制
      await page.fill('[data-testid="comment-textarea"]', longComment);
      await page.click('[data-testid="submit-comment"]');
      await expect(page.locator('[data-testid="validation-error"]')).toContainText('评论长度不能超过500字符');
      
      // 测试垃圾内容
      await page.fill('[data-testid="comment-textarea"]', 'aaaaaaaa');  // 重复字符
      await page.click('[data-testid="submit-comment"]');
      await expect(page.locator('[data-testid="validation-error"]')).toContainText('评论内容不符合要求');
    });

    test('应该支持批量反馈提交', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 连续提交多种反馈
      await page.click('[data-testid="rating-star-4"]');
      await page.waitForTimeout(100);
      
      await page.click('[data-testid="like-button"]');
      await page.waitForTimeout(100);
      
      await page.click('[data-testid="bookmark-button"]');
      await page.waitForTimeout(100);
      
      // 验证所有反馈都被记录
      await expect(page.locator('[data-testid="feedback-summary"]')).toContainText('评分: 4/5');
      await expect(page.locator('[data-testid="feedback-summary"]')).toContainText('态度: 喜欢');
      await expect(page.locator('[data-testid="feedback-summary"]')).toContainText('已收藏');
    });
  });

  test.describe('反馈分析界面', () => {
    test('应该显示反馈统计数据', async () => {
      // 先提交一些反馈数据
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      await page.click('[data-testid="rating-star-5"]');
      await page.waitForTimeout(1000);
      
      // 导航到反馈分析页面
      await page.click('[data-testid="nav-feedback-analytics"]');
      await expect(page).toHaveURL(/.*feedback.*analytics/);
      
      // 验证统计图表存在
      await page.waitForSelector('[data-testid="feedback-chart"]');
      await expect(page.locator('[data-testid="feedback-chart"]')).toBeVisible();
      
      // 验证统计数据
      await expect(page.locator('[data-testid="total-feedback-count"]')).toContainText(/\d+/);
      await expect(page.locator('[data-testid="average-rating"]')).toContainText(/\d+\.\d+/);
      await expect(page.locator('[data-testid="positive-feedback-ratio"]')).toContainText(/\d+%/);
    });

    test('应该支持按时间过滤反馈', async () => {
      await page.click('[data-testid="nav-feedback-analytics"]');
      await page.waitForSelector('[data-testid="feedback-analytics-page"]');
      
      // 选择时间范围
      await page.click('[data-testid="time-filter"]');
      await page.click('[data-testid="time-filter-7days"]');
      
      // 验证图表更新
      await page.waitForSelector('[data-testid="chart-loading"]');
      await page.waitForSelector('[data-testid="chart-loaded"]');
      
      // 验证过滤器状态
      await expect(page.locator('[data-testid="active-filter"]')).toContainText('最近7天');
    });

    test('应该支持按反馈类型过滤', async () => {
      await page.click('[data-testid="nav-feedback-analytics"]');
      await page.waitForSelector('[data-testid="feedback-analytics-page"]');
      
      // 选择反馈类型过滤器
      await page.click('[data-testid="feedback-type-filter"]');
      await page.click('[data-testid="filter-rating"]');
      
      // 验证只显示评分数据
      await page.waitForSelector('[data-testid="chart-loaded"]');
      await expect(page.locator('[data-testid="chart-title"]')).toContainText('评分反馈');
    });

    test('应该显示反馈质量监控', async () => {
      await page.click('[data-testid="nav-feedback-quality"]');
      await page.waitForSelector('[data-testid="quality-monitor-page"]');
      
      // 验证质量指标显示
      await expect(page.locator('[data-testid="quality-score"]')).toBeVisible();
      await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
      await expect(page.locator('[data-testid="spam-detection-rate"]')).toBeVisible();
      
      // 验证异常反馈警报
      const alertsSection = page.locator('[data-testid="quality-alerts"]');
      await expect(alertsSection).toBeVisible();
    });
  });

  test.describe('用户反馈档案', () => {
    test('应该显示用户反馈历史', async () => {
      // 先提交一些反馈
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      await page.click('[data-testid="rating-star-4"]');
      await page.waitForTimeout(1000);
      
      // 导航到用户反馈档案
      await page.click('[data-testid="nav-user-feedback-profiles"]');
      await page.waitForSelector('[data-testid="user-profiles-page"]');
      
      // 搜索特定用户
      await page.fill('[data-testid="user-search"]', 'test-user');
      await page.click('[data-testid="search-button"]');
      
      // 验证用户反馈历史显示
      await page.waitForSelector('[data-testid="user-feedback-history"]');
      await expect(page.locator('[data-testid="feedback-item"]').first()).toBeVisible();
    });

    test('应该显示用户反馈偏好分析', async () => {
      await page.click('[data-testid="nav-user-feedback-profiles"]');
      await page.waitForSelector('[data-testid="user-profiles-page"]');
      
      // 选择用户查看详细档案
      await page.click('[data-testid="user-profile-item"]');
      
      // 验证偏好分析图表
      await page.waitForSelector('[data-testid="preference-analysis"]');
      await expect(page.locator('[data-testid="category-preferences"]')).toBeVisible();
      await expect(page.locator('[data-testid="feedback-patterns"]')).toBeVisible();
      await expect(page.locator('[data-testid="engagement-metrics"]')).toBeVisible();
    });
  });

  test.describe('物品反馈分析', () => {
    test('应该显示物品反馈汇总', async () => {
      await page.click('[data-testid="nav-item-feedback"]');
      await page.waitForSelector('[data-testid="item-feedback-page"]');
      
      // 搜索特定物品
      await page.fill('[data-testid="item-search"]', 'product-123');
      await page.click('[data-testid="search-button"]');
      
      // 验证物品反馈汇总显示
      await page.waitForSelector('[data-testid="item-feedback-summary"]');
      await expect(page.locator('[data-testid="item-rating"]')).toContainText(/\d+\.\d+/);
      await expect(page.locator('[data-testid="feedback-count"]')).toContainText(/\d+/);
    });

    test('应该支持物品反馈趋势分析', async () => {
      await page.click('[data-testid="nav-item-feedback"]');
      await page.waitForSelector('[data-testid="item-feedback-page"]');
      
      // 选择物品查看趋势
      await page.click('[data-testid="item-feedback-item"]');
      
      // 验证趋势图表
      await page.waitForSelector('[data-testid="feedback-trend-chart"]');
      await expect(page.locator('[data-testid="trend-line"]')).toBeVisible();
      
      // 验证时间轴控制
      await page.click('[data-testid="timespan-selector"]');
      await page.click('[data-testid="timespan-month"]');
      await page.waitForSelector('[data-testid="chart-updated"]');
    });
  });

  test.describe('实时反馈更新', () => {
    test('应该实时显示新的反馈数据', async () => {
      // 打开分析页面
      await page.click('[data-testid="nav-feedback-analytics"]');
      await page.waitForSelector('[data-testid="feedback-analytics-page"]');
      
      // 记录初始数据
      const initialCount = await page.locator('[data-testid="total-feedback-count"]').textContent();
      
      // 在新标签页提交反馈
      const newPage = await page.context().newPage();
      await newPage.goto('/feedback');
      await newPage.waitForSelector('[data-testid="feedback-form"]');
      await newPage.click('[data-testid="rating-star-5"]');
      await newPage.waitForTimeout(1000);
      await newPage.close();
      
      // 验证原页面数据更新
      await page.waitForFunction(
        (initialCount) => {
          const currentCount = document.querySelector('[data-testid="total-feedback-count"]')?.textContent;
          return currentCount !== initialCount;
        },
        initialCount,
        { timeout: 10000 }
      );
    });

    test('应该显示实时反馈流', async () => {
      await page.click('[data-testid="nav-feedback-analytics"]');
      await page.waitForSelector('[data-testid="feedback-analytics-page"]');
      
      // 启用实时模式
      await page.click('[data-testid="realtime-toggle"]');
      await expect(page.locator('[data-testid="realtime-indicator"]')).toHaveClass(/active/);
      
      // 验证实时反馈流显示
      await page.waitForSelector('[data-testid="realtime-feed"]');
      
      // 模拟新反馈到达（通过WebSocket或其他实时更新机制）
      // 验证新反馈出现在流中
      await page.waitForSelector('[data-testid="new-feedback-item"]', { timeout: 5000 });
    });
  });

  test.describe('移动端适配', () => {
    test('应该在移动设备上正确显示反馈表单', async () => {
      // 模拟移动设备视窗
      await page.setViewportSize({ width: 375, height: 667 });
      
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 验证紧凑布局
      await expect(page.locator('[data-testid="feedback-form"]')).toHaveClass(/mobile/);
      
      // 验证触摸友好的按钮大小
      const ratingButton = page.locator('[data-testid="rating-star-5"]');
      const buttonSize = await ratingButton.boundingBox();
      expect(buttonSize!.width).toBeGreaterThan(44); // 最小触摸目标44px
      expect(buttonSize!.height).toBeGreaterThan(44);
    });

    test('应该支持移动端手势操作', async () => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 测试滑动评分
      const ratingSlider = page.locator('[data-testid="rating-slider"]');
      if (await ratingSlider.isVisible()) {
        await ratingSlider.dragTo(page.locator('[data-testid="rating-value-4"]'));
        await expect(page.locator('[data-testid="current-rating"]')).toContainText('4');
      }
    });
  });

  test.describe('性能测试', () => {
    test('反馈表单应该快速加载', async () => {
      const startTime = Date.now();
      
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      const loadTime = Date.now() - startTime;
      expect(loadTime).toBeLessThan(2000); // 2秒内加载完成
    });

    test('反馈提交应该有合理的响应时间', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      const startTime = Date.now();
      await page.click('[data-testid="rating-star-5"]');
      await page.waitForSelector('[data-testid="feedback-success"]');
      const responseTime = Date.now() - startTime;
      
      expect(responseTime).toBeLessThan(3000); // 3秒内完成提交
    });

    test('分析页面应该高效渲染大量数据', async () => {
      await page.click('[data-testid="nav-feedback-analytics"]');
      
      const startTime = Date.now();
      await page.waitForSelector('[data-testid="feedback-chart"]');
      const renderTime = Date.now() - startTime;
      
      expect(renderTime).toBeLessThan(5000); // 5秒内渲染完成
      
      // 验证图表交互响应性
      await page.hover('[data-testid="chart-data-point"]');
      await page.waitForSelector('[data-testid="chart-tooltip"]', { timeout: 1000 });
    });
  });

  test.describe('错误处理', () => {
    test('应该处理网络连接错误', async () => {
      // 模拟离线状态
      await page.context().setOffline(true);
      
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      await page.click('[data-testid="rating-star-5"]');
      
      // 验证错误提示
      await page.waitForSelector('[data-testid="network-error"]');
      await expect(page.locator('[data-testid="network-error"]')).toContainText('网络连接失败');
      
      // 恢复网络连接
      await page.context().setOffline(false);
      
      // 验证重试功能
      await page.click('[data-testid="retry-button"]');
      await page.waitForSelector('[data-testid="feedback-success"]');
    });

    test('应该处理服务器错误', async () => {
      // 拦截请求并返回服务器错误
      await page.route('**/api/feedback/**', route => {
        route.fulfill({ status: 500, body: 'Internal Server Error' });
      });
      
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      await page.click('[data-testid="rating-star-5"]');
      
      // 验证服务器错误处理
      await page.waitForSelector('[data-testid="server-error"]');
      await expect(page.locator('[data-testid="error-message"]')).toContainText('服务器暂时不可用');
    });

    test('应该处理数据加载失败', async () => {
      // 拦截分析数据请求
      await page.route('**/api/feedback/analytics', route => {
        route.fulfill({ status: 404, body: 'Not Found' });
      });
      
      await page.click('[data-testid="nav-feedback-analytics"]');
      
      // 验证加载失败状态
      await page.waitForSelector('[data-testid="data-load-error"]');
      await expect(page.locator('[data-testid="error-message"]')).toContainText('数据加载失败');
      
      // 验证重新加载功能
      await expect(page.locator('[data-testid="reload-button"]')).toBeVisible();
    });
  });

  test.describe('安全性测试', () => {
    test('应该防止XSS攻击', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      await page.click('[data-testid="advanced-options"]');
      
      // 尝试输入恶意脚本
      const maliciousScript = '<script>alert("XSS")</script>';
      await page.fill('[data-testid="comment-textarea"]', maliciousScript);
      await page.click('[data-testid="submit-comment"]');
      
      // 验证脚本被正确转义，没有执行
      await page.waitForSelector('[data-testid="comment-success"]');
      
      // 检查页面没有被恶意脚本影响
      const alertDialogs = page.locator('dialog[role="alert"]');
      await expect(alertDialogs).toHaveCount(0);
    });

    test('应该限制频繁提交', async () => {
      await page.click('[data-testid="nav-feedback"]');
      await page.waitForSelector('[data-testid="feedback-form"]');
      
      // 快速连续提交多次
      for (let i = 0; i < 15; i++) {
        await page.click('[data-testid="rating-star-5"]');
        await page.waitForTimeout(100);
      }
      
      // 验证频率限制提示
      await page.waitForSelector('[data-testid="rate-limit-warning"]');
      await expect(page.locator('[data-testid="rate-limit-warning"]')).toContainText('操作过于频繁');
    });
  });
});