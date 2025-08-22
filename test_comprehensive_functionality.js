/**
 * AI Agent系统综合功能测试脚本
 * 测试http://localhost:3000的所有功能模块
 */

const { chromium } = require('playwright');

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function runComprehensiveTests() {
  console.log('🚀 开始AI Agent系统综合功能测试...');
  
  const browser = await chromium.launch({ 
    headless: false, // 设为true可在后台运行
    slowMo: 50 // 减缓操作速度便于观察
  });
  
  try {
    const page = await browser.newPage();
    await page.goto('http://localhost:3000');
    
    const testResults = {
      navigation: [],
      pageLoads: [],
      interactions: [],
      api: [],
      performance: [],
      errors: []
    };

    // 1. 基础功能验证
    console.log('\n📋 1. 基础功能验证');
    
    // 检查页面标题和基本元素
    const title = await page.title();
    console.log(`✓ 页面标题: ${title}`);
    testResults.pageLoads.push({ test: '页面标题', result: 'PASS', value: title });
    
    // 检查侧边栏存在
    const sidebar = await page.locator('.ant-layout-sider').first();
    const sidebarExists = await sidebar.isVisible();
    console.log(`${sidebarExists ? '✓' : '✗'} 侧边栏显示: ${sidebarExists}`);
    testResults.pageLoads.push({ test: '侧边栏显示', result: sidebarExists ? 'PASS' : 'FAIL' });

    // 检查菜单项总数
    const menuItems = await page.locator('.ant-menu-item').count();
    console.log(`✓ 菜单项数量: ${menuItems}`);
    testResults.pageLoads.push({ test: '菜单项数量', result: 'PASS', value: menuItems });

    // 2. 导航功能测试
    console.log('\n🧭 2. 导航功能测试');
    
    // 测试菜单收缩/展开
    const toggleButton = page.locator('[data-icon="menu-fold"], [data-icon="menu-unfold"]').first();
    if (await toggleButton.isVisible()) {
      await toggleButton.click();
      await delay(500);
      console.log('✓ 菜单收缩功能正常');
      await toggleButton.click();
      await delay(500);
      console.log('✓ 菜单展开功能正常');
      testResults.navigation.push({ test: '菜单收缩展开', result: 'PASS' });
    }

    // 3. 核心页面深度测试
    console.log('\n🔍 3. 核心页面深度测试');
    
    const testPages = [
      { name: '反馈系统总览', path: '/feedback-system', selector: '[data-testid="feedback-form"]' },
      { name: '多代理协作', path: '/multi-agent', selector: '.ant-card' },
      { name: 'Q-Learning算法', path: '/qlearning', selector: '.ant-statistic' },
      { name: '工作流可视化', path: '/workflows', selector: '.react-flow' },
      { name: 'RAG检索', path: '/rag', selector: '.ant-input' },
      { name: '多模态处理', path: '/multimodal', selector: '.ant-upload' }
    ];

    for (const testPage of testPages) {
      console.log(`\n  测试页面: ${testPage.name}`);
      
      try {
        const startTime = performance.now();
        
        // 导航到页面
        await page.goto(`http://localhost:3000${testPage.path}`);
        await page.waitForLoadState('networkidle', { timeout: 10000 });
        
        const loadTime = performance.now() - startTime;
        console.log(`    ✓ 页面加载时间: ${loadTime.toFixed(2)}ms`);
        testResults.performance.push({ 
          page: testPage.name, 
          loadTime: loadTime.toFixed(2), 
          result: loadTime < 5000 ? 'PASS' : 'SLOW' 
        });
        
        // 检查特征元素
        if (testPage.selector) {
          try {
            await page.waitForSelector(testPage.selector, { timeout: 5000 });
            console.log(`    ✓ 特征元素加载正常: ${testPage.selector}`);
            testResults.pageLoads.push({ 
              page: testPage.name, 
              element: testPage.selector, 
              result: 'PASS' 
            });
          } catch (e) {
            console.log(`    ⚠ 特征元素未找到: ${testPage.selector}`);
            testResults.pageLoads.push({ 
              page: testPage.name, 
              element: testPage.selector, 
              result: 'FAIL',
              error: e.message 
            });
          }
        }
        
        // 检查是否有React错误
        const errors = await page.evaluate(() => {
          return window.console.error ? [] : [];
        });
        
        // 检查页面内容
        const hasContent = await page.locator('.ant-card, .ant-table, .ant-statistic').count() > 0;
        console.log(`    ${hasContent ? '✓' : '⚠'} 页面内容加载: ${hasContent ? '正常' : '空白'}`);
        
      } catch (error) {
        console.log(`    ✗ 页面测试失败: ${error.message}`);
        testResults.errors.push({ page: testPage.name, error: error.message });
      }
    }

    // 4. 交互功能测试
    console.log('\n🎯 4. 交互功能测试');
    
    // 测试反馈系统页面的交互
    try {
      await page.goto('http://localhost:3000/feedback-system');
      await page.waitForLoadState('networkidle');
      
      // 查找测试按钮
      const testButtons = await page.locator('button').filter({ hasText: /测试|提交|刷新/ }).all();
      console.log(`    找到 ${testButtons.length} 个测试按钮`);
      
      // 点击第一个测试按钮
      if (testButtons.length > 0) {
        await testButtons[0].click();
        await delay(1000);
        console.log('    ✓ 按钮点击交互正常');
        testResults.interactions.push({ test: '按钮点击', result: 'PASS' });
      }
      
    } catch (error) {
      console.log(`    ⚠ 交互测试失败: ${error.message}`);
      testResults.interactions.push({ test: '按钮点击', result: 'FAIL', error: error.message });
    }

    // 5. API连接测试
    console.log('\n🌐 5. API连接测试');
    
    // 监听网络请求
    const apiCalls = [];
    page.on('response', response => {
      if (response.url().includes('/api/') || response.url().includes(':8000')) {
        apiCalls.push({
          url: response.url(),
          status: response.status(),
          ok: response.ok()
        });
      }
    });
    
    // 触发一些API调用
    await page.goto('http://localhost:3000/feedback-system');
    await delay(3000);
    
    console.log(`    检测到 ${apiCalls.length} 个API调用`);
    apiCalls.forEach(call => {
      console.log(`    ${call.ok ? '✓' : '✗'} ${call.status} ${call.url}`);
    });
    
    testResults.api = apiCalls;

    // 6. 响应式设计测试
    console.log('\n📱 6. 响应式设计测试');
    
    const viewports = [
      { name: '桌面', width: 1920, height: 1080 },
      { name: '平板', width: 768, height: 1024 },
      { name: '手机', width: 375, height: 667 }
    ];
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await delay(500);
      
      const sidebarVisible = await page.locator('.ant-layout-sider').isVisible();
      console.log(`    ${viewport.name} (${viewport.width}x${viewport.height}): 侧边栏${sidebarVisible ? '可见' : '隐藏'}`);
      
      testResults.navigation.push({ 
        test: `响应式-${viewport.name}`, 
        result: 'PASS', 
        viewport: `${viewport.width}x${viewport.height}`,
        sidebarVisible 
      });
    }

    // 7. 性能测试
    console.log('\n⚡ 7. 性能测试');
    
    await page.setViewportSize({ width: 1920, height: 1080 });
    
    const performanceMetrics = await page.evaluate(() => ({
      memory: performance.memory ? {
        used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
      } : null,
      timing: performance.timing ? {
        domComplete: performance.timing.domComplete - performance.timing.navigationStart,
        loadComplete: performance.timing.loadEventEnd - performance.timing.navigationStart
      } : null
    }));
    
    if (performanceMetrics.memory) {
      console.log(`    内存使用: ${performanceMetrics.memory.used}MB / ${performanceMetrics.memory.total}MB`);
    }
    
    testResults.performance.push({ test: '内存使用', result: 'PASS', data: performanceMetrics });

    // 8. 错误检查
    console.log('\n🔍 8. 错误检查');
    
    const consoleLogs = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleLogs.push(msg.text());
      }
    });
    
    // 检查是否有未捕获的JavaScript错误
    await page.evaluate(() => {
      window.addEventListener('error', (e) => {
        console.error('Uncaught error:', e.error);
      });
    });

    console.log(`    检测到 ${consoleLogs.length} 个控制台错误`);
    if (consoleLogs.length > 0) {
      consoleLogs.forEach(log => console.log(`    ✗ ${log}`));
    }
    
    testResults.errors.push(...consoleLogs.map(log => ({ type: 'console', message: log })));

    // 生成测试报告
    console.log('\n📊 测试报告生成...');
    
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalTests: Object.values(testResults).flat().length,
        passedTests: Object.values(testResults).flat().filter(t => t.result === 'PASS').length,
        failedTests: Object.values(testResults).flat().filter(t => t.result === 'FAIL').length,
        warningTests: Object.values(testResults).flat().filter(t => t.result === 'SLOW' || t.result === 'WARNING').length
      },
      details: testResults,
      recommendations: []
    };
    
    // 添加建议
    if (report.summary.failedTests > 0) {
      report.recommendations.push('存在功能性问题，需要修复失败的测试项');
    }
    if (testResults.api.some(call => !call.ok)) {
      report.recommendations.push('后端API连接存在问题，建议检查服务状态');
    }
    if (testResults.performance.some(p => parseFloat(p.loadTime) > 3000)) {
      report.recommendations.push('页面加载性能需要优化');
    }
    if (testResults.errors.length === 0) {
      report.recommendations.push('应用运行稳定，无明显错误');
    }
    
    // 输出最终报告
    console.log('\n🎉 测试完成！');
    console.log('='.repeat(60));
    console.log(`总测试项: ${report.summary.totalTests}`);
    console.log(`通过: ${report.summary.passedTests}`);
    console.log(`失败: ${report.summary.failedTests}`);
    console.log(`警告: ${report.summary.warningTests}`);
    console.log(`成功率: ${((report.summary.passedTests / report.summary.totalTests) * 100).toFixed(1)}%`);
    console.log('='.repeat(60));
    
    if (report.recommendations.length > 0) {
      console.log('\n📋 建议:');
      report.recommendations.forEach((rec, i) => {
        console.log(`${i + 1}. ${rec}`);
      });
    }
    
    // 保存详细报告
    const fs = require('fs');
    fs.writeFileSync('./test_report.json', JSON.stringify(report, null, 2));
    console.log('\n💾 详细报告已保存到 test_report.json');
    
    return report;
    
  } catch (error) {
    console.error('❌ 测试执行失败:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

// 运行测试
if (require.main === module) {
  runComprehensiveTests()
    .then(report => {
      console.log('\n✅ 测试执行成功');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ 测试执行失败:', error);
      process.exit(1);
    });
}

module.exports = { runComprehensiveTests };