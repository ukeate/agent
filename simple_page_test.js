/**
 * 简单页面检查脚本
 */

const { chromium } = require('playwright');

async function simplePageTest() {
  console.log('🔍 开始简单页面检查...');
  
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 1000 
  });
  
  try {
    const page = await browser.newPage();
    
    // 监听控制台输出
    page.on('console', msg => {
      console.log(`[浏览器控制台 ${msg.type()}] ${msg.text()}`);
    });
    
    // 监听页面错误
    page.on('pageerror', error => {
      console.error('❌ 页面错误:', error.message);
    });
    
    console.log('📄 访问主页...');
    await page.goto('http://localhost:3000');
    
    // 等待页面加载
    await page.waitForTimeout(3000);
    
    // 获取页面HTML
    const html = await page.content();
    console.log('📝 页面HTML长度:', html.length);
    
    // 检查是否有root元素
    const rootElement = await page.$('#root');
    if (rootElement) {
      const rootHTML = await rootElement.innerHTML();
      console.log('🎯 Root元素内容长度:', rootHTML.length);
      
      if (rootHTML.trim() === '') {
        console.log('⚠ Root元素为空，React应用可能未正确加载');
      } else {
        console.log('✓ Root元素有内容');
        // 输出前500个字符
        console.log('内容预览:', rootHTML.substring(0, 500) + '...');
      }
    }
    
    // 检查网络请求
    const requests = [];
    page.on('request', request => {
      requests.push({
        url: request.url(),
        method: request.method(),
        resourceType: request.resourceType()
      });
    });
    
    // 检查响应
    const responses = [];
    page.on('response', response => {
      responses.push({
        url: response.url(),
        status: response.status(),
        ok: response.ok()
      });
    });
    
    // 重新加载页面以捕获请求
    console.log('🔄 重新加载页面以分析网络请求...');
    await page.reload();
    await page.waitForTimeout(5000);
    
    console.log(`\n📊 网络请求分析:`);
    console.log(`总请求数: ${requests.length}`);
    
    const jsRequests = requests.filter(r => r.resourceType === 'script');
    console.log(`JavaScript请求: ${jsRequests.length}`);
    jsRequests.forEach(req => {
      console.log(`  - ${req.url}`);
    });
    
    const cssRequests = requests.filter(r => r.resourceType === 'stylesheet');
    console.log(`CSS请求: ${cssRequests.length}`);
    
    const failedResponses = responses.filter(r => !r.ok);
    console.log(`失败的请求: ${failedResponses.length}`);
    failedResponses.forEach(resp => {
      console.log(`  ❌ ${resp.status} ${resp.url}`);
    });
    
    // 截图
    await page.screenshot({ path: './page_screenshot.png', fullPage: true });
    console.log('📸 页面截图已保存到 page_screenshot.png');
    
    await page.waitForTimeout(5000);
    
  } catch (error) {
    console.error('❌ 测试失败:', error);
  } finally {
    await browser.close();
  }
}

// 运行测试
simplePageTest();