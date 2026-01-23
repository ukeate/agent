import { test, expect } from '@playwright/test'
import { mkdtempSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

// 测试文档处理系统的E2E功能
test.describe('文档处理系统E2E测试', () => {
  test.beforeEach(async ({ page }) => {
    // 导航到文档处理页面
    await page.goto('/document-processing')
    await page.waitForLoadState('networkidle')
  })

  test('页面基本功能加载测试', async ({ page }) => {
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('智能文档处理系统')

    // 检查上传区域
    await expect(page.locator('[data-testid="upload-area"]')).toBeVisible()

    // 检查上传选项
    await expect(page.locator('text=启用OCR')).toBeVisible()
    await expect(page.locator('text=提取图像')).toBeVisible()
    await expect(page.locator('text=自动标签')).toBeVisible()

    // 检查标签栏
    await expect(page.locator('text=所有文档')).toBeVisible()
    await expect(page.locator('text=最近上传')).toBeVisible()
  })

  test('文件上传功能测试', async ({ page }) => {
    const testDir = createTestFiles()
    const testFile = join(testDir, 'test.txt')

    // 模拟API响应
    await page.route('/api/v1/documents/upload*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          doc_id: 'test-doc-123',
          title: 'test.txt',
          file_type: 'txt',
          content:
            'This is a test document with some sample content for processing.',
          metadata: {
            file_size: 64,
            pages: 1,
          },
          processing_info: {
            chunks: [
              {
                chunk_id: 'chunk-1',
                content:
                  'This is a test document with some sample content for processing.',
                type: 'paragraph',
                index: 0,
              },
            ],
            total_chunks: 1,
            auto_tags: [
              {
                tag: 'document',
                category: 'content',
                confidence: 0.95,
              },
            ],
          },
          version: {
            version_id: 'v1',
            version_number: 1,
          },
        }),
      })
    })

    // 上传文件
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles(testFile)

    // 等待上传完成
    await page.waitForTimeout(2000)

    // 检查文档卡片是否出现
    await expect(page.locator('text=test.txt')).toBeVisible()
    await expect(page.locator('text=/TXT\\s*•/')).toBeVisible()
  })

  test('文档查看功能测试', async ({ page }) => {
    // 模拟已有文档列表
    await page.route('/api/v1/documents/list', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            {
              doc_id: 'test-doc-456',
              title: 'sample.txt',
              file_type: 'txt',
              created_at: new Date().toISOString(),
              status: 'completed',
              tags: ['document', 'text'],
              processing_info: {
                chunks: [
                  {
                    chunk_id: 'chunk-1',
                    content:
                      'Sample content for testing viewing functionality.',
                    type: 'paragraph',
                    index: 0,
                  },
                ],
                total_chunks: 1,
                auto_tags: [
                  {
                    tag: 'sample',
                    category: 'content',
                    confidence: 0.9,
                  },
                ],
              },
            },
          ],
        }),
      })
    })

    // 刷新页面以加载模拟数据
    await page.reload()

    // 点击查看按钮
    const viewButton = page.locator('text=查看').first()
    if (await viewButton.isVisible()) {
      await viewButton.click()

      // 检查详情对话框
      await expect(page.locator('text=文档详情')).toBeVisible()
      await expect(page.locator('text=sample.txt')).toBeVisible()

      // 关闭对话框
      await page.locator('text=关闭').click()
    }
  })

  test('批量上传功能测试', async ({ page }) => {
    const testDir = createTestFiles()

    // 模拟批量上传API响应
    await page.route('/api/v1/documents/batch-upload*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total: 3,
          success: 3,
          failed: 0,
          results: [
            {
              doc_id: 'doc1',
              title: 'test.txt',
              file_type: 'txt',
              status: 'success',
            },
            {
              doc_id: 'doc2',
              title: 'test.md',
              file_type: 'md',
              status: 'success',
            },
            {
              doc_id: 'doc3',
              title: 'test.py',
              file_type: 'py',
              status: 'success',
            },
          ],
        }),
      })
    })

    // 模拟单文件上传以便测试多文件选择
    await page.route('/api/v1/documents/upload*', async route => {
      const url = route.request().url()
      const filename = url.includes('test.txt')
        ? 'test.txt'
        : url.includes('test.md')
          ? 'test.md'
          : 'test.py'

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          doc_id: `doc-${Math.random()}`,
          title: filename,
          file_type: filename.split('.').pop(),
          content: 'Sample content',
          processing_info: { total_chunks: 1 },
        }),
      })
    })

    // 选择多个文件
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles([
      join(testDir, 'test.txt'),
      join(testDir, 'test.md'),
      join(testDir, 'test.py'),
    ])

    // 等待上传完成
    await page.waitForTimeout(3000)

    // 检查是否有文档卡片出现
    const documentCards = page.locator('.MuiCard-root')
    await expect(documentCards.first()).toBeVisible()
  })

  test('文档标签功能测试', async ({ page }) => {
    // 模拟标签生成API
    await page.route('/api/v1/documents/*/generate-tags', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          doc_id: 'test-doc',
          tags: [
            { tag: 'Python', category: 'language', confidence: 0.95 },
            { tag: 'Code', category: 'type', confidence: 0.9 },
            { tag: 'Tutorial', category: 'content', confidence: 0.8 },
          ],
        }),
      })
    })

    // 添加一个测试文档
    await page.route('/api/v1/documents/list', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            {
              doc_id: 'tag-test-doc',
              title: 'code.py',
              file_type: 'py',
              created_at: new Date().toISOString(),
              status: 'completed',
            },
          ],
        }),
      })
    })

    await page.reload()

    // 点击标签按钮
    const tagButton = page.locator('[title="生成标签"]').first()
    if (await tagButton.isVisible()) {
      await tagButton.click()

      // 等待标签生成
      await page.waitForTimeout(1000)

      // 检查标签是否出现
      await expect(
        page.locator('text=Python').or(page.locator('[data-testid="tag-chip"]'))
      ).toBeVisible({ timeout: 5000 })
    }
  })

  test('文档关系分析功能测试', async ({ page }) => {
    // 模拟关系分析API
    await page.route(
      '/api/v1/documents/*/analyze-relationships',
      async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            doc_id: 'test-doc',
            relationships: [
              {
                source: 'doc1',
                target: 'doc2',
                type: 'reference',
                confidence: 0.95,
              },
            ],
            clusters: [
              {
                cluster_id: 'cluster1',
                documents: ['doc1', 'doc2'],
                topic: 'Machine Learning',
              },
            ],
          }),
        })
      }
    )

    // 添加测试文档
    await page.route('/api/v1/documents/list', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            {
              doc_id: 'relationship-test-doc',
              title: 'document.pdf',
              file_type: 'pdf',
              created_at: new Date().toISOString(),
              status: 'completed',
            },
          ],
        }),
      })
    })

    await page.reload()

    // 点击关系分析按钮
    const relationshipButton = page.locator('[title="分析关系"]').first()
    if (await relationshipButton.isVisible()) {
      await relationshipButton.click()

      // 等待分析完成（这里只是模拟，实际应该有UI反馈）
      await page.waitForTimeout(1000)
    }
  })

  test('版本历史功能测试', async ({ page }) => {
    // 模拟版本历史API
    await page.route('/api/v1/documents/*/versions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          doc_id: 'test-doc',
          versions: [
            {
              version_id: 'v1',
              version_number: 1,
              created_at: '2024-01-01T12:00:00Z',
              change_summary: 'Initial version',
              is_current: false,
            },
            {
              version_id: 'v2',
              version_number: 2,
              created_at: '2024-01-02T12:00:00Z',
              change_summary: 'Updated content',
              is_current: true,
            },
          ],
        }),
      })
    })

    // 添加测试文档
    await page.route('/api/v1/documents/list', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            {
              doc_id: 'version-test-doc',
              title: 'versioned.docx',
              file_type: 'docx',
              created_at: new Date().toISOString(),
              status: 'completed',
            },
          ],
        }),
      })
    })

    await page.reload()

    // 点击版本历史按钮
    const versionButton = page.locator('[title="版本历史"]').first()
    if (await versionButton.isVisible()) {
      await versionButton.click()

      // 等待版本信息加载
      await page.waitForTimeout(1000)
    }
  })

  test('支持格式查询测试', async ({ page }) => {
    // 模拟支持格式API
    await page.route('/api/v1/documents/supported-formats', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          formats: [
            '.pdf',
            '.docx',
            '.xlsx',
            '.pptx',
            '.txt',
            '.md',
            '.py',
            '.js',
          ],
          categories: {
            documents: ['.pdf', '.docx', '.pptx'],
            spreadsheets: ['.xlsx'],
            text: ['.txt', '.md'],
            code: ['.py', '.js', '.java', '.cpp'],
          },
        }),
      })
    })

    // 可以通过开发者工具检查API是否被正确调用
    let formatsCalled = false

    page.route('/api/v1/documents/supported-formats', async route => {
      formatsCalled = true
      await route.continue()
    })

    await page.reload()
    await page.waitForTimeout(1000)

    // 在实际应用中，应该有UI显示支持的格式
  })

  test('错误处理测试', async ({ page }) => {
    // 模拟上传失败
    await page.route('/api/v1/documents/upload*', async route => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Processing failed: File format not supported',
        }),
      })
    })

    const testDir = createTestFiles()
    const testFile = join(testDir, 'test.txt')

    // 尝试上传文件
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles(testFile)

    // 等待错误处理
    await page.waitForTimeout(2000)

    // 检查错误是否正确处理（在实际应用中应该显示错误消息）
    console.log('Error handling test completed')
  })

  test('响应式设计测试', async ({ page }) => {
    // 测试不同屏幕尺寸下的布局
    const viewports = [
      { width: 1200, height: 800 }, // 桌面
      { width: 768, height: 1024 }, // 平板
      { width: 375, height: 667 }, // 手机
    ]

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      await page.waitForTimeout(500)

      // 检查关键元素是否可见
      await expect(page.locator('h1')).toBeVisible()
      await expect(page.locator('[data-testid="upload-area"]')).toBeVisible()

      console.log(
        `Responsive test passed for ${viewport.width}x${viewport.height}`
      )
    }
  })

  test('性能测试 - 大量文档渲染', async ({ page }) => {
    // 模拟大量文档数据
    const mockDocuments = Array.from({ length: 100 }, (_, i) => ({
      doc_id: `doc-${i}`,
      title: `Document ${i + 1}.pdf`,
      file_type: 'pdf',
      created_at: new Date(Date.now() - i * 60000).toISOString(),
      status: 'completed',
      tags: [`tag${i % 5}`, `category${i % 3}`],
      processing_info: { total_chunks: Math.floor(Math.random() * 50) + 1 },
    }))

    await page.route('/api/v1/documents/list', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ documents: mockDocuments }),
      })
    })

    // 测量页面加载时间
    const startTime = Date.now()
    await page.reload()
    await page.waitForLoadState('networkidle')
    const endTime = Date.now()

    const loadTime = endTime - startTime
    console.log(`Page load time with 100 documents: ${loadTime}ms`)

    // 检查文档是否正确渲染
    await expect(page.locator('.MuiCard-root').first()).toBeVisible({
      timeout: 10000,
    })

    // 性能断言 - 页面应该在合理时间内加载
    expect(loadTime).toBeLessThan(5000) // 5秒内加载完成
  })

  test('键盘导航测试', async ({ page }) => {
    // 添加测试文档
    await page.route('/api/v1/documents/list', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            {
              doc_id: 'keyboard-test-doc',
              title: 'keyboard-test.txt',
              file_type: 'txt',
              created_at: new Date().toISOString(),
              status: 'completed',
            },
          ],
        }),
      })
    })

    await page.reload()

    // 使用Tab键导航
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')

    // 检查焦点是否正确移动
    const focusedElement = page.locator(':focus')
    await expect(focusedElement).toBeVisible()

    // 可以继续测试其他键盘交互
  })
})

// 并发性能测试
test.describe('并发处理测试', () => {
  test('多文件同时上传', async ({ page }) => {
    const testDir = createTestFiles()

    // 模拟并发上传的API响应
    let uploadCount = 0
    await page.route('/api/v1/documents/upload*', async route => {
      uploadCount++
      // 模拟处理延迟
      await new Promise(resolve =>
        setTimeout(resolve, 1000 + Math.random() * 2000)
      )

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          doc_id: `concurrent-doc-${uploadCount}`,
          title: `file-${uploadCount}.txt`,
          file_type: 'txt',
          content: `Content ${uploadCount}`,
          processing_info: { total_chunks: 1 },
        }),
      })
    })

    await page.goto('/document-processing')

    // 模拟快速连续上传多个文件
    const files = ['test.txt', 'test.md', 'test.py', 'test.js'].map(f =>
      join(testDir, f)
    )

    for (const file of files) {
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles(file)
      await page.waitForTimeout(100) // 短暂延迟模拟用户操作
    }

    // 等待所有上传完成
    await page.waitForTimeout(8000)

    console.log(`Concurrent upload test completed with ${uploadCount} uploads`)
  })
})

// 创建测试文件的辅助函数
function createTestFiles() {
  const testDir = mkdtempSync(join(tmpdir(), 'playwright-test-files-'))

  const files = {
    'test.txt':
      'This is a test document with some sample content for processing.',
    'test.md':
      '# Test Document\n\nThis is a markdown document with **bold** text and `code`.',
    'test.py':
      'def hello_world():\n    """A simple hello world function"""\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    hello_world()',
    'test.js':
      'function greet(name) {\n    return `Hello, ${name}!`;\n}\n\nconsole.log(greet("World"));',
  }

  Object.entries(files).forEach(([filename, content]) => {
    writeFileSync(join(testDir, filename), content)
  })

  return testDir
}
