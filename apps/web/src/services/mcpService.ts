/**
 * MCP工具服务
 * 提供MCP协议工具的前端接口
 */

import apiClient from './apiClient'

// 类型定义
export interface ToolInfo {
  name: string
  description?: string
  inputSchema?: Record<string, any>
  server_type: string
}

export interface ToolCallRequest {
  server_type: string
  tool_name: string
  arguments: Record<string, any>
}

export interface ToolCallResponse {
  success: boolean
  result?: any
  error?: string
  error_type?: string
  tool_name: string
  server_type: string
}

export interface ServerStatus {
  initialized: boolean
  healthy: boolean
  status: string
  tools_count: number
  server_info?: Record<string, any>
}

export interface HealthCheckResponse {
  initialized: boolean
  overall_healthy: boolean
  servers: Record<string, ServerStatus>
}

export interface AvailableToolsResponse {
  tools: Record<string, ToolInfo[]>
}

export interface MetricsResponse {
  monitoring_stats: {
    total_calls: number
    successful_calls: number
    failed_calls: number
    average_response_time: number
    calls_by_server: Record<string, number>
    calls_by_tool: Record<string, number>
    error_counts: Record<string, number>
  }
  retry_stats: {
    total_retries: number
    successful_retries: number
    failed_retries: number
    retry_reasons: Record<string, number>
  }
}

// 文件系统操作接口
export interface FileSystemReadRequest {
  path: string
  encoding?: string
}

export interface FileSystemWriteRequest {
  path: string
  content: string
  encoding?: string
}

export interface FileSystemListRequest {
  path: string
  include_hidden?: boolean
}

// 数据库操作接口
export interface DatabaseQueryRequest {
  query: string
  parameters?: Record<string, any>
}

// 系统操作接口
export interface SystemCommandRequest {
  command: string
  timeout?: number
}

class MCPService {
  /**
   * 调用MCP工具
   */
  async callTool(request: ToolCallRequest): Promise<ToolCallResponse> {
    try {
      const response = await apiClient.post('/mcp/tools/call', request)
      return response.data
    } catch (error) {
      console.error('MCP工具调用失败:', error)
      throw error
    }
  }

  /**
   * 获取可用工具列表
   */
  async listAvailableTools(serverType?: string): Promise<AvailableToolsResponse> {
    try {
      const params = serverType ? { server_type: serverType } : {}
      const response = await apiClient.get('/mcp/tools', { params })
      return response.data
    } catch (error) {
      console.error('获取MCP工具列表失败:', error)
      // 返回默认值
      return {
        tools: {
          filesystem: [
            { name: 'read_file', description: '读取文件内容', server_type: 'filesystem' },
            { name: 'write_file', description: '写入文件内容', server_type: 'filesystem' },
            { name: 'list_directory', description: '列出目录内容', server_type: 'filesystem' },
            { name: 'file_info', description: '获取文件信息', server_type: 'filesystem' }
          ],
          database: [
            { name: 'execute_query', description: '执行SQL查询', server_type: 'database' },
            { name: 'describe_tables', description: '获取表结构', server_type: 'database' },
            { name: 'execute_transaction', description: '执行事务', server_type: 'database' }
          ],
          system: [
            { name: 'run_command', description: '运行系统命令', server_type: 'system' },
            { name: 'check_process', description: '检查进程状态', server_type: 'system' },
            { name: 'get_env', description: '获取环境变量', server_type: 'system' },
            { name: 'get_system_info', description: '获取系统信息', server_type: 'system' }
          ]
        }
      }
    }
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    try {
      const response = await apiClient.get('/mcp/health')
      return response.data
    } catch (error) {
      console.error('MCP健康检查失败:', error)
      // 返回默认值
      return {
        initialized: false,
        overall_healthy: false,
        servers: {
          filesystem: {
            initialized: false,
            healthy: false,
            status: 'unknown',
            tools_count: 0
          },
          database: {
            initialized: false,
            healthy: false,
            status: 'unknown',
            tools_count: 0
          },
          system: {
            initialized: false,
            healthy: false,
            status: 'unknown',
            tools_count: 0
          }
        }
      }
    }
  }

  /**
   * 获取指标
   */
  async getMetrics(): Promise<MetricsResponse> {
    try {
      const response = await apiClient.get('/mcp/metrics')
      return response.data
    } catch (error) {
      console.error('获取MCP指标失败:', error)
      // 返回默认值
      return {
        monitoring_stats: {
          total_calls: 0,
          successful_calls: 0,
          failed_calls: 0,
          average_response_time: 0,
          calls_by_server: {},
          calls_by_tool: {},
          error_counts: {}
        },
        retry_stats: {
          total_retries: 0,
          successful_retries: 0,
          failed_retries: 0,
          retry_reasons: {}
        }
      }
    }
  }

  // 文件系统操作
  async readFile(path: string, encoding: string = 'utf-8'): Promise<any> {
    try {
      const response = await apiClient.post('/mcp/tools/filesystem/read', null, {
        params: { path, encoding }
      })
      return response.data
    } catch (error) {
      console.error('读取文件失败:', error)
      throw error
    }
  }

  async writeFile(path: string, content: string, encoding: string = 'utf-8'): Promise<any> {
    try {
      const response = await apiClient.post('/mcp/tools/filesystem/write', null, {
        params: { path, content, encoding }
      })
      return response.data
    } catch (error) {
      console.error('写入文件失败:', error)
      throw error
    }
  }

  async listDirectory(path: string, includeHidden: boolean = false): Promise<any> {
    try {
      const response = await apiClient.get('/mcp/tools/filesystem/list', {
        params: { path, include_hidden: includeHidden }
      })
      return response.data
    } catch (error) {
      console.error('列出目录失败:', error)
      throw error
    }
  }

  // 数据库操作
  async executeQuery(query: string, parameters?: Record<string, any>): Promise<any> {
    try {
      const response = await apiClient.post('/mcp/tools/database/query', null, {
        params: { query, parameters }
      })
      return response.data
    } catch (error) {
      console.error('执行查询失败:', error)
      throw error
    }
  }

  // 系统操作
  async runCommand(command: string, timeout?: number): Promise<any> {
    try {
      const response = await apiClient.post('/mcp/tools/system/command', null, {
        params: { command, timeout }
      })
      return response.data
    } catch (error) {
      console.error('运行命令失败:', error)
      throw error
    }
  }
}

export default new MCPService()