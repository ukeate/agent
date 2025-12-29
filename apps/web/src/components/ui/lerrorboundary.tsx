import { Component, ErrorInfo, ReactNode } from 'react'
import { Button, Result, Typography } from 'antd'
import { ReloadOutlined, BugOutlined } from '@ant-design/icons'

import { logger } from '../../utils/logger'
const { Paragraph, Text } = Typography

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
  showDetails?: boolean
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, showDetails: false }
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    logger.error('错误边界捕获到异常:', error, errorInfo)
    this.setState({
      error,
      errorInfo,
    })
  }

  handleReload = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined, showDetails: false })
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <Result
            status="error"
            title="应用程序遇到错误"
            subTitle="很抱歉，应用程序遇到了意外错误。请尝试刷新页面或联系技术支持。"
            extra={[
              <Button 
                type="primary" 
                icon={<ReloadOutlined />} 
                onClick={this.handleReload}
                key="reload"
              >
                刷新页面
              </Button>,
              <Button 
                icon={<BugOutlined />} 
                onClick={() => this.setState({ showDetails: !this.state.showDetails })}
                key="details"
              >
                查看详情
              </Button>,
            ]}
          >
            {this.state.showDetails && this.state.error && (
              <div className="text-left mt-4 p-4 bg-gray-100 rounded">
                <Paragraph>
                  <Text strong>错误信息:</Text>
                </Paragraph>
                <Paragraph>
                  <Text code>{this.state.error.message}</Text>
                </Paragraph>
                
                {this.state.errorInfo && (
                  <>
                    <Paragraph>
                      <Text strong>错误堆栈:</Text>
                    </Paragraph>
                    <Paragraph>
                      <Text code className="whitespace-pre-wrap text-xs">
                        {this.state.errorInfo.componentStack}
                      </Text>
                    </Paragraph>
                  </>
                )}
              </div>
            )}
          </Result>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
export { ErrorBoundary }
