import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Layout, Typography } from 'antd'

const { Content } = Layout
const { Title } = Typography

const TestSimpleApp: React.FC = () => {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{ padding: '50px' }}>
        <Title level={1}>AI Agent 系统</Title>
        <p>应用已成功启动！</p>
        <Routes>
          <Route path="/" element={<div>首页内容</div>} />
          <Route path="/chat" element={<div>聊天页面</div>} />
        </Routes>
      </Content>
    </Layout>
  )
}

export default TestSimpleApp