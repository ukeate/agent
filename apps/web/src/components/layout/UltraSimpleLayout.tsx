import React from 'react'
import { Layout } from 'antd'

const { Content } = Layout

interface UltraSimpleLayoutProps {
  children: React.ReactNode
}

const UltraSimpleLayout: React.FC<UltraSimpleLayoutProps> = ({ children }) => {
  return (
    <Layout>
      <Content style={{ padding: '20px', minHeight: '100vh' }}>
        <div style={{ background: '#fff', padding: '20px', borderRadius: '8px' }}>
          {children}
        </div>
      </Content>
    </Layout>
  )
}

export default UltraSimpleLayout
export { UltraSimpleLayout }