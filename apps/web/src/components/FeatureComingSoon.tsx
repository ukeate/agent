import React from 'react'
import { Card } from 'antd'
import { ExperimentOutlined } from '@ant-design/icons'

interface FeatureComingSoonProps {
  title: string
  description?: string
}

export const FeatureComingSoon: React.FC<FeatureComingSoonProps> = ({ 
  title, 
  description = "该功能正在开发中，敬请期待。" 
}) => {
  return (
    <div style={{ padding: '24px', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
      <Card style={{ textAlign: 'center', maxWidth: '500px', width: '100%' }}>
        <ExperimentOutlined style={{ fontSize: '64px', color: '#1890ff', marginBottom: '16px' }} />
        <h2 style={{ marginBottom: '8px', fontSize: '24px' }}>{title}</h2>
        <p style={{ color: '#666', fontSize: '16px', marginBottom: '0' }}>{description}</p>
      </Card>
    </div>
  )
}

export default FeatureComingSoon