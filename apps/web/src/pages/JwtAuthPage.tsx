import React from 'react'
import FeatureComingSoon from '../components/FeatureComingSoon'

const JwtAuthPage: React.FC = () => {
  return (
    <FeatureComingSoon 
      title="JWT身份认证"
      description="安全的JWT认证系统，支持令牌生成、验证和刷新机制。"
    />
  )
}

export default JwtAuthPage
