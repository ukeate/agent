import React from 'react'
import FeatureComingSoon from '../components/FeatureComingSoon'
import { BellOutlined } from '@ant-design/icons'

const DistributedEventsPage: React.FC = () => {
  return (
    <FeatureComingSoon
      title="分布式事件总线 (Event Bus)"
      description="高性能分布式事件系统，支持异步事件处理、路由和持久化"
      icon={<BellOutlined />}
      backendApi="/api/v1/events/*"
      technicalDetails={[
        '事件总线：高吞吐量的事件分发和路由系统',
        '异步处理：事件驱动的异步任务处理机制',
        '持久化存储：事件存储、重放和审计日志',
        '分布式协调：跨服务的事件协调和状态同步',
        '失败重试：智能重试机制和死信队列处理',
        '实时监控：事件流监控和性能指标统计',
      ]}
    />
  )
}

export default DistributedEventsPage
