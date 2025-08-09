# Testing Strategy

基于全栈AI应用的复杂性，定义分层测试策略确保系统质量：

## Testing Pyramid

```text
                  E2E Tests (10%)
                 /              \
            Integration Tests (20%)
               /                    \
          Frontend Unit (35%)    Backend Unit (35%)
```

## Test Organization

### Frontend Tests
```text
apps/web/tests/
├── __mocks__/                     # Mock数据和服务
├── components/                    # 组件测试
├── hooks/                         # Hook测试
├── services/                      # 服务层测试
├── stores/                        # 状态管理测试
├── utils/                         # 工具函数测试
├── pages/                         # 页面集成测试
└── e2e/                          # E2E测试
```

### Backend Tests
```text
apps/api/tests/
├── conftest.py                    # pytest配置和fixture
├── api/                          # API端点测试
├── services/                     # 业务逻辑测试
├── repositories/                 # 数据访问测试
├── ai/                          # AI模块测试
├── utils/                       # 工具函数测试
└── integration/                 # 集成测试
```

**测试覆盖率目标:**
- **单元测试覆盖率**: ≥80%
- **集成测试覆盖率**: ≥70%
- **E2E测试覆盖率**: ≥60% (关键用户流程)
- **AI模块测试覆盖率**: ≥85% (关键业务逻辑)
