# Unified Project Structure

基于monorepo架构和选择的技术工具，以下是完整的项目结构定义：

```plaintext
ai-agent-system/
├── .github/                           # CI/CD工作流
│   └── workflows/
│       ├── ci.yaml                    # 持续集成流水线
│       ├── deploy-staging.yaml        # 预发环境部署
│       └── deploy-production.yaml     # 生产环境部署
├── apps/                              # 应用程序包
│   ├── web/                           # 前端React应用
│   │   ├── public/                    # 静态资源
│   │   ├── src/
│   │   │   ├── components/            # React组件
│   │   │   │   ├── ui/                # 通用UI组件
│   │   │   │   ├── layout/            # 布局组件
│   │   │   │   ├── agent/             # 智能体组件
│   │   │   │   ├── conversation/      # 对话组件
│   │   │   │   ├── task/              # 任务组件
│   │   │   │   └── knowledge/         # 知识库组件
│   │   │   ├── pages/                 # 页面组件
│   │   │   ├── hooks/                 # 自定义hooks
│   │   │   ├── services/              # API服务层
│   │   │   ├── stores/                # 状态管理
│   │   │   ├── styles/                # 全局样式和主题
│   │   │   ├── utils/                 # 前端工具函数
│   │   │   ├── types/                 # TypeScript类型定义
│   │   │   ├── App.tsx                # 根组件
│   │   │   └── main.tsx               # 应用入口
│   │   ├── tests/                     # 前端测试
│   │   ├── package.json               # 前端依赖配置
│   │   ├── tailwind.config.js         # Tailwind CSS配置
│   │   ├── tsconfig.json              # TypeScript配置
│   │   └── vite.config.ts             # Vite构建配置
│   └── api/                           # 后端FastAPI应用
│       ├── src/
│       │   ├── api/                   # API路由层
│       │   │   ├── v1/
│       │   │   ├── deps.py            # 依赖注入
│       │   │   ├── middleware.py      # 中间件
│       │   │   └── exceptions.py      # 异常处理
│       │   ├── core/                  # 核心配置
│       │   │   ├── config.py          # 应用配置
│       │   │   ├── security.py        # 安全相关
│       │   │   ├── database.py        # 数据库连接
│       │   │   └── logging.py         # 日志配置
│       │   ├── services/              # 业务逻辑层
│       │   ├── models/                # 数据模型
│       │   │   ├── database/          # 数据库模型
│       │   │   ├── schemas/           # Pydantic数据模型
│       │   │   └── enums.py           # 枚举定义
│       │   ├── repositories/          # 数据访问层
│       │   ├── ai/                    # AI集成模块
│       │   │   ├── langgraph/         # LangGraph集成
│       │   │   ├── autogen/           # AutoGen集成
│       │   │   ├── mcp/               # MCP协议实现
│       │   │   ├── rag/               # RAG系统
│       │   │   ├── dag/               # DAG执行引擎
│       │   │   └── openai_client.py   # OpenAI API客户端
│       │   ├── utils/                 # 工具函数
│       │   ├── alembic/               # 数据库迁移
│       │   └── main.py                # FastAPI应用入口
│       ├── tests/                     # 后端测试
│       ├── Dockerfile                 # Docker镜像
│       ├── pyproject.toml             # Python项目配置
│       └── requirements.txt           # Python依赖
├── packages/                          # 共享包
│   ├── shared/                        # 共享类型和工具
│   │   ├── src/
│   │   │   ├── types/                 # 共享TypeScript类型
│   │   │   ├── constants/             # 共享常量
│   │   │   ├── utils/                 # 共享工具函数
│   │   │   └── index.ts               # 包导出入口
│   │   ├── package.json
│   │   └── tsconfig.json
│   ├── ui/                            # 共享UI组件库
│   │   ├── src/
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── styles/
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── config/                        # 共享配置
│       ├── eslint/
│       ├── typescript/
│       └── jest/
├── infrastructure/                    # 基础设施即代码
│   ├── docker/                        # Docker配置
│   │   ├── Dockerfile.web
│   │   ├── Dockerfile.api
│   │   ├── docker-compose.yml         # 本地开发环境
│   │   ├── docker-compose.prod.yml    # 生产环境
│   │   └── nginx.conf                 # Nginx配置
│   ├── k8s/                          # Kubernetes部署配置
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   └── hpa.yaml
│   └── terraform/                     # Terraform IaC (可选)
├── scripts/                           # 构建和部署脚本
│   ├── build.sh                       # 构建脚本
│   ├── deploy.sh                      # 部署脚本
│   ├── test.sh                        # 测试脚本
│   ├── setup-dev.sh                   # 开发环境设置
│   ├── db-migrate.sh                  # 数据库迁移
│   └── seed-data.py                   # 种子数据生成
├── docs/                              # 项目文档
│   ├── brief.md                       # 项目简介
│   ├── prd.md                         # 产品需求文档
│   ├── front-end-spec.md              # 前端规格文档
│   ├── architecture.md                # 架构设计文档
│   ├── api/                           # API文档
│   ├── deployment/                    # 部署文档
│   └── development/                   # 开发文档
├── .env.example                       # 全局环境变量模板
├── .gitignore                         # Git忽略文件
├── .editorconfig                      # 编辑器配置
├── .prettierrc                        # Prettier配置
├── .eslintrc.js                       # ESLint配置
├── package.json                       # 根package.json (monorepo)
├── package-lock.json                  # 依赖锁文件
├── tsconfig.json                      # 根TypeScript配置
├── jest.config.js                     # Jest测试配置
└── README.md                         # 项目说明文档
```
