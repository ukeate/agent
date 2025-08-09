-- AI Agent System Database Initialization

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 创建schemas
CREATE SCHEMA IF NOT EXISTS agent_core;
CREATE SCHEMA IF NOT EXISTS knowledge_graph;
CREATE SCHEMA IF NOT EXISTS conversations;
CREATE SCHEMA IF NOT EXISTS tasks;

-- 设置默认权限
GRANT ALL PRIVILEGES ON SCHEMA agent_core TO ai_agent_user;
GRANT ALL PRIVILEGES ON SCHEMA knowledge_graph TO ai_agent_user;
GRANT ALL PRIVILEGES ON SCHEMA conversations TO ai_agent_user;
GRANT ALL PRIVILEGES ON SCHEMA tasks TO ai_agent_user;

-- 创建基础表结构的示例（实际由Alembic管理）
-- 这里只是为了验证连接和权限

CREATE TABLE IF NOT EXISTS agent_core.health_check (
    id SERIAL PRIMARY KEY,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'healthy'
);

INSERT INTO agent_core.health_check (status) VALUES ('initialized');