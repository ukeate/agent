# Database Schema

基于PostgreSQL数据库和已定义的数据模型，以下是完整的数据库架构定义：

```sql
-- 启用必要的扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 智能体表
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('code_expert', 'architect', 'doc_expert', 'supervisor', 'rag_specialist')),
    status VARCHAR(20) DEFAULT 'idle' CHECK (status IN ('active', 'idle', 'busy', 'offline')),
    capabilities TEXT[] DEFAULT '{}',
    configuration JSONB NOT NULL DEFAULT '{}',
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 对话表
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    type VARCHAR(30) NOT NULL CHECK (type IN ('single_agent', 'multi_agent', 'workflow', 'rag_enhanced')),
    participants UUID[] DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    metadata JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 消息表
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_type VARCHAR(10) NOT NULL CHECK (sender_type IN ('user', 'agent', 'system')),
    sender_id VARCHAR(255) NOT NULL,
    content JSONB NOT NULL DEFAULT '{}',
    message_type VARCHAR(30) DEFAULT 'chat' CHECK (message_type IN ('chat', 'command', 'tool_call', 'system_notification')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- DAG执行表
CREATE TABLE dag_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    graph_definition JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'created' CHECK (status IN ('created', 'running', 'completed', 'failed', 'cancelled')),
    current_stage VARCHAR(100),
    progress JSONB DEFAULT '{"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "success_rate": 0}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 任务表
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL CHECK (type IN ('code_generation', 'code_review', 'documentation', 'analysis', 'planning')),
    dag_execution_id UUID REFERENCES dag_executions(id) ON DELETE CASCADE,
    assigned_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
    dependencies UUID[] DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority VARCHAR(10) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    execution_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 知识库条目表
CREATE TABLE knowledge_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('code', 'documentation', 'conversation', 'web_content', 'file')),
    source JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引优化查询性能
CREATE INDEX idx_agents_role_status ON agents(role, status);
CREATE INDEX idx_conversations_created_by ON conversations(created_by);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_tasks_dag_execution_id ON tasks(dag_execution_id);
CREATE INDEX idx_knowledge_items_content_type ON knowledge_items(content_type);
CREATE INDEX idx_knowledge_items_tags ON knowledge_items USING GIN(tags);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表创建更新时间触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_items_updated_at BEFORE UPDATE ON knowledge_items FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认智能体
INSERT INTO agents (name, role, capabilities, configuration) VALUES
('代码专家', 'code_expert', ARRAY['代码生成', '代码审查', '调试', '重构'], '{"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 4096, "tools": ["code_execution", "file_operations"], "system_prompt": "你是一位专业的代码专家，专注于高质量代码的生成、审查和优化。"}'),
('系统架构师', 'architect', ARRAY['系统设计', '技术选型', '架构评估', '文档编写'], '{"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 4096, "tools": ["documentation", "diagram_generation"], "system_prompt": "你是一位经验丰富的系统架构师，负责设计可扩展、可维护的软件架构。"}'),
('文档专家', 'doc_expert', ARRAY['技术文档', 'API文档', '用户手册', '代码注释'], '{"model": "gpt-4o-mini", "temperature": 0.4, "max_tokens": 4096, "tools": ["markdown_generation", "file_operations"], "system_prompt": "你是一位专业的技术文档专家，擅长创建清晰、准确、易懂的技术文档。"}'),
('任务调度器', 'supervisor', ARRAY['任务分解', '智能体协调', '工作流管理', '质量控制'], '{"model": "gpt-4o-mini", "temperature": 0.6, "max_tokens": 4096, "tools": ["task_management", "agent_coordination"], "system_prompt": "你是智能体团队的协调者，负责任务分解、分配和质量管控。"}'),
('知识检索专家', 'rag_specialist', ARRAY['语义搜索', '知识整合', '答案生成', '内容验证'], '{"model": "gpt-4o-mini", "temperature": 0.4, "max_tokens": 4096, "tools": ["vector_search", "knowledge_management"], "system_prompt": "你是知识检索和整合专家，擅长从大量信息中找到相关内容并生成准确答案。"}');
```
