-- 创建工作流相关数据库表

-- 工作流表
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(50) NOT NULL,
    definition JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    paused_at TIMESTAMP WITH TIME ZONE,
    resumed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    failed_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    error_message TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- 工作流检查点表
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    id VARCHAR PRIMARY KEY,
    workflow_id VARCHAR NOT NULL,
    checkpoint_id VARCHAR NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    state_data JSONB NOT NULL,
    checkpoint_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN DEFAULT FALSE
);

-- DAG执行表
CREATE TABLE IF NOT EXISTS dag_executions (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    graph_definition JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_task_id UUID,
    context JSONB NOT NULL DEFAULT '{}',
    checkpoints JSONB NOT NULL DEFAULT '[]',
    progress JSONB NOT NULL DEFAULT '{"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0}',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- 任务表
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY,
    dag_execution_id UUID NOT NULL REFERENCES dag_executions(id),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB,
    error_message TEXT,
    dependencies TEXT[],
    agent_id VARCHAR(255),
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_type ON workflows(workflow_type);
CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_workflow_id ON workflow_checkpoints(workflow_id);
CREATE INDEX IF NOT EXISTS idx_dag_executions_conversation_id ON dag_executions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tasks_dag_execution_id ON tasks(dag_execution_id);