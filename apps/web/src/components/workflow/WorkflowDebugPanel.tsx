import React from 'react';
import { Card } from 'antd';

interface WorkflowDebugPanelProps {
  workflowId: string;
  className?: string;
}

export const WorkflowDebugPanel: React.FC<WorkflowDebugPanelProps> = ({ 
  workflowId,
  className 
}) => {
  return (
    <Card 
      title="Workflow Debug Panel" 
      className={className}
    >
      <div className="p-4 text-center text-gray-500">
        Debug panel for workflow: {workflowId}
        <br />
        <small>This component is under development.</small>
      </div>
    </Card>
  );
};

export default WorkflowDebugPanel;