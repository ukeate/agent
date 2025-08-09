import React from 'react';
import { Card } from 'antd';

interface WorkflowVisualizationProps {
  workflowId: string;
  onNodeClick?: (nodeId: string) => void;
}

export const WorkflowVisualization: React.FC<WorkflowVisualizationProps> = ({
  workflowId,
  onNodeClick = () => {}
}) => {
  const handleClick = () => {
    if (onNodeClick) {
      onNodeClick('sample-node');
    }
  };

  return (
    <Card title="Workflow Visualization" className="h-96">
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <div>Workflow visualization for: {workflowId}</div>
          <small>This component is under development.</small>
          <br />
          <button 
            onClick={handleClick} 
            className="mt-2 text-blue-500 hover:text-blue-700"
          >
            Test Node Click
          </button>
        </div>
      </div>
    </Card>
  );
};

export default WorkflowVisualization;