import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { ChatPage } from './pages/ChatPage'
import { MultiAgentPage } from './pages/MultiAgentPage'
import { SupervisorPage } from './pages/SupervisorPage'
import RagPage from './pages/RagPage'
import AgenticRagPage from './pages/AgenticRagPage'

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<ChatPage />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/multi-agent" element={<MultiAgentPage />} />
      <Route path="/supervisor" element={<SupervisorPage />} />
      <Route path="/rag" element={<RagPage />} />
      <Route path="/agentic-rag" element={<AgenticRagPage />} />
    </Routes>
  )
}

export default App