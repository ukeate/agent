import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ConfigProvider } from 'antd'
import RagQueryPanel from '../../../src/components/rag/RagQueryPanel'

const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <ConfigProvider>
      {ui}
    </ConfigProvider>
  )
}

describe('RagQueryPanel', () => {
  const mockOnQuery = vi.fn()
  const mockOnClearHistory = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders basic elements', () => {
    renderWithProviders(
      <RagQueryPanel
        onQuery={mockOnQuery}
        onClearHistory={mockOnClearHistory}
        loading={false}
        queryHistory={[]}
      />
    )
    
    expect(screen.getByPlaceholderText(/请输入查询问题/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /搜索/ })).toBeInTheDocument()
  })

  it('handles query input and submission', async () => {
    renderWithProviders(
      <RagQueryPanel
        onQuery={mockOnQuery}
        onClearHistory={mockOnClearHistory}
        loading={false}
        queryHistory={[]}
      />
    )
    
    const input = screen.getByPlaceholderText(/请输入查询问题/)
    const searchButton = screen.getByRole('button', { name: /搜索/ })
    
    fireEvent.change(input, { target: { value: 'test query' } })
    fireEvent.click(searchButton)
    
    await waitFor(() => {
      expect(mockOnQuery).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'test query'
        })
      )
    })
  })

  it('shows loading state', () => {
    renderWithProviders(
      <RagQueryPanel
        onQuery={mockOnQuery}
        onClearHistory={mockOnClearHistory}
        loading={true}
        queryHistory={[]}
      />
    )
    
    const searchButton = screen.getByRole('button', { name: /搜索/ })
    expect(searchButton).toBeDisabled()
  })

  it('displays query history', () => {
    const mockHistory = [
      { id: '1', query: 'previous query 1', timestamp: '2023-12-01T10:00:00Z' },
      { id: '2', query: 'previous query 2', timestamp: '2023-12-01T10:01:00Z' }
    ]
    
    renderWithProviders(
      <RagQueryPanel
        onQuery={mockOnQuery}
        onClearHistory={mockOnClearHistory}
        loading={false}
        queryHistory={mockHistory}
      />
    )
    
    expect(screen.getByText('previous query 1')).toBeInTheDocument()
    expect(screen.getByText('previous query 2')).toBeInTheDocument()
  })

  it('validates empty query input', async () => {
    renderWithProviders(
      <RagQueryPanel
        onQuery={mockOnQuery}
        onClearHistory={mockOnClearHistory}
        loading={false}
        queryHistory={[]}
      />
    )
    
    const searchButton = screen.getByRole('button', { name: /搜索/ })
    fireEvent.click(searchButton)
    
    // Should show validation error or prevent submission
    await waitFor(() => {
      expect(mockOnQuery).not.toHaveBeenCalled()
    })
  })
})