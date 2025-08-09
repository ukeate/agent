import { render } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import App from '../src/App'

describe('App Component', () => {
  it('renders without crashing', () => {
    render(
      <BrowserRouter>
        <ConfigProvider>
          <App />
        </ConfigProvider>
      </BrowserRouter>
    )
    expect(document.body).toBeTruthy()
  })
})