# Security and Performance

基于全栈AI应用的特殊需求，定义安全和性能的综合策略：

## Security Requirements

**Frontend Security:**
- CSP Headers: `default-src 'self'; script-src 'self' 'unsafe-eval'; connect-src 'self' ws: wss: https://api.anthropic.com https://api.openai.com;`
- XSS Prevention: DOMPurify sanitization for user-generated content, Content Security Policy enforcement
- Secure Storage: JWT tokens in httpOnly cookies, sensitive data encrypted in localStorage using Web Crypto API

**Backend Security:**
- Input Validation: Pydantic models with comprehensive validation, SQL injection prevention through parameterized queries
- Rate Limiting: `{"global": {"requests_per_minute": 1000}, "per_user": {"requests_per_minute": 100}, "ai_api": {"requests_per_minute": 50}}`
- CORS Policy: `{"allow_origins": ["https://ai-agent-system.com"], "allow_methods": ["GET", "POST", "PUT", "DELETE"], "allow_headers": ["Authorization", "Content-Type"]}`

**Authentication Security:**
- Token Storage: JWT access tokens (30min expiry) + refresh tokens (7 days) stored in secure httpOnly cookies
- Session Management: Redis-based session store with automatic cleanup, concurrent session limits (5 sessions per user)
- Password Policy: Minimum 8 characters, must include uppercase, lowercase, number, and special character; bcrypt hashing with cost factor 12

## Performance Optimization

**Frontend Performance:**
- Bundle Size Target: `{"initial": "< 500KB gzipped", "total": "< 2MB", "code_splitting": "route-based + component-based"}`
- Loading Strategy: Progressive loading with skeleton screens, image lazy loading, virtual scrolling for large lists
- Caching Strategy: `{"static_assets": "1 year", "api_responses": "5 minutes", "user_data": "session-based"}`

**Backend Performance:**
- Response Time Target: `{"p95": "< 200ms", "p99": "< 500ms", "ai_operations": "< 5s"}`
- Database Optimization: Connection pooling (min: 5, max: 20), query optimization with EXPLAIN ANALYZE, index optimization
- Caching Strategy: `{"redis": {"ttl": 300, "keys": ["user_sessions", "api_responses", "computed_results"]}, "in_memory": {"lru_cache": 1000}}`
