# Repository Guidelines

## 交流
- 交流、注释、日志统一使用简体中文。技术术语、提交记录统一使用英文。
- 请审视的输入的潜在问题，能提出明显在我思考框架外的建议。

## 不能做
- 不写兼容性代码，严禁暂时注释, 严禁写补丁代码，严禁假实现，严禁任何TODO
- 不写重复代码，总是先查找复用或重写已有代码，不保留死代码
- 严禁跳过发现的错误
- 不写脚本，不写总结性md文档
- YAGNI原则「You aren’t gonna need it！」: 你自以为有用的功能，实际上是用不到的。除了要求的核心功能，其他功能一概不要部署。这一原则的核心思想是，尽可能快、尽可能简单的将软件运行起来。

## Project Structure & Module Organization
This npm workspace repo groups runtime code under `apps/`: `apps/api` is the FastAPI backend (source in `src/`, tests in `tests/`), and `apps/web` is the React UI (modules in `src/`). Shared TypeScript utilities and schemas live in `packages/shared`. Experimental AI workflows sit in `src/ai`. Supporting references stay in `docs/`, automation scripts in `scripts/`, and container config in `infrastructure/docker/`.

## Build, Test, and Development Commands
Run `npm run install:all` to install Node packages and sync Python deps with `uv`. Use `npm run dev` for full-stack hot reload, or scope to `npm run dev:web` and `npm run dev:api`. Build all workspaces with `npm run build`, or target pieces through `npm run build:web` / `npm run build:api`. Quality gates: `npm run lint`, `npm run typecheck`, and `npm run format`. Container workflows rely on `npm run docker:up|down|logs`.

## Coding Style & Naming Conventions
TypeScript follows ESLint + Prettier defaults (2-space indent, semicolons, trailing commas). Components and context providers stay in PascalCase; hooks begin with `use`; shared contracts live in `packages/shared`. Python is formatted by Black (4 spaces, 88 columns) and linted by Ruff/Black via `npm run lint:api`; keep modules and functions snake_case and classes PascalCase. API routes should use kebab-case segments.

## Testing Guidelines
Pytest drives backend tests; follow `apps/api/pytest.ini` conventions (`test_*.py`, `Test*` classes). Run `npm run test:api` or `uv run pytest` before sending PRs, and add fixtures under `apps/api/tests/fixtures`. Frontend units use Vitest via `npm run test:web`; colocate files as `.test.ts(x)` beside the component. Critical flows deserve Playwright coverage with `npm run test:e2e`. Include coverage switches (`vitest --coverage`, `pytest --cov`) when changing shared logic.

## Commit & Pull Request Guidelines
History favors Conventional Commits (`feat:`, `docs:`, `refactor:`). Keep subjects ≤72 chars, write in imperative mood, and split refactors from behaviour changes. Before opening a PR, run lint, tests, and type checks for relevant workspaces. PR descriptions should explain the problem, outline the solution, link issues/OKRs, and include screenshots or API responses for UI or contract updates.

## Environment & Configuration Tips
Copy `.env.example` to `.env` in `apps/api` (and any new workspace) and keep secrets out of Git. Prefer Docker Compose files in `infrastructure/docker/` for parity with CI. Update `scripts/setup-dev.sh` and README sections whenever dependencies or services shift so downstream agents stay synchronized.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
