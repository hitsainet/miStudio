# Contributing to miStudio

Thank you for your interest in contributing to miStudio. This project is part of an open mechanistic interpretability research toolchain and contributions are welcome.

## Ways to Contribute

- **Bug reports** — Open an issue describing the problem, your environment (GPU, VRAM, OS), and steps to reproduce.
- **Feature requests** — Open an issue describing the use case and how it fits the research workflow.
- **Code contributions** — See the development setup below.
- **Documentation** — Corrections and additions to the [documentation site](https://onegaishimas.github.io/miStudio/) are welcome.

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Running Locally

```bash
# Clone the repository
git clone https://github.com/Onegaishimas/miStudio.git
cd miStudio

# Start all services
./start-mistudio.sh

# Access at http://mistudio.hitsai.local
# (requires 127.0.0.1 mistudio.hitsai.local in /etc/hosts)
```

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
DATABASE_URL=postgresql+asyncpg://... DATABASE_URL_SYNC=postgresql://... \
  python -m pytest tests/ -v
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev           # Dev server at http://localhost:3000
npm run type-check    # TypeScript type checking
npm run build         # Production build
```

## Code Standards

**Python:**
- Formatter: Black (line length 100)
- Linter: Ruff
- Type checker: MyPy (strict)
- Docstrings: Google style

**TypeScript:**
- Formatter: Prettier
- Linter: ESLint
- All components must be strictly typed

Run linters before submitting:

```bash
# Backend
cd backend
ruff check .
black --check .
mypy src/

# Frontend
cd frontend
npm run lint
npm run type-check
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with tests where applicable.
3. Ensure all existing tests pass.
4. Submit a pull request with a clear description of what changed and why.
5. Link any related issues in the PR description.

Maintainers review PRs on a best-effort basis. Please be patient.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add TopK SAE architecture support
fix: correct JumpReLU L0 loss calculation
docs: update extraction configuration guide
refactor: consolidate layer discovery into single function
test: add integration tests for steering endpoint
```

## Questions

Open a [GitHub Discussion](https://github.com/Onegaishimas/miStudio/discussions) for questions about the codebase, research questions, or general discussion about mechanistic interpretability tooling.
