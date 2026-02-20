# Contributing to MLOps Pipeline

Thanks for considering contributing! Here's how to get started.

## 🚀 Quick Setup

```bash
# Clone the repo
git clone https://github.com/sarifi08/mlops-pipeline.git
cd mlops-pipeline

# Install everything
make setup

# Train the model
make train

# Run the API
make serve

# Run tests
make test
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code
- Add tests for new functionality
- Update documentation if needed

### 3. Run Quality Checks

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test

# Check performance thresholds
make check
```

### 4. Commit & Push

Pre-commit hooks run automatically on `git commit`:
- Code formatting (ruff)
- Linting (ruff, mypy)
- Security checks (bandit)
- Large file detection

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

### 5. Open a Pull Request

- Describe what you changed and why
- Reference any related issues
- Ensure CI passes

## 🏗️ Project Structure

| Directory | Purpose |
|-----------|---------|
| `api/` | FastAPI serving layer |
| `model/` | Training pipeline |
| `monitoring/` | Drift detection, Prometheus, Grafana |
| `config/` | Settings and logging |
| `tests/` | Unit tests, load tests, performance checks |
| `.github/` | CI/CD workflows |

## 🧪 Testing Guidelines

- All new features must include tests
- Maintain test coverage above 70%
- Use descriptive test names: `test_prediction_returns_valid_format`
- Mark slow tests with `@pytest.mark.slow`

## 📝 Code Style

- Python 3.10+ type hints
- Docstrings for all public functions
- Use `ruff` for formatting and linting
- Follow existing code patterns

## 🔒 Security

- Never commit secrets or API keys
- Use environment variables for configuration
- Run `bandit` to check for security issues
- Report vulnerabilities privately via GitHub Security tab
