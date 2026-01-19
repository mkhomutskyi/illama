# Contributing to illama

Thank you for your interest in contributing to illama! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming environment for everyone.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Include your hardware (GPU model, driver version)
- Include your software versions (Ubuntu, Docker, OpenVINO)
- Provide steps to reproduce the issue
- Include relevant log output

### Suggesting Features

- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider if the feature aligns with the project's goals (Ollama-like experience on Intel Arc)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Test your changes on Intel Arc hardware if possible
5. Update documentation as needed
6. Submit a pull request

## Development Setup

### Prerequisites

- Ubuntu 24.04+ (or compatible Linux distro)
- Intel Arc GPU with Level Zero runtime
- Docker and Docker Compose
- Python 3.10+

### Local Development

```bash
# Clone the repo
git clone https://github.com/mkhomutskyi/illama.git
cd illama

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Testing with Docker

```bash
# Build the image locally
docker build -t illama-manager:dev -f docker/Dockerfile .

# Run with development compose
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where practical
- Write docstrings for public functions
- Keep functions focused and readable

## Areas for Contribution

- **Model support**: Adding new model mappings and testing compatibility
- **Performance**: OpenVINO optimization, inference speed improvements
- **Documentation**: Guides, troubleshooting, model-specific notes
- **CLI features**: New commands, better output formatting
- **Testing**: Unit tests, integration tests, hardware-specific tests

## Questions?

Open an issue or discussion for any questions about contributing.
