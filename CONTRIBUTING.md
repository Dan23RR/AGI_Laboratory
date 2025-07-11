# 🤝 Contributing to AGI Laboratory

First off, thank you for considering contributing to AGI Laboratory! It's people like you that make AGI Laboratory such a great tool for the AI research community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/AGI_Laboratory.git
   cd AGI_Laboratory
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Run the tests** to ensure everything is working:
   ```bash
   python -m pytest tests/
   python tests/test_full_integration.py
   ```

## 💡 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- System information (OS, Python version, PyTorch version)
- Relevant logs or error messages

### 💭 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- Step-by-step description of the suggested enhancement
- Why this enhancement would be useful
- Examples of how it would work

### 🎯 Priority Areas

We especially need help with:

#### 1. **Fitness Functions** 
```python
# Example: Create a new fitness function for language understanding
def language_understanding_fitness(genome: ExtendedGenome) -> float:
    """Evaluate language comprehension capabilities"""
    # Your implementation here
    pass
```

#### 2. **New Cognitive Modules**
- Implement new V3/V4 modules following our architecture
- Improve existing modules for efficiency
- Add new capabilities to modules

#### 3. **Test Environments**
- Create challenging scenarios for AGI testing
- Implement benchmark tasks
- Design evaluation metrics

#### 4. **Performance Optimization**
- Profile and optimize hot paths
- Reduce memory usage
- Implement GPU optimizations

#### 5. **Documentation**
- Improve code documentation
- Write tutorials
- Create example notebooks

## 🔧 Development Process

### 1. **Create a Branch**
```bash
git checkout -b feature/amazing-feature
# or
git checkout -b fix/bug-description
```

### 2. **Make Your Changes**
- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. **Test Your Changes**
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python tests/test_full_integration.py

# Check code style
black --check .
flake8 .
```

### 4. **Commit Your Changes**
See [Commit Messages](#commit-messages) section below

### 5. **Push to Your Fork**
```bash
git push origin feature/amazing-feature
```

### 6. **Create a Pull Request**
Go to the original repository and create a PR from your fork

## 🎨 Style Guidelines

### Python Style

We follow PEP 8 with these specifications:

```python
# Use Black for formatting
black --line-length 100 .

# Use type hints
def evolve_genome(genome: ExtendedGenome, generations: int) -> ExtendedGenome:
    """Evolve a genome for specified generations."""
    pass

# Use descriptive variable names
# Good
fitness_scores = calculate_fitness(population)

# Bad
f = calc_f(p)

# Document complex logic
# Calculate adaptive mutation rate based on fitness variance
mutation_rate = base_rate * (1 + fitness_variance / max_variance)
```

### Docstring Format

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of function.
    
    More detailed description if needed, explaining the purpose
    and any important details about the implementation.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["status"])
        'success'
    """
    pass
```

### Module Structure

New modules should follow this structure:

```python
"""Module description.

Detailed module documentation here.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from core.base_module import BaseAGIModule
from core.module_config import ModuleConfig


class YourModuleV3(BaseAGIModule):
    """Your module description."""
    
    def __init__(self, config: ModuleConfig):
        """Initialize the module."""
        super().__init__(config)
        self._build_module()
        
    def _build_module(self):
        """Build module architecture."""
        pass
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        pass
        
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get module state for integration."""
        pass
```

## 💬 Commit Messages

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Examples
```
feat(modules): add QuantumReasonerV3 module

Implement quantum-inspired reasoning module with:
- Superposition state handling
- Entanglement detection
- Measurement collapse simulation

Closes #123
```

## 🔄 Pull Request Process

1. **Ensure all tests pass** and code meets style guidelines
2. **Update the README.md** if needed
3. **Add yourself to CONTRIBUTORS.md** (if first contribution)
4. **Fill out the PR template** completely
5. **Link related issues** using keywords (Closes #123)
6. **Request review** from maintainers

### PR Title Format
```
[Module] Add consciousness streaming support
[Fix] Resolve memory leak in evolution lab
[Docs] Improve fitness function documentation
```

### PR Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog entry added (for significant changes)
- [ ] No merge conflicts

## 🎯 Areas of Special Interest

### 1. **Novel Architectures**
We're particularly interested in:
- Attention mechanisms for AGI
- Memory architectures
- Multi-modal integration
- Emergent communication

### 2. **Evolutionary Strategies**
- New selection mechanisms
- Crossover operators for neural architectures
- Adaptive mutation strategies
- Co-evolution approaches

### 3. **Safety & Alignment**
- Interpretability tools
- Safety constraints in fitness functions
- Robustness testing
- Alignment metrics

### 4. **Real-World Applications**
- Domain-specific fitness functions
- Practical benchmarks
- Integration examples
- Deployment strategies

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to contributor-only discussions
- Given credit in any publications

## ❓ Questions?

- **Discord**: [Join our server](https://discord.gg/vCbefGjhES)
- **GitHub Issues**: For questions and discussions
- **GitHub Discussions**: Coming soon

---

**Thank you for contributing to the future of AGI! 🚀**