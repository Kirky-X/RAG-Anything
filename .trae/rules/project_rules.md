# Python Project Rules & Guidelines

## 1. Code Style & Formatting

### 1.1 Mandatory Tooling
- Follow **PEP 8** coding standards.
- Use **black** for automatic code formatting.
- CI pipeline must include `black --check` and `flake8` checks.
- Recommended: Configure **isort** for automatic import sorting.

### 1.2 Basic Formatting Rules
- **Indentation**: 4 spaces, **NO Tabs**.
- **Line Length**: Maximum 79 characters per line; 72 characters for comments or docstrings.
- **Blank Lines**: 2 blank lines around top-level definitions (functions/classes); 1 blank line between method definitions.
- **Spaces**: Spaces around operators and after commas; no spaces for function calls.

```python
# Correct
def calculate_total(price, quantity):
    total = price * quantity + tax  # Space around operators
    return total

# Incorrect
def calculate_total(price,quantity):
    total=price*quantity+ tax
    return total
```

## 2. Naming Conventions

| Element Type | Naming Style | Example |
| :--- | :--- | :--- |
| Variables, Functions, Methods, Modules | snake_case | `user_id`, `calculate_total` |
| Classes | PascalCase | `UserInfo`, `OrderStatus` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRY`, `CONFIG_PATH` |
| Private Members | Single underscore prefix (`_private`) | `_internal_data` |
| Protected Members | Single underscore prefix (`_protected`) | `_protected_method` |
| Special Methods | Double underscore prefix/suffix (`__init__`) | `__str__`, `__call__` |

**Special Rules:**
- Boolean variables should imply a question/condition (e.g., `is_active`, `has_permission`).
- Avoid single-character variable names (except for loop counters).
- Package/Module names should be short, all lowercase, and may use underscores.

## 3. Code Organization & Architecture

### 3.1 Directory Structure
```plaintext
project/
├── src/                     # Source code
│   ├── __init__.py
│   ├── models/              # Data models
│   ├── services/            # Business logic
│   ├── api/                 # API interfaces
│   └── utils/               # Utility functions
├── tests/                   # Test code
├── docs/                    # Documentation
├── pyproject.toml           # Project configuration
└── requirements.txt         # Dependencies
```

### 3.2 Import Rules
- **Order**: Standard Library → Third-Party Libraries → Local Modules (separated by a blank line).
```python
import os  # Standard Library
from typing import List

import requests  # Third-Party Library
from pydantic import BaseModel

from .models import User  # Local Module
from .utils import helper
```
- Avoid `from module import *` (wildcard imports).
- Prefer absolute imports; relative imports are allowed only within packages.

### 3.3 Modern Project Configuration
- Use `pyproject.toml` instead of `setup.py`.
- Use **poetry** or **uv** for dependency and virtual environment management.

## 4. Type Hints
- **Mandatory Type Hints**: All function parameters and return values must be typed.
- Use **mypy** for static type checking.

```python
from typing import List, Dict, Optional, Union

def process_data(
    users: List[Dict[str, Union[str, int]]],
    threshold: Optional[float] = None
) -> Dict[str, int]:
    """Process user data and return statistics."""
    result: Dict[str, int] = {}
    # ... implementation
    return result
```
- Use `TypeAlias` for complex types to enhance readability.
- Use `typing.NamedTuple` or `dataclasses` instead of simple classes.

## 5. Error Handling
- **Exceptions over Return Codes**: Use exceptions for error handling.
- **Specific Exceptions**: Catch specific exceptions; avoid bare `except:`. At minimum use `except Exception:`.
- Custom exceptions should inherit from `Exception`.

```python
# Correct
def fetch_data(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.Timeout:
        logger.error("Request timed out")
        raise  # Re-raise
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise CustomAPIError(f"Unable to fetch data: {e}")
```
- Use the `logging` module; **disable `print()`**.
- Define a clear exception hierarchy in library code.

## 6. Performance & Resource Management

### 6.1 Efficient Data Structures
- Prefer built-in structures (`dict`, `list`, `set`).
- Use generator expressions instead of list comprehensions for large datasets.

```python
# Memory friendly
sum(x * x for x in range(10**6))
```

### 6.2 Resource Management
- Use the `with` statement to manage files and network connections.
- Use `contextlib.contextmanager` when implementing context managers.

### 6.3 Performance Pitfalls
- Use `join()` for string concatenation instead of `+`.
- Avoid accessing global variables inside loops.
- Use `functools.lru_cache` to cache results of repeated calculations.

## 7. Testing Strategy

### 7.1 Test Pyramid
```plaintext
tests/
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── conftest.py              # pytest configuration
└── fixtures/                # Test fixtures
```
- **Unit Tests**: Cover all public functions and boundary conditions.
- **Integration Tests**: Test interactions between components.
- Test filenames and functions must start with `test_`.

### 7.2 Toolchain
- Use **pytest** as the testing framework.
- Use **pytest-cov** to check coverage (Requirement: ≥ 80%).
- Use **pytest-mock** or `unittest.mock` for mocking.

```python
# Example: using pytest
def test_calculate_total_with_tax():
    # Given
    price = 100
    quantity = 2

    # When
    result = calculate_total(price, quantity)

    # Then
    assert result == 220
    assert result > 0
```

## 8. Documentation Standards

### 8.1 Docstrings
- **All public modules, classes, and functions must have docstrings.**
- Use **Google Style** or **NumPy Style**. Recommended: use **Sphinx** for generation.

```python
def calculate_total(price: float, quantity: int, tax_rate: float = 0.1) -> float:
    """Calculate total price including tax.

    Args:
        price: Unit price.
        quantity: Quantity.
        tax_rate: Tax rate (default 10%).

    Returns:
        Total price (including tax).

    Raises:
        ValueError: If price or quantity is negative.
    """
    if price < 0 or quantity < 0:
        raise ValueError("Price and quantity must be non-negative")
    return price * quantity * (1 + tax_rate)
```

### 8.2 Code Comments
- Add inline comments for complex algorithms or business logic.
- Use `# FIXME:` for temporary solutions and `# TODO:` for pending tasks.

## 9. Modern Python Practices

### 9.1 Python 3.8+ Features
- Use **f-strings** for string formatting.
- Use the **walrus operator (`:=`)** to simplify loops and conditions.
- Use `dataclasses` instead of simple classes.

### 9.2 Asynchronous Programming
- Use `async/await` for I/O-bound tasks.
- Prefer **httpx** over synchronous `requests`.
- Async function names should have an `_async` suffix (optional but recommended for clarity).

```python
async def fetch_data_async(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

### 9.3 Virtual Environments & Dependencies
- **Must use virtual environments** (`venv`, `conda`, `poetry`).
- Separate development dependencies from production dependencies.
- Use `pip-tools` or `poetry` to lock dependency versions.

## 10. Security & Production Deployment

### 10.1 Security Best Practices
- Use the `secrets` module for sensitive information.
- **Validate user input** to prevent SQL injection and XSS.
- Use **bandit** to scan for security vulnerabilities.

### 10.2 Logging
- Use the `logging` module with appropriate levels (DEBUG/INFO/WARNING/ERROR).
- Use structured logging (e.g., `structlog`) in production.

```python
import logging

logger = logging.getLogger(__name__)

def process_order(order_id: int):
    try:
        logger.info(f"Start processing order {order_id}")
        # ... business logic
    except Exception as e:
        logger.error(f"Order {order_id} failed", exc_info=True)
        raise
```
