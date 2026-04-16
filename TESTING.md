# Testing Guide

This document explains how to run backward compatibility tests to ensure that changes to the codebase don't break existing functionality.

## Overview

The test suite covers the entire codebase with backward compatibility tests:

- **Models**: `SimpleFCLateFusion`, `SimpleFCFourier`, `ResNet18LateFusion`
- **Utilities**: Configuration loading/parsing, random seeding, device selection
- **Data Pipeline**: `ArithmeticDataset`, `FixedSampleDataset`, dataloader creation

### Test Data

Tests use actual MNIST training data from torchvision (will auto-download on first run). This ensures backward compatibility is tested against real data distributions, not mock data.

## Setup

### 1. Install Test Dependencies

Install the test dependencies listed in `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

This installs `pytest` and other testing tools.

### 2. Verify Installation

Check that `pytest` is installed:

```bash
pytest --version
```

## Running Tests

### Run All Tests

Execute all tests in the test suite:

```bash
pytest tests/
```

**Note**: First test run will download MNIST data (≈11 MB) to `./data/MNIST/raw/`. This is cached for subsequent runs.

### Run Tests with Verbose Output

Show detailed output for each test:

```bash
pytest tests/ -v
```

### Run a Specific Test File

Test only models:

```bash
pytest tests/test_models.py -v
```

Test only utilities:

```bash
pytest tests/test_utils.py -v
```

Test only data pipeline:

```bash
pytest tests/test_data_pipeline.py -v
```

### Run a Specific Test Class

Test only the ResNet18LateFusion model:

```bash
pytest tests/test_models.py::TestResNet18LateFusion -v
```

Test only configuration parsing:

```bash
pytest tests/test_utils.py::TestExperimentConfig -v
```

### Run a Specific Test Function

Test a single function:

```bash
pytest tests/test_models.py::TestSimpleFCLateFusion::test_model_forward_pass -v
```

### Run Tests with Coverage Report

Generate a code coverage report to see which lines are tested:

```bash
pytest tests/ --cov=models --cov=utils --cov=data_pipeline --cov-report=html
```

This creates an HTML coverage report in `htmlcov/index.html`.

### Run Tests in Watch Mode (Requires pytest-watch)

Install `pytest-watch`:

```bash
pip install pytest-watch
```

Then run:

```bash
ptw tests/
```

This re-runs tests automatically whenever you save a file.

## Test Structure

### Test Files

- `tests/conftest.py`: Shared fixtures (dummy datasets, temporary directories, batch data)
- `tests/test_models.py`: Model architecture and forward/backward pass tests
- `tests/test_utils.py`: Configuration parsing and utility function tests
- `tests/test_data_pipeline.py`: Dataset and dataloader tests

### Test Fixtures

Available pytest fixtures defined in `conftest.py`:

- `dummy_mnist_dataset`: A lightweight dummy MNIST dataset (no real download)
- `temp_data_dir`: A temporary directory for test files
- `device`: The appropriate compute device (CPU for consistency)
- `batch_data`: Sample batch tensors for model testing

## Common Test Scenarios for Teammates

### After Adding a New Model

1. Add tests to `tests/test_models.py` in a new `TestYourModel` class
2. Run: `pytest tests/test_models.py -v`
3. Ensure all tests pass before merging

### After Modifying Configuration

1. Run configuration tests: `pytest tests/test_utils.py::TestExperimentConfig -v`
2. Check roundtrip serialization: `pytest tests/test_utils.py::TestExperimentConfig::test_from_dict_roundtrip -v`

### After Changing the Data Pipeline

1. Run data pipeline tests: `pytest tests/test_data_pipeline.py -v`
2. Specifically check dataset shapes: `pytest tests/test_data_pipeline.py::TestArithmeticDataset::test_dataset_getitem_tensor_types -v`

### Before Committing

Run the full test suite to ensure nothing is broken:

```bash
pytest tests/ -v
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. The basic command for CI is:

```bash
pytest tests/ --tb=short
```

The `--tb=short` flag provides concise traceback format suitable for CI logs.

## Troubleshooting

### Tests Fail with Import Errors

Make sure you're running tests from the project root:

```bash
cd /path/to/mnist-math
pytest tests/
```

### CUDA/MPS Device Issues

Tests use CPU by default for consistency. If you want to test GPU code:

1. Modify `tests/conftest.py` to change the `device` fixture
2. Run with: `pytest tests/ -k "test_name"`

### Slow Test Execution

Tests should complete in seconds. If they're slow:

1. Check disk I/O (avoid network in tests)
2. Check for `time.sleep()` calls in test code
3. Use `pytest -n auto` for parallel execution (requires `pytest-xdist`)

## Adding New Tests

When adding features, follow this pattern:

```python
def test_new_feature_is_backward_compatible() -> None:
    """Test that new_feature doesn't break existing code."""
    # Setup
    component = MyComponent()
    
    # Execute
    result = component.new_method()
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

Key principles:

1. **Test backward compatibility**: Ensure new features don't break existing APIs
2. **Test edge cases**: Empty inputs, boundary values, etc.
3. **Test determinism**: Models in eval mode should be reproducible
4. **Test device movement**: Code should work on CPU and (where applicable) GPU

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Guide](https://pytorch.org/docs/stable/testing.html)
- [Test Fixtures](https://docs.pytest.org/en/stable/fixture.html)
