# Model Registry System

A system for tracking, comparing, and managing model versions for text sentiment analysis.

## Overview

The model registry system allows you to:

- Register and track multiple model versions
- Store model metadata, metrics, and hyperparameters
- Compare models based on different evaluation metrics
- Designate an active model for production use
- Find the best model based on specific metrics

## Components

1. **Model Registry Class** (`scripts/model_training/model_registry.py`)
   - Core class that handles model registration and tracking
   - Persists model information in a JSON registry file

2. **Training Scripts**
   - `hyperparameter_tester.py` - Test different hyperparameter combinations
   - `train_with_params.py` - Train a model using parameters from a file
   - `simple_train.py` - Simplified training script for demonstration

3. **Utility Scripts**
   - `demo_registry.py` - Demonstrates registry functionality
   - `compare_models.py` - Compares models based on metrics

## Directory Structure

```
models/
├── parameters/             # Parameter files for model training
│   ├── default_params.json
│   └── optimized_params.json
├── versions/               # Individual model versions
│   ├── v1/
│   ├── v2/
│   └── ...
└── registry.json           # Registry file storing all model metadata
```

## Usage

### Training a Model with Parameters

```bash
python scripts/model_training/simple_train.py models/parameters/default_params.json --version my_model_v1
```

This will:
1. Load parameters from the specified JSON file
2. Train a model using those parameters
3. Register the model in the registry

### Comparing Models

```bash
python scripts/model_training/compare_models.py --chart --output models/comparison.png
```

This will:
1. Load all models from the registry
2. Generate a comparison table
3. Create a chart comparing model metrics
4. Save the chart to the specified output file

You can also compare specific models and metrics:

```bash
python scripts/model_training/compare_models.py --versions v1 v2 --metrics accuracy f1
```

### Testing Hyperparameters

```bash
python scripts/model_training/hyperparameter_tester.py
```

This will:
1. Test different combinations of hyperparameters
2. Train models for each combination
3. Register all models in the registry
4. Generate a comparison report
5. Set the best-performing model as active

## Parameter Files

Parameter files are JSON files that specify hyperparameters for model training. Example:

```json
{
  "batch_size": 16,
  "epochs": 3,
  "learning_rate": 5e-5,
  "max_length": 256,
  "weight_decay": 0.01,
  "warmup_steps": 100,
  "logging_steps": 50,
  "metric_for_best_model": "accuracy",
  "sample_size": 1000,
  "model_type": "distilbert"
}
```

## Registry File

The registry file (`models/registry.json`) contains information about all registered models:

```json
{
  "models": {
    "v1": {
      "version": "v1",
      "path": "models/versions/v1",
      "metrics": {
        "accuracy": 0.82,
        "f1": 0.80
      },
      "hyperparameters": {
        "batch_size": 16,
        "epochs": 3
      },
      "description": "Baseline model",
      "registered_at": "2024-05-01T10:00:00.000000"
    }
  },
  "active_model": "v1",
  "last_updated": "2024-05-01T10:00:00.000000"
}
```

## API Reference

### ModelRegistry Class

```python
# Initialize the registry
from model_training.model_registry import ModelRegistry
registry = ModelRegistry()

# Register a model
registry.register_model(
    model_path="models/versions/v1",
    version="v1",
    metrics={"accuracy": 0.82, "f1": 0.84},
    hyperparameters={"batch_size": 16, "epochs": 3},
    description="First model version"
)

# List all models
models = registry.list_models()

# Get the active model
active_model = registry.get_active_model()

# Get the best model by a specific metric
best_model = registry.get_best_model(metric="accuracy")

# Compare models
comparison = registry.compare_models(
    versions=["v1", "v2"],
    metrics=["accuracy", "f1"]
)
``` 