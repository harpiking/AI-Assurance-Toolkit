# AI Assurance Toolkit
AI Assurance Toolkit is a lightweight Python package for evaluating machine learning model reliability, performance, and deployment readiness.

# What it does

The toolkit evaluates a trained machine learning model against a labeled test dataset and generates a structured performance report.

# Metrics included

- Accuracy
- Precision
- Recall
- F1 score
- False positive rate
- False negative rate
- AUC-ROC
- Calibration / Brier score
- Per-class metrics
- Plain-English deployment signal

# Installation

```bash
pip install ai-assurance-toolkit
```

# Quick start

```bash
ai-assurance evaluate \
  --model model.pkl \
  --dataset test_data.csv \
  --target credit_risk \
  --model-name "German Credit Risk Classifier"
```

# Example

Generate a test model and sample dataset:

```bash
python examples/setup_test_model.py
```

Then run the evaluator:
```bash
ai-assurance evaluate \
  --model model.pkl \
  --dataset test_data.csv \
  --target credit_risk \
  --model-name "German Credit Risk Classifier"
```

# Output
The package creates:

```text
module_a_outputs/performance_report.json
```

# Python usage
```python
from ai_assurance_toolkit import run_performance_evaluation
```

# License
Apache License 2.0.
