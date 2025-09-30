
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
language: en
license: apache-2.0
tags:
- xgboost
- machine-learning
- classification
- cybersecurity
- phishing-detection
datasets:
- custom
metrics:
- accuracy
- precision
- recall
- f1
---

# XGBoost Phishing Detection Models

## Model Description

XGBoost models trained for phishing detection using URL and HTML content features.

This model is trained using XGBoost for binary classification tasks.

## Model Architecture

- **Model Type**: XGBoost Classifier
- **Framework**: XGBoost
- **Task**: Binary Classification

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(repo_id="th1enq/xgboost_checkpoint", filename="xgboost phishing detection models.joblib")

# Load the model
model = joblib.load(model_path)

# Make predictions
predictions = model.predict(X_test)
```

## Training

The model was trained using the XGBoost library with the following approach:
- Feature extraction from URLs/HTML content
- Binary classification (legitimate vs phishing)
- Cross-validation for model evaluation

## Files

- `xgboost phishing detection models.joblib`: The trained XGBoost model
- `features.py`: Feature extraction functions
- `URLFeatureExtraction.py`: URL-specific feature extraction

## License

This model is released under the Apache 2.0 License.

Output:
{
    "extracted_code": "import joblib\nfrom huggingface_hub import hf_hub_download\n\nmodel_path = hf_hub_download(repo_id=\"th1enq/xgboost_checkpoint\", filename=\"xgboost phishing detection models.joblib\")\n\nmodel = joblib.load(model_path)\n\npredictions = model.predict(X_test)"
}
