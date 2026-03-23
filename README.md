# Email Spam Detection

A machine learning pipeline that classifies emails as **spam** or **ham** (not spam) using TF-IDF features and classical ML models. Achieves **99.3% accuracy** and **0.994 F1 score** on a held-out test set.

## Dataset

- **83,448 emails** (53% spam, 47% ham)
- Two columns: `label` (1 = spam, 0 = ham) and `text` (raw email content)
- Source: `data/data.csv`

## Approach

1. **Text preprocessing** — lowercase, normalize URLs/emails/numbers, remove punctuation
2. **TF-IDF vectorization** — unigrams + bigrams, 120K features, sublinear TF scaling
3. **Model comparison** — Logistic Regression, SGD Classifier, Multinomial Naive Bayes
4. **Evaluation** — accuracy, F1, ROC AUC, confusion matrix, 5-fold cross-validation

## Results

| Model               | Accuracy | F1     | ROC AUC |
|---------------------|----------|--------|---------|
| Logistic Regression | 0.9884   | 0.9891 | 0.9986  |
| **SGD (log loss)**  | **0.9934** | **0.9938** | **0.9988** |
| Multinomial NB      | 0.9760   | 0.9768 | 0.9978  |

5-fold CV F1: **0.9915 +/- 0.0008**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the notebook (recommended)

```bash
source venv/bin/activate
jupyter notebook spam_detector.ipynb
```

### Load the trained model

```python
import joblib

pipeline = joblib.load("outputs/spam_detector_pipeline.joblib")
prediction = pipeline.predict(["Congratulations! You won a free prize!"])
probability = pipeline.predict_proba(["Congratulations! You won a free prize!"])[:, 1]
```

## Project Structure

```
├── data/
│   └── data.csv               # Dataset
├── spam_detector.ipynb        # Notebook with EDA, training, and evaluation
├── requirements.txt           # Python dependencies
├── outputs/
│   └── spam_detector_pipeline.joblib   # Trained model
└── venv/                      # Virtual environment
```
