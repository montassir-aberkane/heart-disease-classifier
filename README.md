# Heart Disease Classifier

Binary classification of heart disease presence using the UCI Cleveland dataset.

## Dataset
- Source: UCI Machine Learning Repository — Heart Disease Dataset
- 303 samples, 13 features, 6 missing rows dropped → 297 clean samples
- License: CC BY 4.0

## Models
- Logistic Regression (C=0.1)
- Random Forest (n_estimators=100, max_depth=5)

## Results
| Model | Val F1 | Val AUROC |
|-------|--------|-----------|
| Logistic Regression | 0.7027 | 0.9008 |
| Random Forest | 0.7027 | 0.8909 |
| **RF (Test)** | **0.8780** | **0.9563** |

## Reproducibility
All random seeds set to 42. Run with:
```
python main.py
```

## Requirements
```
pip install pandas scikit-learn matplotlib seaborn numpy jupyter
```