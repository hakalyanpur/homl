# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a "Hands-On Machine Learning" (HOML) study workspace based on Aurélien Géron's book. The project uses Jupyter notebooks to work through ML concepts and exercises.

## Environment Setup

```bash
# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

## Project Structure

- `notebooks/` - Jupyter notebooks following the book chapters
- `data/` - Placeholder for downloaded datasets (notebooks download data as needed)
- `models/` - Placeholder for saved trained models
- `scripts/` - Placeholder for standalone Python scripts
- `venv/` - Python 3.11 virtual environment

## Key Libraries

- **ML/Data Science**: scikit-learn, pandas, numpy, scipy
- **Deep Learning**: PyTorch (torch, torchvision, torchaudio)
- **Visualization**: matplotlib, seaborn
- **Development**: Jupyter Lab/Notebook

## Notebook Conventions

The notebooks download datasets on first run (e.g., California housing data from `github.com/ageron/data`). Datasets are extracted to `datasets/` within the working directory.

Version checks at notebook start:
- Python >= 3.10
- scikit-learn >= 1.6.1

## ML Concepts & Notes

### pandas vs scikit-learn

| Aspect | pandas | scikit-learn |
|--------|--------|--------------|
| **Purpose** | Data manipulation & analysis | Machine learning algorithms |
| **Core concept** | DataFrames (tables) | Estimators (fit/transform/predict) |
| **Remembers state** | No | Yes (stores learned parameters) |
| **Primary use** | Data wrangling | Model building |

**Rule of thumb:**
- Exploration & basic cleaning → pandas
- Anything that must be consistent between train/test → sklearn

### Encoding Categorical Variables

#### OrdinalEncoder
- Assigns integers based on **alphabetical order**
- Use when categories have a **natural order** (e.g., poor → fair → good → excellent)
- Problem: Model may interpret numbers as having magnitude relationship

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoded = encoder.fit_transform(df[["column"]])  # Note: requires 2D input
```

#### OneHotEncoder
- Creates binary columns for each category
- Use for **nominal categories** (no inherent order)
- Each category is equally different from others

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[["column"]])
```

#### pd.get_dummies()
- Pandas alternative for one-hot encoding
- Simpler syntax but **doesn't remember categories**
- Use for quick exploration, not production pipelines

```python
pd.get_dummies(df["column"])
```

### fit_transform Explained

`fit_transform()` combines two steps:
1. **fit** - Learn parameters from data (e.g., unique categories, median values)
2. **transform** - Apply the learned transformation

### Feature Name Tracking

When you fit a sklearn estimator with a DataFrame, it remembers the column names:

```python
scaler = StandardScaler()
scaler.fit(df)
print(scaler.feature_names_in_)  # ['age', 'income']
```

For transformers that change columns (like OneHotEncoder), get output names:

```python
encoder.get_feature_names_out()
# ['ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', ...]

# Convert output back to DataFrame
encoded_df = pd.DataFrame(
    encoder.transform(df).toarray(),
    columns=encoder.get_feature_names_out()
)
```

| Attribute/Method | Purpose |
|------------------|---------|
| `feature_names_in_` | Column names that went **in** during fit |
| `get_feature_names_out()` | Column names that come **out** after transform |

### ML Pipeline

A pipeline chains preprocessing steps with model training:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Why pipelines matter:** Prevent data leakage and ensure reproducibility.

### Data Cleaning: When to Use What

| Task | Tool |
|------|------|
| Remove duplicates, drop columns, filter rows | pandas |
| Fill missing values (production) | sklearn SimpleImputer |
| Encode categories (production) | sklearn encoders |
| Quick exploration | pandas |