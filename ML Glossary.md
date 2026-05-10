> A living glossary — quick-lookup definitions for ML terms encountered in each chapter.
> For deeper explanations with diagrams and examples, see [Concepts.md](Concepts.md).

---

# Phase 1: Data & Preprocessing

---

### Splitting & Sampling

**Training Set**
The data the model learns from (typically 80% of the dataset).

**Test Set**
Data held back for final, one-time evaluation (typically 20%). Never used for tuning.

**Validation Set**
Data used to compare models and tune hyperparameters. Created automatically by cross-validation.

**Stratified Sampling**
Splitting data so each subset mirrors the original distribution of a key feature.
> *Ensures train/test both have ~35% high-income districts, not 38%/29%.*

**Data Leakage**
When information from outside the training set leaks into model training, giving unrealistically good results.
> *Example: fitting a scaler on the full dataset before splitting — test data statistics influence training.*

---

### Cleaning & Imputation

**Missing Values**
Data points where a feature has no value (e.g., `total_bedrooms` is NaN).

**SimpleImputer**
Fills missing values using a learned strategy.
```python
SimpleImputer(strategy="median")  # fills blanks with the column median
```

**Imputation Strategies:** `"median"` · `"mean"` · `"most_frequent"` · `"constant"`

**fit()** — Learns/computes parameters from training data (read-only).
**transform()** — Applies what was learned to data.
**fit_transform()** — Calls fit() then transform() in one step. *Only use on training data.*

---

### Encoding Categorical Features

**Categorical Feature**
A feature with a limited set of text/label values (e.g., `ocean_proximity`: "NEAR BAY", "INLAND").

**OrdinalEncoder** — Maps categories to integers, implies an ordering.
> Use when categories have a natural order: `"bad" → 0, "average" → 1, "good" → 2`

**OneHotEncoder** — Creates one binary column per category, no false ordering.
> Use for nominal categories: `"INLAND" → [1,0,0], "NEAR BAY" → [0,1,0]`

**Sparse Matrix** — Stores only non-zero values; memory efficient. OneHotEncoder output is sparse.
**Dense Matrix** — Standard matrix storing every value including zeros.

---

### Feature Scaling

**Feature Scaling**
Adjusting features to similar ranges so no single feature dominates the model.

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **Min-Max Scaling** | Rescales to a fixed range (default [0, 1]) | When you need bounded values |
| **Standardization** | Centers to mean=0, std=1 | General-purpose; less sensitive to outliers |

**Heavy-Tailed Distribution** — Most values clustered low, extreme outliers stretch far out.
**Log Transform** — Compresses large values, spreads out small ones. Makes skewed distributions more bell-shaped.
```python
np.log1p(income)  # log(1 + x) handles zeros safely
```

**Breaking Sparsity** — Subtracting the mean turns zeros into non-zeros, destroying sparse structure.
> *Fix: use `StandardScaler(with_mean=False)` for sparse data.*

---

### Feature Engineering

**Feature Engineering** — Creating new features from raw columns to make signal easier to learn.

| Technique | What It Does | Example |
|-----------|-------------|---------|
| **Ratio Features** | Divides one feature by another | `bedrooms_ratio = total_bedrooms / total_rooms` |
| **Bucketizing** | Splits continuous values into discrete bins | Age → "child", "adult", "senior" |
| **Cluster Similarity** | KMeans + RBF kernel turns coordinates into cluster proximity scores | lat/lng → 10 similarity scores |

**RBF Kernel (Radial Basis Function)** — Measures similarity to a fixed point; returns ~1.0 at the centroid, fades to ~0 with distance.

---

### Target Scaling

**Target Scaling** — Transforming the output variable (y) to a normalized range before training.
**Inverse Transform** — Converting scaled predictions back to original units.

```python
# Handles both automatically:
TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
```

---

# Phase 2: Model Exploration

---

### Error Metrics

**RMSE (Root Mean Squared Error)**
Average prediction error in the target's units. Penalizes large errors heavily due to squaring.
> *RMSE = \$66k on homes worth ~\$180k → predictions are typically ~37% off.*

**MAE (Mean Absolute Error)**
Average of absolute errors. Treats all errors equally regardless of size.
> *Use when large errors aren't disproportionately bad.*

**Variance** — Average of squared differences from the mean. Measures how spread out values are.
**Standard Deviation** — Square root of variance; same units as the data.

---

### Cross-Validation

**Cross-Validation (K-Fold)**
Split training data into K folds. Train on K-1, validate on 1, repeat K times.

- **CV Mean** — Average score across folds → expected performance on unseen data
- **CV Std** — Standard deviation across folds → how stable the estimate is

> *Low std (\$1k) = trustworthy. High std (\$8k) = performance varies wildly by fold.*

**`cross_val_score`** — Runs K fresh fit/predict cycles internally. Ignores any prior `.fit()` you called. Evaluates the *approach*, not a specific fitted model.

---

### Bias-Variance Tradeoff

```
                    Training Error    CV Error     Diagnosis
                    ──────────────    ────────     ─────────
Underfitting:          ~$70k           ~$70k      Both high, small gap (too simple)
Overfitting:            ~$0            ~$66k      Low train, high CV (memorizing)
Good Fit:              ~$15k           ~$18k      Both low, small gap
```

**Overfitting (High Variance)** — Model memorizes training noise. Fix: more data, regularization, limit complexity.
**Underfitting (High Bias)** — Model too simple to capture patterns. Fix: more complex model, add features.
**Regularization** — Constraining a model to reduce overfitting (e.g., limiting tree depth, L2 penalty).

---

### Model Types

**Linear Regression**
Fits one equation across all data: `y = w1·x1 + w2·x2 + ... + bias`
> *Fast and interpretable, but assumes linear relationships. Prone to underfitting.*

**Decision Tree**
Learns if/else rules by splitting data into branches. Each leaf holds the average of its training rows.
> *Captures non-linear patterns, but prone to overfitting (can memorize training data perfectly).*

**Ensemble**
Combines multiple models so individual weaknesses cancel out.
> *Like averaging 100 people's guesses at a jellybean count — individual errors cancel, truth emerges.*

---

# Phase 3: Hyperparameter Tuning

---

### Core Concepts

**Model Parameter** — Learned from data during training. You never set it.
> *Weights in regression, split thresholds in trees.*

**Hyperparameter** — Set by you before training. Controls *how* the model learns.
> *`n_estimators=100`, `max_depth=10`, `n_clusters=15`*

---

### Ensemble Methods

| Method | Strategy | Example |
|--------|----------|---------|
| **Bagging** | Same model on random data subsets, average results | Random Forest |
| **Boosting** | Sequential models, each fixing the last one's mistakes | Gradient Boosting, XGBoost |
| **Stacking** | Different model types + a meta-model on their outputs | Linear + Tree + SVM |

**Random Forest** — Bagging applied to Decision Trees + random feature subsets at each split.
Forces diversity between trees; averaging cancels noise while preserving real signal.

---

### Random Forest Hyperparameters

```
Parameter            What It Controls                           Default
─────────            ────────────────                           ───────
n_estimators         Number of trees                            100 (diminishing returns ~300)
max_features         Features available per split               1.0 for regression
max_depth            How deep each tree can grow                None (unlimited)
min_samples_leaf     Minimum rows required in a leaf            1
```

> *`max_features=6` means "at each split, randomly pick 6 features to consider" — not "only use 6 features total." Over 100 trees × hundreds of splits, every feature participates.*

---

### Search Strategies

**GridSearchCV** — Tries every combination from explicit value lists, evaluated with CV.
> *Best for: fine-tuning a narrow range you've already identified.*

**RandomizedSearchCV** — Samples random combinations from distributions, evaluated with CV.
> *Best for: broad initial exploration of wide ranges.*

```
Grid Search:        "try [5, 8, 10, 15]"     → exactly those 4
Randomized Search:  "try anything from 3–49"  → 10 random picks (n_iter=10)
```

**Parameter Path** — Double-underscore syntax to reach into nested pipeline components:
```
"preprocessing__geo__n_clusters"
 └── Pipeline step   └── ColumnTransformer sub-step   └── the param
```

**Multiple Grids** — Pass a list of dicts to GridSearchCV. Each dict is explored independently, avoiding nonsensical combinations.

---

# Phase 4: Model Analysis

---

### Feature Importances

**Feature Importance** — Total reduction in prediction error from all splits using that feature, averaged across all trees. All importances sum to 1.0.

Three ways to use importances:

1. **Sanity check** — Does the model agree with domain knowledge? Income *should* rank highest for housing prices.
2. **Feature selection** — Drop features with ~0 importance to simplify the model.
3. **Debugging** — If an expected feature scores low, it might be redundant, poorly engineered, or just not predictive.

**Correlated Feature Trap** — Two correlated features split importance between them, making both appear less important than they really are.
> *`total_rooms` and `total_bedrooms` share importance — neither looks critical alone, but together they are.*

---

# Phase 5: Final Evaluation

---

### Test Set Evaluation

**The Rule:** evaluate on the test set **once**. If you keep testing and adjusting, it becomes another training set.

```
Cross-validation  = practice exams     → you adjust strategy based on results
Test set          = the final exam     → taken once, no adjustments allowed
```

**Confidence Interval** — Quantifies uncertainty of a single RMSE number.
> *95% CI: [\$39k, \$44k] — more test data → narrower interval → more certainty.*

**Interpreting results:**

| CV RMSE | Test RMSE | Meaning |
|---------|-----------|---------|
| \$43.6k | \$41.5k | Good — model generalizes well |
| \$43.6k | \$55k | Bad — overfit during tuning |
| \$43.6k | \$38k | Lucky test split — trust CV more |

---

# Scikit-Learn API Patterns

---

**Estimator** — Any object that learns from data (has `fit()`).
**Transformer** — An estimator that also transforms data (has `transform()`).
**Predictor** — An estimator that makes predictions (has `predict()`).

**Pipeline** — Chains preprocessing steps + model into a single object.
**ColumnTransformer** — Applies different transformations to different columns in parallel.

```python
# Track feature names through the pipeline:
scaler.feature_names_in_          # columns that went IN during fit
encoder.get_feature_names_out()   # columns that come OUT after transform
```

---

# Statistical Foundations

---

| Concept | Definition | Formula / Note |
|---------|------------|----------------|
| **Mean** | Average of all values | `sum / count` |
| **Variance** | Average of squared deviations from the mean | Measures spread |
| **Standard Deviation** | Square root of variance | `sqrt(variance)` — same units as data |
| **RMSE** | Std dev of prediction errors | `sqrt(mean(errors²))` |
| **Normal Distribution** | Bell curve — most values near the mean | StandardScaler assumes this shape |
| **Uniform Distribution** | Every value in a range is equally likely | `randint(3, 50)` — each int same chance |
| **Arithmetic Mean** | Sum of values divided by count | `(a + b) / 2` for two values |
| **Harmonic Mean** | Reciprocal of the average of reciprocals; dominated by the smaller value | `2·a·b / (a + b)` for two values — used in F1 |

---

# Chapter 3 — Classification

---

### Class Imbalance

**Class Imbalance** — When one class dominates the dataset (e.g., ~90% non-5s, ~10% 5s in the MNIST binary task).
> *Accuracy becomes misleading — a "always predict the majority class" classifier can score very high without learning anything.*

**Majority Class** — The more frequent class.
**Minority Class** — The less frequent class. Usually the one you actually care about detecting.

---

### Confusion Matrix

**Confusion Matrix** — For binary classification, a 2×2 grid tallying the four possible outcomes.

```
                    Predicted
                 Negative    Positive
Actual  Negative    TN          FP
        Positive    FN          TP
```

**True Positive (TP)** — Model said positive, label was positive. Correct.
**True Negative (TN)** — Model said negative, label was negative. Correct.
**False Positive (FP)** — Model said positive, label was negative. *False alarm.*
**False Negative (FN)** — Model said negative, label was positive. *Missed detection.*

> **Mnemonic:** first letter = was the model right (T/F)? Second letter = what did it say (P/N)?

**`cross_val_predict`** — Like `cross_val_score` but returns the out-of-fold *predictions* rather than a score. Feed directly to `confusion_matrix` for leakage-free evaluation on the training set.

---

### Classification Metrics

**Accuracy** — `(TP + TN) / total`. Fraction correct. Misleading on imbalanced data.

**Precision** — `TP / (TP + FP)`. Of everything the model flagged as positive, what fraction really was?
> *Punishes false alarms. High precision = selective model.*

**Recall (Sensitivity, True Positive Rate)** — `TP / (TP + FN)`. Of the actual positives in the data, what fraction did the model catch?
> *Punishes missed detections. High recall = thorough model.*

**Precision-Recall Tradeoff** — Raising one tends to lower the other. Controlled by the decision threshold (the score above which the model predicts "positive"). Pick the operating point that matches your problem's cost structure.

**When to prefer which:**

| Prefer | Example problems |
|--------|------------------|
| **Precision** | Spam filter, criminal conviction, medical intervention with risky side effects |
| **Recall**    | Cancer screening, fraud / security alerts, airport security |

---

### F1 and Fβ

**F1 Score** — `2 · P · R / (P + R)`. The **harmonic mean** of precision and recall. Single summary number rewarding classifiers that are good at **both**.
> *If either P or R is near 0, F1 is near 0 — unlike the arithmetic mean, which can stay around 0.5.*

**Fβ Score** — Generalization of F1: `(1 + β²) · P · R / (β² · P + R)`.
- β > 1 emphasizes **recall** (e.g., F2 for medical screening)
- β < 1 emphasizes **precision**
- β = 1 → F1 (balanced)

---

### Decision Score vs Prediction

**Decision Score** — The raw real-valued output of a linear classifier: `w · x + b`. Magnitude indicates confidence.
**Prediction** — The discrete class label derived from the score: `predict = positive if score > threshold else negative`.
**Decision Threshold** — The cutoff applied to the score. Default is 0 for `SGDClassifier`. Moving it up raises precision and lowers recall; moving it down does the opposite.

`sgd_clf.decision_function(X)` returns raw scores; `sgd_clf.predict(X)` returns thresholded labels.
