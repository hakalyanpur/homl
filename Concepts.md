# Chapter 2 — Concepts & Theory

> **Notebook reference:** `notebooks/02_end_to_end_machine_learning_project.ipynb`
> Each section below maps to concepts used in the notebook's 5 phases.

---

## Stratified Sampling

### The problem with random splits

When you split data into train/test sets randomly, you risk the split not being representative. If `median_income` is the strongest predictor of housing prices (correlation +0.69), a random split might over-represent high-income districts in training and under-represent them in testing — or vice versa.

### How stratified splitting fixes this

Stratified sampling ensures both sets mirror the full dataset's distribution:

```
Full dataset income distribution:
  Low (0-1.5):     4%
  Medium (1.5-3):  30%
  High (3-4.5):    35%
  Very high (4.5+): 31%

Random split might give:
  Training:  Low 3%, Medium 28%, High 38%, Very high 31%  ← skewed!
  Test:      Low 6%, Medium 34%, High 29%, Very high 31%

Stratified split guarantees:
  Training:  Low 4%, Medium 30%, High 35%, Very high 31%  ← proportional
  Test:      Low 4%, Medium 30%, High 35%, Very high 31%  ← proportional
```

### In practice

```python
# Bin continuous income into 5 categories
housing["income_cat"] = pd.cut(housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

# stratify= ensures proportional representation
train, test = train_test_split(housing, test_size=0.2,
    stratify=housing["income_cat"], random_state=42)
```

### When to use it

Use stratified sampling when:
- A feature is **highly predictive** and its distribution matters
- The dataset is **small enough** that random chance could skew the split
- A category is **rare** (e.g., ISLAND ocean proximity — only 5 districts)

---

# Scikit-Learn Preprocessing Pipeline

---

## Without Pipeline

### Training Data

```
RAW TRAINING DATA
   │
   ▼
SimpleImputer.fit()        → learns median (read only)
SimpleImputer.transform()  → fills NaNs with median
   │
   ▼
StandardScaler.fit()       → learns mean + std (read only)
StandardScaler.transform() → rescales using mean + std
   │
   ▼
CLEAN, SCALED TRAINING DATA
```

### Test Data

```
RAW TEST DATA
   │
   ▼
SimpleImputer.transform()  → uses median learned from training
   │
   ▼
StandardScaler.transform() → uses mean + std learned from training
   │
   ▼
CLEAN, SCALED TEST DATA
```

---

## With Pipeline

### Training Data

```
RAW TRAINING DATA
   │
   ▼
┌─────────────────────────────────────┐
│           num_pipeline              │
│                                     │
│  SimpleImputer.fit()                │
│  SimpleImputer.transform()          │
│             │                       │
│             ▼                       │
│  StandardScaler.fit()               │
│  StandardScaler.transform()         │
│                                     │
└─────────────────────────────────────┘
   │
   ▼
CLEAN, SCALED TRAINING DATA
```

### Test Data

```
RAW TEST DATA
   │
   ▼
┌─────────────────────────────────────┐
│           num_pipeline              │
│                                     │
│  SimpleImputer.transform()  only    │
│             │                       │
│             ▼                       │
│  StandardScaler.transform() only    │
│                                     │
└─────────────────────────────────────┘
   │
   ▼
CLEAN, SCALED TEST DATA
```

---

## Key Rules

| | Training Data | Test Data |
|---|---|---|
| `SimpleImputer` | `fit()` + `transform()` | `transform()` only |
| `StandardScaler` | `fit()` + `transform()` | `transform()` only |

> **Pipeline guarantees**: correct step order, no accidental `fit()` on test data, and clean cross-validation with no data leakage.

---

## RMSE (Root Mean Squared Error)

RMSE measures how far predictions are from actual values, in the **same units as the target**.

### The Steps

```
Predictions:  [200k, 300k, 150k]
Actuals:      [250k, 280k, 100k]

1. Errors:          [-50k,  +20k,  +50k]
2. Squared:         [2.5B,  400M,  2.5B]       ← big errors punished harder
3. Mean of squares: 1.8B
4. Square root:     ~$42,426                    ← back to dollars
```

### Why not just average the errors?

Plain average would let positive and negative errors cancel out:
`(-50k + 20k + 50k) / 3 = +6.7k` — looks great, but two predictions were $50k off!

RMSE squares first, so every error counts regardless of direction.

### Why not use Mean Absolute Error (MAE)?

MAE = `(50k + 20k + 50k) / 3 = $40k`. Also valid, but RMSE **punishes large errors more** because of squaring. A single $100k miss hurts RMSE far more than two $50k misses. Choose RMSE when big errors are especially bad (e.g., house pricing).

### How to interpret

RMSE is in target units (dollars), so compare it to the target's scale:
- Median home value ~$180k, RMSE = $66k → predictions are typically ~37% off
- If RMSE were $10k → ~6% off → much better

---

## Standard Deviation

Standard deviation measures **how spread out values are from their average**.

### Intuition

Imagine two classrooms that both scored an average of 80%:

```
Classroom A: [78, 79, 80, 81, 82]  → std = ~1.4   (everyone near 80)
Classroom B: [50, 65, 80, 95, 110] → std = ~21     (wildly different)
```

Same mean, completely different stories. Std tells you the story the mean hides.

### The math (same pattern as RMSE)

```
Values: [78, 79, 80, 81, 82],  mean = 80

1. Differences from mean: [-2, -1, 0, +1, +2]
2. Squared:               [4,  1,  0,  1,  4]
3. Mean of squares:       2.0
4. Square root:           ~1.4
```

Notice this is exactly the same square-then-root pattern as RMSE. In fact, **RMSE is the standard deviation of prediction errors** (with zero assumed as the "mean" to deviate from).

### Std of RMSE (what cross-validation reports)

When we run 10-fold CV, we get 10 RMSE values. The std of those tells us how **stable** the model is:

```
Decision Tree CV scores: [64.6k, 66.4k, 66.2k, 65.9k, 68.1k, ...]
Mean RMSE = $66,574    → "how good is the approach?"
Std RMSE  = $1,047     → "how much does that number wobble across folds?"
```

Low std = trustworthy estimate. High std = the model's performance depends heavily on which data it sees.

---

## Cross-Validation (K-Fold)

Cross-validation gives an **honest estimate** of how a model will perform on unseen data, without touching the test set.

### Why not just use training error?

A model that memorizes the training data gets perfect training scores but fails on new data. Training error tells you how well the model **remembers**, not how well it **generalizes**.

### How 10-fold CV works

```
Full training set (16,512 rows)
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│Fold 1│Fold 2│Fold 3│Fold 4│Fold 5│Fold 6│Fold 7│Fold 8│Fold 9│Fold10│
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

Round 1:  [EVAL] [train] [train] [train] [train] [train] [train] [train] [train] [train]  → RMSE₁
Round 2:  [train] [EVAL] [train] [train] [train] [train] [train] [train] [train] [train]  → RMSE₂
Round 3:  [train] [train] [EVAL] [train] [train] [train] [train] [train] [train] [train]  → RMSE₃
...
Round 10: [train] [train] [train] [train] [train] [train] [train] [train] [train] [EVAL]  → RMSE₁₀
```

Each round: train on 9 folds, evaluate on the 1 held-out fold. Every row gets to be in the eval set exactly once.

### What it tells you

- **Mean of 10 RMSEs** → expected performance on new data
- **Std of 10 RMSEs** → how stable that estimate is
- It evaluates the **approach** (preprocessing + model type), not a specific fitted model

### Important detail

`cross_val_score` does its own `fit()` internally — it ignores any previous `.fit()` you called. The pre-fitted model and the CV evaluation are independent operations:

```python
lin_reg.fit(housing, housing_labels)       # → produces a usable model

cross_val_score(lin_reg, housing, ...)     # → benchmarks the approach
                                           #   (does 10 fresh fit/predict cycles)
```

---

## Overfitting vs Underfitting (Bias-Variance Tradeoff)

### The core tension

- **Underfitting (high bias)**: Model is too simple to capture real patterns in the data
- **Overfitting (high variance)**: Model is so flexible it memorizes noise instead of learning patterns

### Analogy

Imagine studying for an exam:
- **Underfitting**: You only read the chapter title — you get the gist but miss all the details. You do poorly on both practice and real exams.
- **Overfitting**: You memorize every practice question word-for-word — perfect on practice tests, but any rephrased question on the real exam stumps you.
- **Good fit**: You understand the underlying concepts — you handle both familiar and new questions well.

### How to diagnose: training error vs CV error

```
                    Training Error    CV Error        Diagnosis
                    ──────────────    ────────        ─────────
LinearRegression:      ~$70k           $70k       Underfitting (high bias)
                                                   Both errors high and similar.
                                                   Model can't even fit training data.

DecisionTree:            $0            $66k       Overfitting (high variance)
                                                   Training error is perfect but
                                                   CV error is still high.
                                                   Big gap = memorization.

Good model:            ~$15k           $18k       Good fit
                                                   Both errors low, gap is small.
                                                   Model learned real patterns.
```

### Fixing each problem

| Problem | Cause | Fixes |
|---------|-------|-------|
| Underfitting | Model too simple | Use a more complex model, add features, reduce regularization |
| Overfitting | Model too complex | More training data, regularization, limit model complexity (e.g., max_depth), or use ensembles (RandomForest) |

---

## Decision Trees vs Linear Regression

### How each model learns

**Linear Regression** fits a single equation across all data:

```
predicted_value = w₁·income + w₂·age + w₃·cluster_similarity + ... + bias

It finds the best weights (w) to draw one straight hyperplane
through all 16,512 districts. Every district uses the same formula.
```

**Decision Tree** learns a series of if/else rules by splitting data:

```
                    Is median_income < 3.5?
                    /                     \
                 YES                       NO
           Is cluster_2 > 0.7?       Is age > 30?
           /              \           /          \
      $95,000          $152,000   $280,000    $350,000
```

Each leaf holds the average value of training rows that landed there.

### When to use each

| Aspect | Linear Regression | Decision Tree |
|--------|------------------|---------------|
| Captures non-linear patterns | No — draws a flat surface | Yes — creates complex boundaries |
| Interpretable | Weights show feature importance directly | Tree rules are readable but can be huge |
| Risk of overfitting | Low — too constrained | High — can grow until every row has its own leaf |
| Risk of underfitting | High — assumes linear relationships | Low — flexible enough to fit anything |
| Training speed | Very fast (one matrix operation) | Fast (greedy splits) |
| Best used as | Simple baseline to establish a floor | Stepping stone to ensembles (RandomForest) |

---

## Ensemble Learning

### The core idea

A single model has weaknesses — it might overfit, underfit, or be sensitive to specific data points. An **ensemble** combines multiple models so their individual weaknesses cancel out.

### Analogy

Imagine estimating how many jellybeans are in a jar:
- **One person's guess**: might be way off (e.g., 200 when the answer is 500)
- **Average of 100 people's guesses**: surprisingly close to the truth

Each person makes different mistakes — some guess too high, some too low. Averaging smooths out the noise. This is the same principle behind ensembles.

### Types of ensembles

| Method | How it works | Example |
|--------|-------------|---------|
| **Bagging** | Train same model type on random subsets of data, average results | Random Forest |
| **Boosting** | Train models sequentially, each fixing the previous one's mistakes | Gradient Boosting, XGBoost |
| **Stacking** | Train different model types, then train a meta-model on their outputs | Combining Linear + Tree + SVM |

---

## Random Forest

A Random Forest is **bagging applied to Decision Trees** with an extra twist: each tree also sees a random subset of **features** at each split.

### How it works

```
Training Data (16,512 rows, 24 features)
         │
         ├──→ Tree 1: random 60% of rows, random features at each split → prediction₁
         ├──→ Tree 2: random 60% of rows, random features at each split → prediction₂
         ├──→ Tree 3: random 60% of rows, random features at each split → prediction₃
         │    ...
         └──→ Tree 100: random 60% of rows, random features at each split → prediction₁₀₀
                                    │
                                    ▼
                    Final prediction = average(prediction₁ ... prediction₁₀₀)
```

### Why it works — fixing the Decision Tree's overfitting

A single Decision Tree memorizes training data (training error = $0, CV error = $66k). The problem is it learns **noise** — quirks specific to the training set that don't generalize.

Random Forest fixes this with two layers of randomness:

1. **Random rows (bagging)**: Each tree sees a different subset, so each memorizes different noise
2. **Random features**: At each split, the tree can only choose from a random subset of features, so trees make different decisions even on the same data

When you average 100 trees that memorized **different** noise, the noise cancels out — but the **real patterns** (income → price, location → price) show up in all trees and survive the averaging.

```
Tree 1 prediction for district X:  $210,000  (overestimate — learned noise A)
Tree 2 prediction for district X:  $180,000  (underestimate — learned noise B)
Tree 3 prediction for district X:  $195,000  (close)
...
Average of 100 trees:              $198,500  (noise cancels, signal remains)
Actual value:                      $200,000
```

### Key hyperparameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `n_estimators` | Number of trees | 100 — more trees = better but slower, diminishing returns after ~100-300 |
| `max_depth` | How deep each tree can grow | None (unlimited) — can limit to reduce overfitting further |
| `max_features` | Features available per split | `"sqrt"` for classification, `1.0` for regression — lower = more diversity between trees |
| `min_samples_leaf` | Minimum rows in a leaf | 1 — increase to prevent tiny leaves (reduces overfitting) |

### How to read the results

```
                    Training Error    CV Error        Gap       What it means
                    ──────────────    ────────        ───       ─────────────
DecisionTree:            $0            $66k          $66k      Massive overfitting
RandomForest:          ~$18k           $47k          $29k      Still some overfitting,
                                                               but much better on both fronts
```

- Training error is no longer zero — averaging prevents perfect memorization
- CV error drops significantly — the ensemble generalizes better
- The gap is smaller — less overfitting, though still room to improve (via hyperparameter tuning or more data)

---

## Hyperparameter Tuning (Grid Search)

### Parameters vs Hyperparameters

**Parameters** are learned by the model during `fit()` — you never set them manually.
**Hyperparameters** are settings you choose *before* training — they control *how* the model learns.

```
Analogy: Baking a cake
  Parameters       = the exact texture, rise, moisture (the oven figures this out)
  Hyperparameters  = oven temperature, baking time, rack position (YOU choose these)

You can't know the perfect temperature without experimenting.
That's what Grid Search does — it bakes the cake at every temperature
you suggest and picks the one that tastes best.
```

### The problem: too many combinations to try by hand

A Random Forest has several hyperparameters. Our preprocessing pipeline also has tuneable settings (like how many geographic clusters to create). Trying every combination manually is impractical:

```
n_clusters:    [5, 8, 10, 15]     → 4 options  (preprocessing)
max_features:  [4, 6, 8, 10]      → 4 options  (model)
max_depth:     [None, 10, 20]     → 3 options  (model)
                                     ─────────
                                     4 × 4 × 3 = 48 combinations
                                     × 10 folds = 480 training runs
```

### How GridSearchCV works

GridSearchCV automates this: try every combination, evaluate each with cross-validation, pick the winner.

```
STEP 1: List all combinations
═══════════════════════════════
  Combo 1:  n_clusters=5,  max_features=4
  Combo 2:  n_clusters=5,  max_features=6
  Combo 3:  n_clusters=5,  max_features=8
  ...
  Combo 15: n_clusters=15, max_features=10


STEP 2: For each combo, run k-fold CV
══════════════════════════════════════
  Combo 1: n_clusters=5, max_features=4

    Training data (16,512 rows)
    ┌──────────────────┬──────────┐
    │   Train (11,008) │Eval(5504)│  → preprocessing(n_clusters=5)
    └──────────────────┴──────────┘    → 19 features created
                                       → forest(max_features=4)
    Fold 1 → RMSE $46,800               each tree sees 4 of 19 features per split
    Fold 2 → RMSE $46,100
    Fold 3 → RMSE $46,670
    Mean RMSE = $46,523


  Combo 12: n_clusters=15, max_features=6

    Training data (16,512 rows)
    ┌──────────────────┬──────────┐
    │   Train (11,008) │Eval(5504)│  → preprocessing(n_clusters=15)
    └──────────────────┴──────────┘    → 29 features created
                                       → forest(max_features=6)
    Fold 1 → RMSE $43,200               each tree sees 6 of 29 features per split
    Fold 2 → RMSE $43,800
    Fold 3 → RMSE $43,770
    Mean RMSE = $43,590  ★ BEST


STEP 3: Pick the winner
═══════════════════════
  n_clusters=5,  max_features=4  → $46,523
  n_clusters=5,  max_features=6  → $46,826
  n_clusters=8,  max_features=4  → $44,748
  n_clusters=10, max_features=6  → $44,278
  n_clusters=15, max_features=6  → $43,590  ★ WINNER
  n_clusters=15, max_features=10 → $44,682
  ...

  Best params stored in: grid_search.best_params_
  Best model stored in:  grid_search.best_estimator_
```

### Tuning the pipeline, not just the model

The key insight: GridSearchCV can reach **into** the preprocessing pipeline and change how features are created. This is why sklearn pipelines matter for tuning.

```
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),      ← has tuneable settings
    ("random_forest", RandomForestRegressor()),  ← has tuneable settings
])

Parameter path uses double underscores to navigate the nesting:

  "preprocessing__geo__n_clusters"
   └── step name  └── sub-step  └── the actual setting
       in Pipeline    in ColumnTransformer

  "random_forest__max_features"
   └── step name    └── the actual setting
       in Pipeline
```

With `n_clusters=5`, preprocessing creates 19 features (5 cluster similarities).
With `n_clusters=15`, preprocessing creates 29 features (15 cluster similarities).
The **shape of the data changes** depending on the hyperparameter — Grid Search evaluates whether finer geographic detail is worth the extra features.

### Multiple grids (list of dicts)

You can pass a **list** of parameter grids. GridSearchCV tries all combinations from each dict separately, then compares across all of them:

```python
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],    # Grid 1: fewer clusters
     'random_forest__max_features': [4, 6, 8]},       #         fewer features

    {'preprocessing__geo__n_clusters': [10, 15],       # Grid 2: more clusters
     'random_forest__max_features': [6, 8, 10]},      #         more features
]

# Grid 1: 3 × 3 = 9 combos
# Grid 2: 2 × 3 = 6 combos
# Total:  15 combos (not 5 × 4 = 20 from one big grid)
```

This avoids wasting time on combinations that don't make sense (e.g., 5 clusters with max_features=10 — with only 19 total features, 10 is almost half, reducing tree diversity too much).

### max_features: what it actually means

`max_features=6` does NOT mean "only use 6 features for the whole forest." It means **at each split in each tree, randomly pick 6 features to consider:**

```
Tree 1, Split 1: randomly sees [income, cluster_5, age, ratio_1, INLAND, cluster_12]
                 → picks income (best separator at this point)

Tree 1, Split 2: randomly sees [rooms, cluster_2, pop, cluster_8, NEAR BAY, ratio_3]
                 → picks cluster_2 (best separator here)

Tree 2, Split 1: randomly sees [age, cluster_9, ratio_2, ISLAND, pop, cluster_1]
                 → picks age (income wasn't available this time!)
```

Over 100 trees × hundreds of splits, **every feature participates** — but no single feature dominates every tree. This forces diversity: some trees learn "income drives price," others discover "location near the bay matters most." Averaging these diverse perspectives gives a stronger prediction than 100 identical trees.

### The validation chain

After tuning, the CV scores are no longer fully unbiased — we **chose** the best combo based on them. The test set (held out since the very beginning) gives the final honest evaluation:

```
Cross-validation     →  pick best hyperparameters ($43,590)
                              ↓
Train final model on full training set with best params
                              ↓
Evaluate ONCE on test set  →  honest, final number
                              (might be slightly worse than $43,590)
```

---

## Randomized Search

### Why Grid Search isn't always enough

Grid Search only tries values YOU specify. If you search `n_clusters=[5, 8, 10, 15]`, you'll never discover that `n_clusters=37` might be optimal. You're limited by your own imagination.

### How RandomizedSearchCV works

Instead of explicit lists, you provide **distributions** — ranges to sample from randomly:

```python
from scipy.stats import randint

param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50),   # any int from 3–49
    'random_forest__max_features': randint(low=2, high=20),      # any int from 2–19
}
```

Then you set a **budget** — how many random combinations to try:

```
n_iter=10 → "draw 10 random (n_clusters, max_features) pairs"

Possible draws:
  Combo 1:  n_clusters=37,  max_features=12
  Combo 2:  n_clusters=8,   max_features=17
  Combo 3:  n_clusters=28,  max_features=9
  ...
  Combo 10: n_clusters=22,  max_features=7
```

Each combo is evaluated with k-fold CV, just like Grid Search.

### Grid Search vs Randomized Search

```
Grid Search:
  You say: "try [5, 8, 10, 15]"         → 4 specific values
  It tries: exactly those 4             → might miss the sweet spot between them

Randomized Search:
  You say: "try anything from 3 to 49"   → 47 possible values
  It tries: 10 random picks              → can land anywhere in the range
```

| Aspect | GridSearchCV | RandomizedSearchCV |
|--------|-------------|-------------------|
| You define | Exact values to try | Range to sample from |
| Search space | Narrow (your picks) | Wide (entire range) |
| Budget control | Determined by grid size | You set `n_iter` |
| Best for | Small, focused search | Wide exploration |
| Typical workflow | After RandomizedSearch (zoom in) | First pass (explore broadly) |

### The practical workflow

```
Step 1: RandomizedSearch with wide ranges
        → discover roughly where the good values are
        → e.g., "n_clusters around 30-45 seems good"

Step 2: GridSearch around the best values found
        → fine-tune within that neighborhood
        → e.g., try [30, 35, 37, 40, 45]
```

### scipy.stats.randint

`randint(low, high)` is a uniform distribution over integers in `[low, high)`. Every integer in the range is equally likely to be drawn. `random_state=42` makes the draws reproducible — same "random" combos every time you run it.

---

## Feature Importances

### What they measure

After training, a Random Forest can tell you how much each feature contributed to its predictions. The importance of a feature = the total reduction in prediction error from all splits using that feature, averaged across all trees.

```
How it's calculated (simplified):

Tree 1, Split 3: splits on "median_income" → error drops by 500
Tree 1, Split 7: splits on "median_income" → error drops by 300
Tree 2, Split 1: splits on "median_income" → error drops by 450
...across 100 trees × hundreds of splits...

Total error reduction from "median_income" = very high → importance ~0.19
Total error reduction from "ocean_ISLAND"  = very low  → importance ~0.00
```

All importances sum to 1.0.

### Why they matter

1. **Sanity check**: Does the model agree with domain knowledge? Income SHOULD matter most for housing prices. If "housing_median_age" ranked #1, something might be wrong.

2. **Feature selection**: Features with ~0 importance could be dropped to simplify the pipeline without losing accuracy. Simpler models are faster and easier to maintain.

3. **Debugging**: If a feature you expected to matter scores low, it might be:
   - Redundant with another feature (both capture the same information)
   - Poorly engineered (the transformation lost useful signal)
   - Genuinely not predictive (your intuition was wrong)

### How to access them

```python
final_model = rnd_search.best_estimator_

# Feature importances from the Random Forest
importances = final_model["random_forest"].feature_importances_

# Feature names from preprocessing
names = final_model["preprocessing"].get_feature_names_out()

# Sort together, highest first
sorted(zip(importances, names), reverse=True)
```

### Limitations

Feature importances can be **misleading** when features are correlated. If two features carry similar information (e.g., `total_rooms` and `total_bedrooms`), the model might split the importance between them — making both look less important than they really are. This doesn't mean either feature is unimportant; it means their contributions are shared.

---

## Test Set Evaluation

### Why the test set exists

Throughout the ML workflow, we used cross-validation to make decisions:
- Which model to use (Random Forest beat Linear Regression)
- Which hyperparameters to choose (n_clusters=15, max_features=6)

But we **chose based on CV scores** — so even those scores are slightly optimistic. The test set is the only truly unbiased estimate because it was never involved in any decision.

### The exam analogy

```
Cross-validation  = practice exams    → helpful for studying, but you adjust
                                        your strategy based on results
Test set          = the final exam    → taken once, no adjustments allowed
```

### The critical rule: evaluate ONCE

```
                    ┌─────────────────────────────┐
                    │          TEST SET            │
                    │                              │
                    │  ✓ Evaluate once             │
                    │  ✗ Do NOT tune based on it   │
                    │  ✗ Do NOT "try one more       │
                    │    thing" and re-evaluate     │
                    │                              │
                    │  If you keep testing and      │
                    │  adjusting, the test set      │
                    │  becomes another training     │
                    │  set — you lose your only     │
                    │  honest estimate.             │
                    └─────────────────────────────┘
```

### Confidence interval

A single RMSE number has uncertainty — a different random 20% split would give a slightly different result. The 95% confidence interval quantifies this:

```
Test RMSE: $41,500
95% CI: [$39,200, $43,700]

Meaning: we're 95% confident the true generalization error
is somewhere between $39,200 and $43,700.
```

The interval width depends on test set size — more test data → narrower interval → more certainty.

### Interpreting the result

```
                      CV RMSE        Test RMSE      What it means
                      ──────         ─────────      ─────────────
Close match:          $43,590        $41,500        CV was honest. Model generalizes
                                                    well. Ship it.

Test much worse:      $43,590        $55,000        Overfit during tuning. Tried too
                                                    many combinations and got lucky
                                                    on CV. Need more data or simpler
                                                    model.

Test much better:     $43,590        $38,000        Got lucky with the test split.
                                                    Don't celebrate too hard — the
                                                    CV estimate is more reliable.
```