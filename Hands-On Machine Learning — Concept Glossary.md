> A living document. Add new terms as you go through each chapter. Revisit and refine definitions as your understanding deepens.

---

## Core ML Terminology

|Term|Definition|Example|
|---|---|---|
|**Model Parameter**|A value the model **learns from data** during training|Weights in a neural network, coefficients in linear regression|
|**Hyperparameter**|A value **you set before training**; not learned from data|`feature_range=(-1, 1)` in MinMaxScaler, `n_estimators` in RandomForest|
|**Training Set**|The subset of data used to train the model|80% of your housing dataset|
|**Test Set**|The subset held back to evaluate final model performance|20% of your housing dataset|
|**Validation Set**|A subset used to tune hyperparameters and compare models|Created via cross-validation|
|**Cross-Validation**|Splitting training data into folds, training on some and validating on others|10-fold CV: train on 9 folds, validate on 1, repeat 10 times|
|**Overfitting**|Model learns noise in training data, performs poorly on new data|Memorizing training examples instead of learning patterns|
|**Underfitting**|Model is too simple to capture the underlying pattern|Using a straight line for data that curves|
|**Pipeline**|A sequence of data processing steps chained together|`Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])`|

---

## Clean the Data

|Term|Definition|Example|
|---|---|---|
|**Missing Values**|Data points where a feature has no value|A row where `total_bedrooms` is NaN|
|**SimpleImputer**|Fills missing values with a strategy (mean, median, mode)|`SimpleImputer(strategy="median")` fills blanks with the median|
|**Imputation Strategy**|The method chosen to fill missing values|`"median"`, `"mean"`, `"most_frequent"`, `"constant"`|
|**fit()**|Learns/computes what it needs from training data|Imputer calculates the median of each column|
|**transform()**|Applies what was learned to data|Imputer fills missing values using the median it learned|
|**fit_transform()**|Calls fit() then transform() in one step|`imputer.fit_transform(housing_num)`|
|**DataFrame**|A 2D labeled data structure in pandas (rows and columns)|`housing_num = housing.select_dtypes(include=[np.number])`|

---

## Handling Text and Categorical Attributes

|Term|Definition|Example|
|---|---|---|
|**Categorical Feature**|A feature with a limited set of possible text/label values|`ocean_proximity`: "NEAR BAY", "INLAND", "NEAR OCEAN"|
|**OrdinalEncoder**|Converts categories to integers, implies an order|"bad"→0, "average"→1, "good"→2|
|**OneHotEncoder**|Creates a binary column for each category (no false ordering)|"INLAND"→[1,0,0], "NEAR BAY"→[0,1,0], "NEAR OCEAN"→[0,0,1]|
|**Sparse Matrix**|Stores only non-zero values and their positions; memory efficient|OneHotEncoder output — mostly zeros, only one 1 per row|
|**Dense Matrix**|A regular matrix where every value (including zeros) is stored|A standard NumPy array|

---

## Feature Scaling and Transformation

|Term|Definition|Example|
|---|---|---|
|**Feature Scaling**|Adjusting features to similar ranges so no single feature dominates|Scaling income (0–100k) and age (0–100) to the same range|
|**Min-Max Scaling (Normalization)**|Rescales values to a fixed range, default [0, 1]|`MinMaxScaler(feature_range=(-1, 1))`|
|**Standardization**|Centers to mean=0 and scales to std=1|`StandardScaler()` subtracts mean, divides by std deviation|
|**Mean**|Average of all values|[0, 0, 5, 10] → mean = 15/4 = 3.75|
|**Variance**|Average of squared differences from the mean|Measures how spread out values are|
|**Standard Deviation**|Square root of variance; same unit as original data|std = √variance|
|**with_mean**|Hyperparameter in StandardScaler; set to False for sparse data|`StandardScaler(with_mean=False)` skips centering, preserves sparsity|
|**Breaking Sparsity**|Subtracting mean turns zeros into non-zeros, destroying sparse structure|[0,0,5] with mean=1.67 → [-1.67, -1.67, 3.33] — no more zeros|
|**Heavy-Tailed Distribution**|Most values clustered on one side with extreme outliers stretching out|Income: most people earn 30k–80k, a few earn millions|
|**Log Transform**|Compresses large values, useful for heavy-tailed features|`np.log1p(income)` makes the distribution more bell-shaped|
|**Bucketizing**|Splitting a continuous feature into discrete bins/ranges|Age: 0–18→"child", 19–65→"adult", 65+→"senior"|
|**RBF Kernel (Radial Basis Function)**|Measures similarity to a fixed point; decays with distance|High value when close to a landmark, low when far|


---

## Scikit-Learn API Patterns

|Term|Definition|Example|
|---|---|---|
|**Estimator**|Any object that learns from data (has a `fit()` method)|`SimpleImputer`, `RandomForestRegressor`|
|**Transformer**|An estimator that can also transform data (has `transform()`)|`StandardScaler`, `OneHotEncoder`|
|**Predictor**|An estimator that can make predictions (has `predict()`)|`LinearRegression`, `DecisionTreeClassifier`|

---

## Template for New Entries

```
| **Term** | Definition here | `code_example` or short illustration |
```