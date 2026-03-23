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