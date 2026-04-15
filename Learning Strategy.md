# Learning Strategy

Personal notes on how to work through this book, distilled from early-chapter reflections.

## Guiding principle

**Don't pause the book to master prerequisites.** Interleave instead. Math and stats land much better when anchored to a practical need in the chapter you're working on. Blocking on "finish linear algebra first" kills momentum and retention.

- **Rusty is not missing.** A software engineering background with prior exposure to calculus/stats is plenty to start. What feels like "gaps" is usually reactivation, not cold-start learning.
- **JIT, not upfront.** Learn the specific concept when you hit it, not in anticipation.
- **Skim first, deepen on demand.** Build a mental index of *where* things are, then return to sections when they become relevant.

## Math

### Priority ordering

1. **Linear algebra** — gives you the shapes and notation (vectors, matrices, dot products). Needed to even parse ML formulas.
2. **Calculus** — narrower. Mostly "gradient = direction of steepest increase" and chain-rule intuition.

### Primary reference — Géron's companion notebooks

Tuned to the exact math this book uses, notation matches, interactive.

- [math_linear_algebra.ipynb](https://github.com/ageron/handson-ml3/blob/main/math_linear_algebra.ipynb)
- [math_differential_calculus.ipynb](https://github.com/ageron/handson-ml3/blob/main/math_differential_calculus.ipynb)

**Approach:** Skim each notebook once (~45 min). Don't try to master on first pass. Return to specific sections when a concept blocks you.

### Minimum viable topics

**Linear algebra:**
- Vectors as points in space
- Dot product as "how aligned are two vectors"
- Matrix × vector as a transformation
- Shapes and broadcasting (`(m,n) × (n,) → (m,)`)
- Norms (vector "size")

**Calculus:**
- Derivative as slope / rate of change
- Gradient as a vector pointing uphill in multi-dim space
- Partial derivatives — recognize the notation
- Chain rule — intuition only

**Safe to defer:**
- Eigenvalues/eigenvectors (comes back for PCA in Ch. 8 — learn then)
- Hessians, Jacobians, Taylor series
- Proof-heavy material

## Statistics

More important than math for ML judgment — most ML mistakes are statistical, not mathematical. Géron doesn't ship a dedicated stats notebook; concepts are introduced inline across chapters (Ch. 2 histograms, Ch. 3 precision/recall, Ch. 4 bias-variance).

### Primary reference

**The StatQuest Illustrated Guide to Statistics: With hands-on examples in Python and R** — Josh Starmer.

Fills the gap left by the absence of a stats companion notebook. Illustrated, intuition-first, matches the StatQuest YouTube style. Python/R examples for hands-on work. Good as both read-through and reference.

### Minimum viable topics

1. **Descriptive stats** — mean, median, std, variance, quartiles, distributions
2. **Probability basics** — random variables, expectation, conditional probability, Bayes' rule
3. **Sampling & estimation** — sample vs population, sampling bias, standard error, confidence intervals (intuition)
4. **Hypothesis testing** — intuition for p-values, Type I/II errors (these map directly to false positive/negative)
5. **The statistical view of ML** — bias-variance tradeoff, overfitting, cross-validation, regularization as a prior

### Supplementary — video form

**StatQuest (YouTube, Josh Starmer)** — short, targeted, intuitive. Watch videos as they map to chapter topics:
- "Confusion Matrix"
- "Sensitivity and Specificity"
- "ROC and AUC"
- "Bias and Variance"

### Stats concepts hidden in ML terminology

A lot of statistics shows up relabeled:

| Stats concept | ML equivalent |
|---------------|----------------|
| Sample vs population | train set vs "true" data distribution |
| Standard error | variance across CV folds |
| Type I / Type II error | false positive / false negative |
| Bias | model systematically wrong |
| Variance | model sensitive to training data |
| Law of large numbers | why more training data helps |
| Central limit theorem | why ensembles help |

## Optional ML companion

**The StatQuest Illustrated Guide to Machine Learning** — Josh Starmer.

Visual, intuition-first companion to HOML. Useful for alternate explanations when Géron's version doesn't click. Not essential — HOML is self-contained — but good to have on hand.

## When "overwhelmed" hits

Normal, especially early. You're stacking math + stats + library conventions + problem framing + engineering patterns simultaneously. Tactics:

- **Lower the resolution.** Rough mental model beats exact derivation.
- **Finish the chapter end-to-end before going deep.** Pipeline in hand > one concept mastered in isolation.
- **Second pass effect.** Concepts come back in later chapters (SGD → Ch. 4 → Ch. 10). Each pass deepens the groove.
- **Translate to SWE terms.** `fit()` = populate state from data. Weights = learned config. Training = search over parameter space.

## Concrete next-session plan (for reference)

1. Skim linear algebra notebook (~45 min) — run cells, no notes.
2. Skim calculus notebook (~45 min) — same approach.
3. Return to Ch. 3 performance measures.
4. Watch StatQuest videos as specific topics come up (precision/recall, ROC, bias-variance).
5. Pick up the StatQuest stats book as a parallel track — read alongside the chapters, not before.
