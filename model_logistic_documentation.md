# Bug Report Severity Classification - Logistic Regression Model
## Complete Documentation (model_logistic.py)

---

## Table of Contents
1. [What This Program Does](#what-this-program-does)
2. [Why Logistic Regression?](#why-logistic-regression)
3. [Key Concepts Explained](#key-concepts-explained)
4. [Comparison: All Three Models](#comparison-all-three-models)
5. [Line-by-Line Code Explanation](#line-by-line-code-explanation)
6. [How Logistic Regression Works](#how-logistic-regression-works)
7. [Input/Output Examples](#inputoutput-examples)

---

## What This Program Does

Same goal as the other models - classifies bug reports into severity levels:
- **blocker**, **critical**, **major**, **normal**, **minor**, **trivial**

**Uses:** Logistic Regression - one of the most popular and effective algorithms for text classification.

---

## Why Logistic Regression?

Logistic Regression is often the **first choice** for text classification because:

| Advantage | Description |
|-----------|-------------|
| **Fast** | Trains quickly even on large datasets |
| **Interpretable** | You can see which words influence predictions |
| **Effective** | Often performs as well as complex models |
| **Probabilistic** | Naturally outputs probabilities |
| **Widely Used** | Industry standard, well-documented |
| **No Tuning Needed** | Works well with default settings |

**Fun Fact:** Despite having "Regression" in its name, it's used for **classification**, not regression!

---

## Key Concepts Explained

### What is Logistic Regression?

**Linear Regression** predicts a number (e.g., house price = $250,000)

**Logistic Regression** predicts a probability (e.g., probability of spam = 87%)

### The Sigmoid Function

Logistic Regression uses the **sigmoid function** to convert any number to a probability (0 to 1):

```
                    1
    σ(x) = ─────────────────
            1 + e^(-x)
```

**Visualized:**
```
Probability
    1.0 |                  ___________
        |                /
    0.5 |              /
        |            /
    0.0 |___________/
        -∞        0        +∞
                Score
```

- Large positive score → probability close to 1 (likely this class)
- Large negative score → probability close to 0 (unlikely this class)
- Score of 0 → probability of 0.5 (uncertain)

### How It Makes Predictions

For text classification:

```
Step 1: Calculate weighted sum of word features
        Score = w1×(word1) + w2×(word2) + w3×(word3) + ... + bias

Step 2: Apply sigmoid function
        Probability = sigmoid(Score)

Step 3: If probability > 0.5, predict that class
```

**Example:**
```
Bug report: "application crashes on startup"

Weights learned during training:
- "crashes" → +2.5 (indicates critical)
- "startup" → +0.8 (somewhat critical)
- "application" → +0.1 (neutral)

Score = 2.5 + 0.8 + 0.1 = 3.4
Probability = sigmoid(3.4) = 0.97

Prediction: critical (97% confident)
```

### Multi-class Classification

Logistic Regression is naturally binary (two classes). For multiple classes, it uses **One-vs-Rest (OvR)**:

```
Train 6 separate classifiers:
1. blocker vs (not blocker)
2. critical vs (not critical)
3. major vs (not major)
4. normal vs (not normal)
5. minor vs (not minor)
6. trivial vs (not trivial)

Final prediction = class with highest probability
```

### The C Parameter

**C = Inverse of regularization strength**

```
High C (C=100):                   Low C (C=0.01):
- Less regularization            - More regularization
- Fits training data closely     - Simpler model
- Risk of overfitting            - Better generalization
- Larger weights allowed         - Weights kept small
```

**Default C=1.0** is usually a good balance.

**What is Regularization?**
Prevents the model from becoming too complex and memorizing training data instead of learning patterns.

### The Solver

**Solver** = The algorithm used to find optimal weights

| Solver | Best For | Description |
|--------|----------|-------------|
| `lbfgs` | Small/medium datasets | Limited-memory BFGS, default choice |
| `liblinear` | Small datasets, L1 penalty | Fast for small data |
| `sag` | Large datasets | Stochastic Average Gradient |
| `saga` | Large datasets, L1 penalty | Improved SAG |
| `newton-cg` | Medium datasets | Newton conjugate gradient |

**We use `lbfgs`** - works well for most cases.

### max_iter Parameter

**max_iter** = Maximum number of iterations for the solver to converge

- Default is 100, but text data often needs more
- We use 1000 to ensure convergence
- If you see "convergence warning", increase this

### class_weight='balanced'

Same as in SVM - automatically adjusts for imbalanced classes.

```
If dataset has:
- 700 "normal" bugs
- 50 "critical" bugs

Without balancing: Model ignores "critical"
With balancing: Model pays 14x more attention to "critical"
```

---

## Comparison: All Three Models

| Aspect | Naive Bayes | SVM | Logistic Regression |
|--------|-------------|-----|---------------------|
| **Algorithm** | Probability | Geometry | Probability |
| **Speed** | Fastest | Slowest | Fast |
| **Accuracy** | Good | Best | Very Good |
| **Interpretability** | Medium | Low | High |
| **Memory** | Low | High | Low |
| **Handles Imbalance** | ComplementNB | class_weight | class_weight |
| **Best For** | Quick baseline | Complex patterns | Production systems |

### When to Use Each:

```
Start here: Naive Bayes (fastest, good baseline)
     ↓
Need better accuracy? Try Logistic Regression
     ↓
Still not good enough? Try SVM
     ↓
Compare all three, pick best for your data
```

---

## Line-by-Line Code Explanation

### Lines 1-11: Importing Libraries

```python
import pandas as pd
```
**pandas** - Data manipulation for tables and confusion matrix display.

```python
import numpy as np
```
**numpy** - Numerical operations for finding max probability.

```python
import re
```
**re** - Regular expressions for text cleaning.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
**TfidfVectorizer** - Converts text to TF-IDF numerical features.

```python
from sklearn.model_selection import train_test_split
```
**train_test_split** - Splits data into training (80%) and testing (20%) sets.

```python
from sklearn.linear_model import LogisticRegression
```
**LogisticRegression** - The main classifier we're using.
- Located in `linear_model` module because it's based on linear combination of features
- Despite the name, used for classification, not regression

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
**Metrics** - Evaluate model performance.

```python
from sklearn.preprocessing import LabelEncoder
```
**LabelEncoder** - Convert text labels to numbers.

```python
from datasets import load_dataset
```
**load_dataset** - Download bug report dataset from HuggingFace.

```python
import warnings
warnings.filterwarnings('ignore')
```
**Suppress warnings** for cleaner output.

---

### Lines 13-77: load_and_preprocess_data() Function

**Identical to other models** - loads Eclipse bug reports dataset.

Key steps:
1. Try multiple dataset names
2. Extract training split
3. Find text and label columns
4. Clean invalid rows
5. Limit to 1000 samples

---

### Lines 79-87: preprocess_text() Function

**Identical to other models** - cleans text:
- Lowercase
- Remove extra whitespace
- Remove special characters
- Strip edges

---

### Lines 89-111: prepare_data() Function

**Identical to Naive Bayes model:**

```python
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
```

Uses same TF-IDF settings as Naive Bayes (5000 features).

---

### Lines 113-123: train_model() Function - THE KEY DIFFERENCE

```python
def train_model(X_train, y_train):
    model = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
```

Let's break down each parameter:

#### `C=1.0`
**Inverse regularization strength**
- C=1.0 is the default, balanced choice
- Higher C = less regularization = more complex model
- Lower C = more regularization = simpler model

```
C=0.01: Very simple model (high regularization)
        Weights: crash=0.5, error=0.3, bug=0.2

C=100:  Complex model (low regularization)
        Weights: crash=15.3, error=8.7, bug=4.2
```

#### `solver='lbfgs'`
**Optimization algorithm**
- LBFGS = Limited-memory Broyden-Fletcher-Goldfarb-Shanno
- Good default for most problems
- Works well with L2 regularization (default)
- Efficient for medium-sized datasets

#### `max_iter=1000`
**Maximum iterations**
- How many times the algorithm adjusts weights
- Default 100 often isn't enough for text data
- 1000 ensures the model converges (finds optimal weights)

#### `class_weight='balanced'`
**Handle imbalanced data**
- Automatically calculates: n_samples / (n_classes * class_count)
- Rare classes get higher weight
- Common classes get lower weight

```
Example calculation:
Total samples: 1000
Classes: 6

"normal" (600 samples): weight = 1000 / (6 × 600) = 0.28
"critical" (50 samples): weight = 1000 / (6 × 50) = 3.33

"critical" gets 12x more weight than "normal"
```

#### `random_state=42`
**Reproducibility**
- Same results every run
- Important for comparing models

```python
    model.fit(X_train, y_train)
```
**Training process:**
1. Initialize weights randomly
2. For each iteration:
   - Calculate predictions
   - Measure error (log loss)
   - Adjust weights to reduce error
3. Stop when error stops decreasing (or max_iter reached)

```python
    return model
```
Return trained model containing:
- Learned weights for each word (model.coef_)
- Bias term (model.intercept_)

---

### Lines 125-139: evaluate_model() Function

**Identical to other models:**

```python
    y_pred = model.predict(X_test)
```
Prediction process:
1. Calculate score for each class: Score = weights · features + bias
2. Apply softmax (multi-class sigmoid) to get probabilities
3. Return class with highest probability

```python
    accuracy = accuracy_score(y_test, y_pred)
```
Percentage of correct predictions.

```python
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```
Detailed per-class metrics.

```python
    cm = confusion_matrix(y_test, y_pred)
```
Prediction patterns visualization.

---

### Lines 141-159: predict_custom_examples() Function

**Identical to other models** - tests on sample bug descriptions.

---

### Lines 161-190: interactive_prediction_loop() Function

**Identical to other models** - interactive user input mode.

```python
    prediction_probs = model.predict_proba(input_tfidf)[0]
```
**Logistic Regression naturally outputs probabilities** - no extra computation needed (unlike SVM).

---

### Lines 192-206: main() Function

**Identical to other models:**
1. Prepare data
2. Train model
3. Evaluate
4. Test examples
5. Interactive mode

---

## How Logistic Regression Works

### Training Phase

```
┌──────────────────────────────────────────────────────────────┐
│  TRAINING: Learning word weights                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Training data:                                               │
│  "app crashes" → critical                                     │
│  "minor typo" → trivial                                       │
│  "system fails" → critical                                    │
│                                                               │
│  Initialize weights randomly:                                 │
│  crash=0.1, typo=0.2, fails=0.1, ...                         │
│                                                               │
│  Iteration 1:                                                 │
│  - Predict "app crashes" → normal (wrong!)                    │
│  - Increase weight for "crashes"                              │
│                                                               │
│  Iteration 2:                                                 │
│  - Predict "app crashes" → major (closer!)                    │
│  - Increase weight for "crashes" more                         │
│                                                               │
│  ... after 1000 iterations ...                                │
│                                                               │
│  Final weights:                                               │
│  crash=2.5, typo=-1.8, fails=2.2, error=1.9, ...             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Prediction Phase

```
┌──────────────────────────────────────────────────────────────┐
│  PREDICTION: Using learned weights                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  New bug report: "database crashes causing errors"            │
│                                                               │
│  Step 1: TF-IDF features                                      │
│  database=0.4, crashes=0.8, causing=0.2, errors=0.6          │
│                                                               │
│  Step 2: Calculate scores for each class                      │
│                                                               │
│  Score(critical) = 2.5×0.8 + 1.9×0.6 + ... = 4.2             │
│  Score(normal)   = 0.3×0.8 + 0.5×0.6 + ... = 1.1             │
│  Score(trivial)  = -0.5×0.8 + -0.2×0.6 + ... = -0.8          │
│                                                               │
│  Step 3: Convert to probabilities (softmax)                   │
│                                                               │
│  P(critical) = e^4.2 / (e^4.2 + e^1.1 + e^-0.8 + ...) = 0.82 │
│  P(normal)   = e^1.1 / (e^4.2 + e^1.1 + e^-0.8 + ...) = 0.11 │
│  P(trivial)  = e^-0.8 / (e^4.2 + e^1.1 + e^-0.8 + ...) = 0.02│
│                                                               │
│  Step 4: Predict highest probability                          │
│                                                               │
│  Output: critical (82% confidence)                            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Visualizing the Decision Boundary

In 2D (simplified):

```
        word2 (crash)
           ^
           |    critical    /
           |      x x      /
           |    x   x     /    <-- Decision boundary
           |             /         (linear for Logistic Regression)
           |   normal   /
           |    o o    /
           |   o o o  /
           +-----------+---> word1 (minor)
                      /
                trivial
```

Unlike SVM, Logistic Regression always creates **linear** (straight) boundaries.

---

## Interpreting the Model

### Viewing Feature Importance

One advantage of Logistic Regression - you can see which words matter:

```python
# After training, you can check:
feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]  # Weights for first class

# Top words indicating "critical":
top_critical = sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True)[:10]

# Might show:
# crash: +3.2
# error: +2.8
# fail: +2.5
# unresponsive: +2.3
# ...
```

This interpretability is valuable for understanding WHY the model makes decisions.

---

## Input/Output Examples

### Input
```
Enter text: The system crashes when uploading large files
```

### Output
```
Predicted: critical (confidence: 0.784)
   critical    : ███████████████░░░░░ 0.784
   blocker     : ██░░░░░░░░░░░░░░░░░░ 0.098
   major       : █░░░░░░░░░░░░░░░░░░░ 0.067
   normal      : ░░░░░░░░░░░░░░░░░░░░ 0.031
   minor       : ░░░░░░░░░░░░░░░░░░░░ 0.015
   trivial     : ░░░░░░░░░░░░░░░░░░░░ 0.005
```

### Another Example

### Input
```
Enter text: Small spelling mistake in settings menu
```

### Output
```
Predicted: trivial (confidence: 0.687)
   trivial     : █████████████░░░░░░░ 0.687
   minor       : ████░░░░░░░░░░░░░░░░ 0.198
   normal      : ██░░░░░░░░░░░░░░░░░░ 0.076
   major       : ░░░░░░░░░░░░░░░░░░░░ 0.023
   critical    : ░░░░░░░░░░░░░░░░░░░░ 0.011
   blocker     : ░░░░░░░░░░░░░░░░░░░░ 0.005
```

---

## Summary

### Key Differences from Other Models

| Aspect | Naive Bayes | SVM | Logistic Regression |
|--------|-------------|-----|---------------------|
| **Import** | `ComplementNB` | `SVC` | `LogisticRegression` |
| **Key Param** | `alpha` | `kernel, C, gamma` | `C, solver, max_iter` |
| **Boundary** | Probabilistic | Can be curved | Always linear |
| **Probabilities** | Native | Needs `probability=True` | Native |
| **Interpretable** | Medium | Hard | Easy (check weights) |

### Logistic Regression Parameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `C` | 1.0 | Regularization strength |
| `solver` | 'lbfgs' | Optimization algorithm |
| `max_iter` | 1000 | Ensure convergence |
| `class_weight` | 'balanced' | Handle imbalanced data |
| `random_state` | 42 | Reproducibility |

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         PIPELINE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load Data        : HuggingFace Eclipse Bug Reports          │
│         ↓                                                        │
│  2. Preprocess       : lowercase, remove special chars          │
│         ↓                                                        │
│  3. Vectorize        : TF-IDF (5000 features)                   │
│         ↓                                                        │
│  4. Split            : 80% train, 20% test                      │
│         ↓                                                        │
│  5. Train            : Logistic Regression learns weights       │
│         ↓                                                        │
│  6. Evaluate         : Accuracy, precision, recall, F1          │
│         ↓                                                        │
│  7. Predict          : Interactive mode for new bugs            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommendation

For bug severity classification:

1. **Start with Logistic Regression** - fast, effective, interpretable
2. **Compare with Naive Bayes** - even faster baseline
3. **Try SVM if needed** - potentially higher accuracy but slower

Most real-world text classification systems use Logistic Regression as their primary model because of its excellent balance of speed, accuracy, and interpretability.
