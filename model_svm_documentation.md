# Bug Report Severity Classification - SVM Model
## Complete Documentation (model_svm.py)

---

## Table of Contents
1. [What This Program Does](#what-this-program-does)
2. [SVM vs Naive Bayes - Key Differences](#svm-vs-naive-bayes---key-differences)
3. [Key Concepts Explained](#key-concepts-explained)
4. [Line-by-Line Code Explanation](#line-by-line-code-explanation)
5. [How SVM Works Visually](#how-svm-works-visually)
6. [Input/Output Examples](#inputoutput-examples)

---

## What This Program Does

Same goal as model.py - classifies bug reports into severity levels:
- **blocker**, **critical**, **major**, **normal**, **minor**, **trivial**

**But uses a different algorithm:** Support Vector Machine (SVM) instead of Naive Bayes.

---

## SVM vs Naive Bayes - Key Differences

| Aspect | Naive Bayes (model.py) | SVM (model_svm.py) |
|--------|------------------------|---------------------|
| **Speed** | Very fast | Slower to train |
| **Approach** | Probability-based | Geometry-based (finds boundaries) |
| **Complexity** | Simple, linear | Can handle complex, non-linear patterns |
| **Memory** | Low | Higher (stores support vectors) |
| **Best for** | Quick baseline, simple text | Complex patterns, higher accuracy |
| **Accuracy** | Good | Often better |

**When to use which?**
- Start with Naive Bayes (fast, simple)
- Try SVM if you need better accuracy
- SVM is slower but often more accurate

---

## Key Concepts Explained

### What is Support Vector Machine (SVM)?

Imagine you have red and blue balls on a table. You want to draw a line separating them.

**SVM finds the BEST line** - the one with maximum distance from both groups.

```
        Red balls                    Blue balls
           o                              x
          o o          |                 x x
         o o o    <----|---->           x x x
          o o          |                 x x
           o           |                  x

                   Best line
              (maximum margin)
```

The balls closest to the line are called **Support Vectors** - they "support" the decision boundary.

### What is a Kernel?

What if data can't be separated by a straight line?

```
Before (2D - not separable):        After (3D - separable):
                                            x x x
    x x x x                                  ___
   x o o o x                               /   \
    x x x x                           o o o     (now we can draw a plane!)

```

**Kernels** transform data to higher dimensions where it becomes separable.

### Types of Kernels

| Kernel | Shape | When to Use |
|--------|-------|-------------|
| `linear` | Straight line | Simple, separable data |
| `rbf` (Radial Basis Function) | Curved, circular | Complex patterns (DEFAULT) |
| `poly` | Polynomial curves | Specific curve patterns |

**RBF is most popular** - it can handle most non-linear patterns.

### What is the C Parameter?

**C = Regularization parameter** (controls trade-off between accuracy and generalization)

```
Low C (C=0.1):                    High C (C=100):
- Smoother boundary              - Tighter fit to training data
- More misclassifications OK     - Tries to classify everything right
- Better generalization          - May overfit (memorize training data)
- Less sensitive to noise        - More sensitive to noise
```

**Default C=1.0** is usually a good balance.

### What is Gamma?

**Gamma = How much influence a single training example has**

```
Low gamma:                        High gamma:
- Far reach (smooth boundary)    - Close reach (wiggly boundary)
- Considers distant points       - Only considers nearby points
- Underfitting risk              - Overfitting risk
```

**`gamma='scale'`** automatically calculates a good value based on your data.

### What is class_weight='balanced'?

When you have imbalanced data (90% "normal", 10% "critical"), the model may ignore minority classes.

**`class_weight='balanced'`** automatically gives more importance to rare classes.

```
Before balancing:                 After balancing:
- "normal": weight 1.0           - "normal": weight 0.5
- "critical": weight 1.0         - "critical": weight 5.0
(Model ignores critical)         (Model pays attention to critical)
```

---

## Line-by-Line Code Explanation

### Lines 1-11: Importing Libraries

```python
import pandas as pd
```
**pandas** - Data manipulation library for handling tabular data.

```python
import numpy as np
```
**numpy** - Numerical operations, used for `np.max()` to find highest probability.

```python
import re
```
**re** - Regular expressions for text cleaning.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
**TfidfVectorizer** - Converts text to numerical features using TF-IDF.
- Same as in model.py
- SVM also needs numerical input

```python
from sklearn.model_selection import train_test_split
```
**train_test_split** - Splits data into training (80%) and testing (20%) sets.

```python
from sklearn.svm import SVC
```
**SVC** - Support Vector Classifier (SVM for classification).
- This is the main difference from model.py
- `SVC` = Support Vector Classification
- `SVR` = Support Vector Regression (for predicting numbers, not used here)

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
**Metrics** - Same evaluation tools as model.py.

```python
from sklearn.preprocessing import LabelEncoder
```
**LabelEncoder** - Converts text labels to numbers.

```python
from datasets import load_dataset
```
**load_dataset** - Downloads bug report dataset from HuggingFace.

```python
import warnings
warnings.filterwarnings('ignore')
```
**Suppress warnings** for cleaner output.

---

### Lines 13-77: load_and_preprocess_data() Function

**Identical to model.py** - loads and cleans the dataset.

Key steps:
1. Try multiple dataset names until one works
2. Extract training split
3. Find text and label columns
4. Filter out empty rows
5. Limit to 1000 samples

```python
    texts = texts[:1000]
    labels = labels[:1000]
```
Using 1000 samples for reasonable training time.
- SVM is slower than Naive Bayes
- More samples = longer training

---

### Lines 79-87: preprocess_text() Function

**Identical to model.py** - cleans text:
1. Lowercase
2. Remove extra whitespace
3. Remove special characters
4. Strip edges

---

### Lines 89-111: prepare_data() Function

**Mostly identical to model.py**, but with one difference:

```python
    vectorizer = TfidfVectorizer(
        max_features=10000,      # <-- DIFFERENT: 10000 vs 5000
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
```

**Why 10000 features for SVM?**
- SVM can handle more features better than Naive Bayes
- More features = more information for finding boundaries
- Naive Bayes with too many features can get confused

---

### Lines 113-117: train_model() Function - THE KEY DIFFERENCE

```python
def train_model(X_train, y_train):
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model
```

Let's break down each parameter:

#### `kernel='rbf'`
**Radial Basis Function kernel** - creates curved decision boundaries.
- Best for text classification
- Can capture complex patterns
- Alternative: `kernel='linear'` for simpler, faster model

#### `C=1.0`
**Regularization strength**
- C=1.0 is the default, balanced choice
- Lower C (0.1): More generalization, some errors OK
- Higher C (10): Try harder to classify everything correctly

#### `gamma='scale'`
**Kernel coefficient** - how far influence reaches
- `'scale'` = automatically calculated as 1 / (n_features * X.var())
- Good default that adapts to your data
- Alternative: `'auto'` = 1 / n_features

#### `probability=True`
**Enable probability estimates**
- Without this: Only get predicted class (0, 1, 2...)
- With this: Get probability for each class (0.7, 0.2, 0.1...)
- Needed for confidence scores
- **Trade-off**: Slightly slower training

#### `random_state=42`
**Reproducibility seed**
- Makes results the same every run
- 42 is just a convention (from Hitchhiker's Guide!)
- Any number works

#### `class_weight='balanced'`
**Handle imbalanced classes**
- Automatically adjusts weights inversely proportional to class frequencies
- Formula: n_samples / (n_classes * class_count)
- Gives more importance to rare classes like "critical"

```python
    model.fit(X_train, y_train)
```
**Training the SVM**:
1. Finds support vectors (data points near the boundary)
2. Calculates optimal hyperplane (decision boundary)
3. Stores support vectors in memory

```python
    return model
```
Return trained model. The model now contains:
- Support vectors (subset of training data)
- Weights for the decision boundary
- Kernel parameters

---

### Lines 119-131: evaluate_model() Function

**Identical to model.py** - evaluates model performance:

```python
    y_pred = model.predict(X_test)
```
SVM prediction process:
1. For new data point, calculate distance to decision boundary
2. Assign class based on which side of boundary

```python
    accuracy = accuracy_score(y_test, y_pred)
```
Calculate percentage of correct predictions.

```python
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```
Detailed metrics per class.

```python
    cm = confusion_matrix(y_test, y_pred)
```
Show prediction patterns (correct vs incorrect).

---

### Lines 133-151: predict_custom_examples() Function

**Identical to model.py** - tests on sample bug descriptions.

The prediction process for SVM:
```
New text → Clean → TF-IDF → Calculate kernel with support vectors → Predict class
```

---

### Lines 153-182: interactive_prediction_loop() Function

**Identical to model.py** - interactive mode for user input.

```python
    prediction_probs = model.predict_proba(input_tfidf)[0]
```
**How SVM calculates probabilities:**
- SVM doesn't naturally output probabilities
- Uses Platt Scaling (fits a logistic regression on SVM scores)
- That's why `probability=True` is needed and makes training slower

---

### Lines 184-198: main() Function

**Identical to model.py** - runs the complete pipeline:
1. Prepare data
2. Train SVM model
3. Evaluate
4. Test examples
5. Interactive mode

---

## How SVM Works Visually

### Step 1: Data in Feature Space

After TF-IDF, each bug report becomes a point in 10000-dimensional space.
(We'll visualize in 2D for simplicity)

```
           word2 (crash)
              ^
              |    x x         x = critical
              |  x   x         o = normal
              | o o
              |o o o o
              +-----------> word1 (minor)
```

### Step 2: SVM Finds Best Boundary

```
           word2 (crash)
              ^
              |    x x
              |  x   x
              |    \            <-- Decision boundary
              | o o  \
              |o o o o\
              +-----------> word1 (minor)
```

### Step 3: Support Vectors Identified

```
           word2 (crash)
              ^
              |    x x
              |  X   x         X = Support vectors
              |    \           O = Support vectors
              | O o  \
              |o o o o\
              +-----------> word1 (minor)
```

The boundary is defined ONLY by support vectors - other points don't matter!

### Step 4: Prediction

New bug report "system crash" maps to a point:

```
           word2 (crash)
              ^
              |    x ? <-- New point
              |  x   x
              |    \
              | o o  \
              |o o o o\
              +-----------> word1 (minor)

Point is on "critical" side → Predict "critical"
```

---

## Input/Output Examples

### Input
```
Enter text: Database connection fails causing data loss
```

### Output
```
Predicted: critical (confidence: 0.756)
   critical    : ███████████████░░░░░ 0.756
   blocker     : ███░░░░░░░░░░░░░░░░░ 0.142
   major       : █░░░░░░░░░░░░░░░░░░░ 0.058
   normal      : ░░░░░░░░░░░░░░░░░░░░ 0.028
   minor       : ░░░░░░░░░░░░░░░░░░░░ 0.011
   trivial     : ░░░░░░░░░░░░░░░░░░░░ 0.005
```

### Another Example

### Input
```
Enter text: Button color looks slightly off
```

### Output
```
Predicted: trivial (confidence: 0.612)
   trivial     : ████████████░░░░░░░░ 0.612
   minor       : █████░░░░░░░░░░░░░░░ 0.243
   normal      : ██░░░░░░░░░░░░░░░░░░ 0.089
   major       : ░░░░░░░░░░░░░░░░░░░░ 0.032
   critical    : ░░░░░░░░░░░░░░░░░░░░ 0.015
   blocker     : ░░░░░░░░░░░░░░░░░░░░ 0.009
```

---

## Comparison: Same Bug, Different Models

### Bug: "Application crashes on startup"

**Naive Bayes (model.py):**
```
Predicted: critical (confidence: 0.723)
```

**SVM (model_svm.py):**
```
Predicted: critical (confidence: 0.812)
```

SVM often has higher confidence because it finds clearer boundaries.

---

## Summary Table

| Component | Purpose | SVM Specific |
|-----------|---------|--------------|
| TfidfVectorizer | Text → Numbers | 10000 features (more than NB) |
| SVC | The SVM classifier | kernel='rbf' for non-linear |
| class_weight | Handle imbalanced data | 'balanced' auto-adjusts |
| probability | Get confidence scores | Enables predict_proba() |
| Support Vectors | Define boundary | Stored in model after training |

---

## When to Choose SVM over Naive Bayes

| Scenario | Recommendation |
|----------|----------------|
| Quick prototype | Naive Bayes |
| Need higher accuracy | SVM |
| Very large dataset (100K+) | Naive Bayes (faster) |
| Complex patterns | SVM |
| Limited computing power | Naive Bayes |
| Production system | Try both, pick best |

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING                                 │
├─────────────────────────────────────────────────────────────────┤
│  Raw Text Data                                                   │
│       ↓                                                          │
│  Preprocessing (lowercase, remove special chars)                 │
│       ↓                                                          │
│  TF-IDF Vectorization (text → 10000 numbers)                    │
│       ↓                                                          │
│  Train/Test Split (80%/20%)                                      │
│       ↓                                                          │
│  SVM Training (find optimal boundary using support vectors)      │
│       ↓                                                          │
│  Trained Model (stored support vectors + boundary)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        PREDICTION                                │
├─────────────────────────────────────────────────────────────────┤
│  New Bug Report: "App crashes when saving"                       │
│       ↓                                                          │
│  Preprocess: "app crashes when saving"                           │
│       ↓                                                          │
│  TF-IDF: [0.0, 0.8, 0.0, 0.5, ...]                              │
│       ↓                                                          │
│  SVM: Calculate distance to boundary                             │
│       ↓                                                          │
│  Output: "critical" (confidence: 81.2%)                          │
└─────────────────────────────────────────────────────────────────┘
```
