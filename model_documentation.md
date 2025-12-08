# Bug Report Severity Classification - Naive Bayes Model
## Complete Documentation (model.py)

---

## Table of Contents
1. [What This Program Does](#what-this-program-does)
2. [Key Concepts Explained](#key-concepts-explained)
3. [Line-by-Line Code Explanation](#line-by-line-code-explanation)
4. [How the Algorithm Works](#how-the-algorithm-works)
5. [Input/Output Examples](#inputoutput-examples)

---

## What This Program Does

This program automatically classifies bug reports into severity levels:
- **blocker**: Completely stops work, highest priority
- **critical**: Major functionality broken
- **major**: Significant issue but workaround exists
- **normal**: Standard bugs
- **minor**: Small issues
- **trivial**: Cosmetic issues, lowest priority

**Example:**
- Input: "Application crashes when saving files"
- Output: "critical" (confidence: 85%)

---

## Key Concepts Explained

### What is Machine Learning?
Machine learning is teaching computers to learn patterns from data instead of programming explicit rules. Like how a child learns to recognize dogs by seeing many dog pictures, ML models learn from examples.

### What is Text Classification?
Converting text into categories. Example: Is this email spam or not? Is this bug critical or minor?

### What is TF-IDF?
**TF-IDF = Term Frequency - Inverse Document Frequency**

Computers can't understand text directly - they need numbers. TF-IDF converts text to numbers by measuring word importance.

- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How rare/unique a word is across all documents

**Example:**
- "the" appears everywhere → Low importance (low IDF)
- "crash" appears only in critical bugs → High importance (high IDF)

### What is Naive Bayes?
A simple but effective algorithm based on probability. It asks: "Given these words, what's the probability this bug is critical vs normal?"

**Why "Naive"?** It assumes all words are independent (which isn't true, but works surprisingly well).

**Why "Bayes"?** Uses Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)

### What is ComplementNB?
A variant of Naive Bayes designed for **imbalanced datasets** (when some categories have way more examples than others). Instead of learning what makes a class, it learns what makes everything EXCEPT that class.

---

## Line-by-Line Code Explanation

### Lines 1-11: Importing Libraries

```python
import pandas as pd
```
**pandas** - Data manipulation library. Used to handle tabular data (like Excel spreadsheets).
- Creates DataFrames (tables with rows and columns)
- Used here to display confusion matrix nicely

```python
import numpy as np
```
**numpy** - Numerical computing library. Handles arrays and mathematical operations.
- Used for `np.max()` to find highest probability
- Much faster than Python lists for math

```python
import re
```
**re** - Regular Expressions library (built into Python). Used for text pattern matching.
- Used to clean text (remove special characters, extra spaces)
- `re.sub(pattern, replacement, text)` replaces patterns

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
**TfidfVectorizer** - Converts text to TF-IDF numerical features.
- Input: List of text strings ["bug report 1", "bug report 2"]
- Output: Matrix of numbers representing word importance

```python
from sklearn.model_selection import train_test_split
```
**train_test_split** - Splits data into training and testing sets.
- Training set: Model learns from this (80%)
- Testing set: We evaluate model accuracy on this (20%)
- Why split? To test on data the model hasn't seen before

```python
from sklearn.naive_bayes import ComplementNB
```
**ComplementNB** - The Complement Naive Bayes classifier.
- Better than regular Naive Bayes for imbalanced data
- Learns patterns to predict bug severity

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
**Evaluation metrics** - Measure how good our model is:
- `accuracy_score`: % of correct predictions
- `classification_report`: Detailed precision/recall/F1 per class
- `confusion_matrix`: Shows what the model predicted vs actual

```python
from sklearn.preprocessing import LabelEncoder
```
**LabelEncoder** - Converts text labels to numbers.
- "critical" → 0, "major" → 1, "minor" → 2, etc.
- ML models need numbers, not text

```python
from datasets import load_dataset
```
**load_dataset** - HuggingFace library to download datasets.
- Downloads the Eclipse bug reports dataset from the internet
- Returns data in a structured format

```python
import warnings
warnings.filterwarnings('ignore')
```
**warnings** - Suppresses warning messages for cleaner output.

---

### Lines 13-77: load_and_preprocess_data() Function

**Purpose:** Download bug report dataset and extract text + labels

```python
def load_and_preprocess_data():
```
Defines a function that loads and cleans the data.

```python
    dataset_names = [
        "AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset",
        "mozilla/bugzilla_eclipse",
        "eclipse/bugzilla",
        "bugzilla_eclipse"
    ]
```
List of possible dataset names to try. If first fails, try next.

```python
    dataset = None
    for name in dataset_names:
        try:
            dataset = load_dataset(name)
            break
        except Exception:
            continue
```
**Try-except loop**: Attempts to load each dataset name.
- If successful, `break` exits the loop
- If fails, `continue` tries the next name

```python
    if dataset is None:
        raise ValueError("Could not load any dataset from HuggingFace.")
```
If all dataset names failed, raise an error.

```python
    if 'train' in dataset:
        data = dataset['train']
    else:
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
```
Datasets have "splits" (train/test/validation). Get the training split.

```python
    df = data.to_pandas()
```
Convert HuggingFace dataset to pandas DataFrame for easier manipulation.

```python
    text_columns = ['Short Description', 'summary', 'description', ...]
    label_columns = ['Severity Label', 'Resolution Status', 'severity', ...]
```
Lists of possible column names. Different datasets use different names.

```python
    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break
```
Find which text column exists in this dataset.

```python
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
```
Extract the text and label columns as Python lists.
- `.astype(str)` ensures everything is text
- `.tolist()` converts DataFrame column to list

```python
    valid_indices = [i for i, (text, label) in enumerate(zip(texts, labels))
                    if text and text != 'nan' and text.strip() and label and label != 'nan']
```
**List comprehension** - Filter out empty/invalid rows.
- `enumerate` gives index + value
- `zip` pairs texts with labels
- Only keep rows where both text and label are valid

```python
    texts = texts[:1000]
    labels = labels[:1000]
```
Limit to 1000 samples for faster training.

```python
    return texts, labels
```
Return the cleaned text and labels.

---

### Lines 79-87: preprocess_text() Function

**Purpose:** Clean and normalize text data

```python
def preprocess_text(texts):
    processed_texts = []
    for text in texts:
```
Loop through each text to clean it.

```python
        text = text.lower()
```
Convert to lowercase. "CRASH" and "crash" should be treated the same.

```python
        text = re.sub(r'\s+', ' ', text)
```
**Regex**: Replace multiple spaces/tabs/newlines with single space.
- `\s+` means "one or more whitespace characters"

```python
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
```
**Regex**: Remove all special characters (keep only letters, numbers, spaces).
- `[^...]` means "NOT these characters"
- Removes: @#$%^&*()!? etc.

```python
        text = text.strip()
```
Remove leading/trailing whitespace.

```python
        processed_texts.append(text)
    return processed_texts
```
Add cleaned text to list, return all cleaned texts.

---

### Lines 89-111: prepare_data() Function

**Purpose:** Prepare data for machine learning (split, vectorize)

```python
def prepare_data():
    texts, labels = load_and_preprocess_data()
    texts = preprocess_text(texts)
```
Load data and clean the text.

```python
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
```
**LabelEncoder**:
- `fit_transform` learns the mapping AND converts in one step
- "blocker"→0, "critical"→1, "major"→2, etc.
- We save `label_encoder` to convert predictions back to text later

```python
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
```
**Split data 80/20**:
- `X_train`: Training texts (800 samples)
- `X_test`: Testing texts (200 samples)
- `y_train`: Training labels
- `y_test`: Testing labels
- `test_size=0.2`: 20% for testing
- `random_state=42`: Same split every run (reproducibility)
- `stratify=encoded_labels`: Keep same class proportions in both sets

```python
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
```
**TF-IDF Configuration**:
- `max_features=5000`: Only keep top 5000 most important words
- `stop_words='english'`: Remove common words ("the", "is", "and")
- `ngram_range=(1, 2)`: Use single words AND word pairs
  - "crash" (unigram) + "app crash" (bigram)
- `min_df=2`: Ignore words appearing in less than 2 documents
- `max_df=0.8`: Ignore words appearing in more than 80% of documents

```python
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
```
**Vectorization**:
- `fit_transform`: Learn vocabulary from training data AND convert to numbers
- `transform`: Convert test data using SAME vocabulary (don't learn new words)
- Result: Matrix where each row is a document, each column is a word's TF-IDF score

```python
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, label_encoder
```
Return everything needed for training and prediction.

---

### Lines 113-117: train_model() Function

**Purpose:** Train the Naive Bayes classifier

```python
def train_model(X_train, y_train):
    model = ComplementNB(alpha=1.0)
```
**Create ComplementNB model**:
- `alpha=1.0`: Smoothing parameter (prevents zero probabilities)
- Called "Laplace smoothing" - adds 1 to all word counts

```python
    model.fit(X_train, y_train)
```
**Train the model**:
- Model looks at training data and learns patterns
- "When these words appear, it's usually critical"
- "When those words appear, it's usually minor"

```python
    return model
```
Return the trained model.

---

### Lines 119-131: evaluate_model() Function

**Purpose:** Measure how good the model is

```python
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
```
**Make predictions** on test data (data model hasn't seen).

```python
    accuracy = accuracy_score(y_test, y_pred)
```
**Accuracy**: Correct predictions / Total predictions
- 0.75 means 75% correct

```python
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
```
Print accuracy with 4 decimal places.

```python
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```
**Classification Report** shows for each class:
- **Precision**: Of all "critical" predictions, how many were actually critical?
- **Recall**: Of all actual "critical" bugs, how many did we find?
- **F1-score**: Balance between precision and recall

```python
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df)
```
**Confusion Matrix**: Grid showing actual vs predicted.
- Rows = Actual labels
- Columns = Predicted labels
- Diagonal = Correct predictions
- Off-diagonal = Mistakes

---

### Lines 133-151: predict_custom_examples() Function

**Purpose:** Test model on sample bug descriptions

```python
    test_examples = [
        "The application crashes when I try to save my work",
        ...
    ]
```
List of sample bug descriptions to test.

```python
    processed_examples = preprocess_text(test_examples)
```
Clean the test examples same way as training data.

```python
    examples_tfidf = vectorizer.transform(processed_examples)
```
Convert text to TF-IDF numbers using the SAME vectorizer.

```python
    predictions = model.predict(examples_tfidf)
    prediction_probs = model.predict_proba(examples_tfidf)
```
- `predict`: Get predicted class (0, 1, 2...)
- `predict_proba`: Get probability for EACH class

```python
    predicted_labels = label_encoder.inverse_transform(predictions)
```
Convert numbers back to text labels (0 → "blocker").

---

### Lines 153-182: interactive_prediction_loop() Function

**Purpose:** Let user type bug descriptions and get predictions

```python
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
```
Infinite loop asking for user input.

```python
            if user_input.lower() == 'exit':
                break
```
Exit if user types "exit".

```python
            processed_input = preprocess_text([user_input])
            input_tfidf = vectorizer.transform(processed_input)
```
Clean user input and convert to TF-IDF.

```python
            prediction = model.predict(input_tfidf)[0]
            prediction_probs = model.predict_proba(input_tfidf)[0]
```
Get prediction and probabilities. `[0]` gets first (only) result.

```python
            predicted_label = label_encoder.inverse_transform([prediction])[0]
```
Convert prediction number to text label.

```python
            sorted_probs = sorted(zip(label_encoder.classes_, prediction_probs),
                                  key=lambda x: x[1], reverse=True)
```
Sort classes by probability (highest first).

```python
            for label, prob in sorted_probs:
                bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                print(f"   {label:12s}: {bar} {prob:.3f}")
```
Print visual bar chart of probabilities.
- `█` filled = probability amount
- `░` empty = remaining

---

### Lines 184-198: main() Function

**Purpose:** Run the complete pipeline

```python
def main():
    try:
        X_train, X_test, y_train, y_test, vectorizer, label_encoder = prepare_data()
```
Step 1: Load and prepare data.

```python
        model = train_model(X_train, y_train)
```
Step 2: Train the model.

```python
        accuracy, predictions = evaluate_model(model, X_test, y_test, label_encoder)
```
Step 3: Evaluate on test data.

```python
        predict_custom_examples(model, vectorizer, label_encoder)
```
Step 4: Test on sample examples.

```python
        interactive_prediction_loop(model, vectorizer, label_encoder)
```
Step 5: Interactive mode for user input.

```python
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
```
Catch and display any errors with full details.

```python
if __name__ == "__main__":
    main()
```
Run `main()` only if this file is executed directly (not imported).

---

## How the Algorithm Works

### Training Phase
```
Bug Reports → Clean Text → TF-IDF Numbers → ComplementNB Learning

"App crashes"     →  "app crashes"  →  [0.5, 0.8, 0.0, ...]  →  Model learns:
(label: critical)                                                "crash" → critical
```

### Prediction Phase
```
New Bug Report → Clean Text → TF-IDF Numbers → Model Prediction

"System crash"  →  "system crash"  →  [0.4, 0.9, 0.0, ...]  →  "critical" (85%)
```

---

## Input/Output Examples

### Input
```
Enter text: Application crashes when opening large files
```

### Output
```
Predicted: critical (confidence: 0.823)
   critical    : ████████████████░░░░ 0.823
   blocker     : ██░░░░░░░░░░░░░░░░░░ 0.095
   major       : █░░░░░░░░░░░░░░░░░░░ 0.042
   normal      : ░░░░░░░░░░░░░░░░░░░░ 0.025
   minor       : ░░░░░░░░░░░░░░░░░░░░ 0.010
   trivial     : ░░░░░░░░░░░░░░░░░░░░ 0.005
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| TF-IDF | Convert text to numbers |
| ComplementNB | Learn patterns, make predictions |
| LabelEncoder | Convert labels text↔numbers |
| train_test_split | Split data for training/testing |
| Classification Report | Measure model performance |

**Flow:** Data → Clean → Vectorize → Train → Predict → Evaluate
