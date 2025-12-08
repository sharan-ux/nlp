import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    dataset_names = [
        "AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset",
        "mozilla/bugzilla_eclipse",
        "eclipse/bugzilla",
        "bugzilla_eclipse"
    ]

    dataset = None
    for name in dataset_names:
        try:
            dataset = load_dataset(name)
            break
        except Exception:
            continue

    if dataset is None:
        raise ValueError("Could not load any dataset from HuggingFace.")

    if 'train' in dataset:
        data = dataset['train']
    else:
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]

    df = data.to_pandas()

    text_columns = ['Short Description', 'summary', 'description', 'title', 'text', 'short_desc', 'body', 'content', 'message']
    label_columns = ['Severity Label', 'Resolution Status', 'severity', 'priority', 'status', 'label', 'component', 'category', 'class', 'type']

    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break

    label_col = None
    for col in label_columns:
        if col in df.columns:
            label_col = col
            break

    if text_col is None:
        raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")

    if label_col is None:
        raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    valid_indices = [i for i, (text, label) in enumerate(zip(texts, labels))
                    if text and text != 'nan' and text.strip() and label and label != 'nan']

    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    # Limit to 1000 samples for faster training
    texts = texts[:1000]
    labels = labels[:1000]

    if len(texts) == 0:
        raise ValueError("No valid text samples found after cleaning!")

    return texts, labels

def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.strip()
        processed_texts.append(text)
    return processed_texts

def prepare_data():
    texts, labels = load_and_preprocess_data()
    texts = preprocess_text(texts)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, label_encoder

def train_model(X_train, y_train):
    # ComplementNB works better for imbalanced datasets
    model = ComplementNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df)

    return accuracy, y_pred

def predict_custom_examples(model, vectorizer, label_encoder):
    test_examples = [
        "The application crashes when I try to save my work",
        "Please add a dark mode feature to the interface",
        "Minor text alignment issue in the header section",
        "System becomes completely unresponsive during file upload",
        "Would be nice to have export to Excel functionality",
    ]

    processed_examples = preprocess_text(test_examples)
    examples_tfidf = vectorizer.transform(processed_examples)
    predictions = model.predict(examples_tfidf)
    prediction_probs = model.predict_proba(examples_tfidf)
    predicted_labels = label_encoder.inverse_transform(predictions)

    print("\nSample Predictions:")
    for i, (text, pred_label, probs) in enumerate(zip(test_examples, predicted_labels, prediction_probs)):
        print(f"{i+1}. {text}")
        print(f"   Predicted: {pred_label} (confidence: {np.max(probs):.3f})\n")

def interactive_prediction_loop(model, vectorizer, label_encoder):
    print("\nInteractive Mode - Type 'exit' to quit")

    while True:
        try:
            user_input = input("\nEnter text: ").strip()

            if user_input.lower() == 'exit':
                break

            if not user_input:
                continue

            processed_input = preprocess_text([user_input])
            input_tfidf = vectorizer.transform(processed_input)
            prediction = model.predict(input_tfidf)[0]
            prediction_probs = model.predict_proba(input_tfidf)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            print(f"Predicted: {predicted_label} (confidence: {np.max(prediction_probs):.3f})")

            sorted_probs = sorted(zip(label_encoder.classes_, prediction_probs), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs:
                bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                print(f"   {label:12s}: {bar} {prob:.3f}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    try:
        X_train, X_test, y_train, y_test, vectorizer, label_encoder = prepare_data()
        model = train_model(X_train, y_train)
        accuracy, predictions = evaluate_model(model, X_test, y_test, label_encoder)
        predict_custom_examples(model, vectorizer, label_encoder)
        interactive_prediction_loop(model, vectorizer, label_encoder)
        print(f"\nFinal Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
