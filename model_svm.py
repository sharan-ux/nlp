"""
Bug Report Severity Classification using Support Vector Machine (SVM)
=====================================================================

WHAT THIS CODE DOES:
-------------------
This script creates an intelligent bug report classifier using Support Vector Machine
that can automatically determine the severity level of software bug reports. It uses 
real-world data from Eclipse project bug reports to train a machine learning model.

HOW IT WORKS:
------------
1. DATA LOADING:
   - Downloads 88,682+ real bug reports from HuggingFace dataset
   - Uses "AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset"
   - Extracts bug descriptions and their severity labels

2. TEXT PREPROCESSING:
   - Cleans text (lowercase, remove special characters)
   - Converts text to numerical features using TF-IDF
   - TF-IDF captures word importance and frequency patterns

3. MACHINE LEARNING:
   - Uses Support Vector Machine (SVM) classifier with RBF kernel
   - SVM finds optimal decision boundaries between severity classes
   - Trains on 80% of data, tests on 20%
   - Learns complex patterns between bug descriptions and severity levels

4. EVALUATION:
   - Shows accuracy, precision, recall, F1-score
   - Displays confusion matrix to see classification errors
   - Tests on sample bug reports

5. INTERACTIVE MODE:
   - Lets you type any bug description
   - Predicts severity level (normal, trivial, critical, etc.)
   - Shows confidence scores and probability distributions
   - Continues until you type "exit"

EXAMPLE WORKFLOW:
----------------
Input:  "Application crashes when saving large files"
Output: Severity: "critical" (Confidence: 91.7%)

Input:  "Minor typo in help text"
Output: Severity: "trivial" (Confidence: 96.3%)

SVM vs NAIVE BAYES:
------------------
- SVM: Better for complex patterns, handles non-linear relationships
- SVM: Often more accurate but slower to train
- SVM: Good with high-dimensional data (like TF-IDF features)
- SVM: Uses kernel tricks to find optimal decision boundaries

TECHNICAL COMPONENTS:
--------------------
- Dataset: HuggingFace Bugzilla Eclipse Bug Reports (88K+ samples)
- Features: TF-IDF vectors (Term Frequency-Inverse Document Frequency)
- Algorithm: Support Vector Machine with RBF kernel
- Evaluation: Accuracy, Classification Report, Confusion Matrix
- Interface: Interactive command-line prediction loop

REQUIREMENTS:
------------
- pandas: Data manipulation
- numpy: Numerical operations  
- scikit-learn: Machine learning algorithms
- datasets: HuggingFace dataset loading

Author: SVM implementation for bug severity classification
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load the HuggingFace dataset and preprocess it for text classification.
    
    Returns:
        tuple: (texts, labels) - preprocessed text data and corresponding labels
    """
    print("Loading dataset from HuggingFace...")
    
    # Try to load the specific Bugzilla Eclipse dataset from HuggingFace
    dataset_names = [
        "AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset",
        # Backup options in case the main dataset has issues
        "mozilla/bugzilla_eclipse",
        "eclipse/bugzilla", 
        "bugzilla_eclipse"
    ]
    
    dataset = None
    dataset_name = None
    
    for name in dataset_names:
        try:
            print(f"Trying to load dataset: {name}")
            dataset = load_dataset(name)
            dataset_name = name
            print(f"Successfully loaded dataset: {name}")
            break
        except Exception as e:
            print(f"Failed to load {name}: {str(e)[:100]}...")
            continue
    
    # If no dataset found, raise an error with helpful message
    if dataset is None:
        error_msg = """
ERROR: Could not load any external dataset from HuggingFace.

Please try one of these options:

1. Use a specific dataset name:
   - Check HuggingFace Hub for available bug report datasets
   - Update the dataset_names list in the code with the correct name

2. Use a different dataset format:
   - CSV file: Use pandas.read_csv('your_file.csv')
   - JSON file: Use pandas.read_json('your_file.json')
   
3. Popular alternative datasets you can try:
   - 'imdb' (movie reviews)
   - 'ag_news' (news classification)
   - 'yelp_review_full' (restaurant reviews)

Example: Change the dataset name in load_dataset() to one of the above.
        """
        raise ValueError(error_msg)
    
    # Extract the training split (or first available split)
    if 'train' in dataset:
        data = dataset['train']
        print("Using 'train' split")
    else:
        # Use the first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Using split: {split_name}")
    
    # Convert to pandas DataFrame for easier manipulation
    df = data.to_pandas()
    
    # Print dataset info for debugging
    print(f"Dataset shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head(3))
    
    # Extract text and labels based on common column names
    text_columns = ['Short Description', 'summary', 'description', 'title', 'text', 'short_desc', 'body', 'content', 'message']
    label_columns = ['Severity Label', 'Resolution Status', 'severity', 'priority', 'status', 'label', 'component', 'category', 'class', 'type']
    
    # Find the best text column
    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            print(f"Found text column: {text_col}")
            break
    
    # Find the best label column
    label_col = None
    for col in label_columns:
        if col in df.columns:
            label_col = col
            print(f"Found label column: {label_col}")
            break
    
    if text_col is None:
        available_cols = df.columns.tolist()
        raise ValueError(f"Could not find text column. Available columns: {available_cols}")
    
    if label_col is None:
        available_cols = df.columns.tolist()
        raise ValueError(f"Could not find label column. Available columns: {available_cols}")
    
    print(f"Using text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    
    # Extract texts and labels
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    
    # Remove any rows with missing data
    valid_indices = [i for i, (text, label) in enumerate(zip(texts, labels)) 
                    if text and text != 'nan' and text.strip() and label and label != 'nan']
    
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    print(f"After cleaning: {len(texts)} samples")
    print(f"Unique labels ({len(set(labels))}): {sorted(set(labels))}")
    
    if len(texts) == 0:
        raise ValueError("No valid text samples found after cleaning!")
    
    return texts, labels

def preprocess_text(texts):
    """
    Preprocess text data by cleaning and normalizing.
    
    Args:
        texts (list): List of text strings to preprocess
        
    Returns:
        list: Preprocessed text strings
    """
    print("Preprocessing text data...")
    
    processed_texts = []
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        processed_texts.append(text)
    
    print(f"Preprocessed {len(processed_texts)} text samples")
    return processed_texts

def prepare_data():
    """
    Load, preprocess, and split the dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer, label_encoder)
    """
    # Load and preprocess data
    texts, labels = load_and_preprocess_data()
    texts = preprocess_text(texts)
    
    # Encode labels to numerical format
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,     # Increased for SVM (can handle more features)
        stop_words='english',   # Remove common English stop words
        ngram_range=(1, 2),     # Use unigrams and bigrams
        min_df=2,               # Ignore terms that appear in less than 2 documents
        max_df=0.8              # Ignore terms that appear in more than 80% of documents
    )
    
    # Transform text to TF-IDF features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, label_encoder

def train_model(X_train, y_train):
    """
    Train the Support Vector Machine classifier.
    
    Args:
        X_train: Training features (TF-IDF matrix)
        y_train: Training labels
        
    Returns:
        SVC: Trained SVM classifier
    """
    print("Training Support Vector Machine classifier...")
    print("Note: SVM training may take longer than Naive Bayes but often provides better accuracy")
    
    # Create and train the SVM model
    # Using RBF kernel for non-linear classification
    # probability=True enables probability estimates for confidence scores
    model = SVC(
        kernel='rbf',           # Radial Basis Function kernel for non-linear patterns
        C=1.0,                  # Regularization parameter
        gamma='scale',          # Kernel coefficient
        probability=True,       # Enable probability estimates
        random_state=42         # For reproducible results
    )
    
    model.fit(X_train, y_train)
    
    print("SVM model training completed!")
    print(f"Number of support vectors: {model.n_support_}")
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model and print detailed metrics.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True test labels
        label_encoder: Label encoder for converting back to original labels
    """
    print("\n" + "="*50)
    print("SVM MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print("-" * 40)
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 20)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)
    
    return accuracy, y_pred

def predict_custom_examples(model, vectorizer, label_encoder):
    """
    Make predictions on custom text examples.
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Label encoder for converting predictions back to original labels
    """
    print("\n" + "="*50)
    print("SAMPLE SVM PREDICTIONS")
    print("="*50)
    
    # Custom examples to test
    test_examples = [
        "The application crashes when I try to save my work",
        "Please add a dark mode feature to the interface", 
        "Minor text alignment issue in the header section",
        "System becomes completely unresponsive during file upload",
        "Would be nice to have export to Excel functionality",
        "Database connection timeout causing data loss",
        "Small typo in the help documentation",
        "Add support for multiple file formats in import",
        "Critical security vulnerability found in login system",
        "UI button is slightly misaligned on mobile devices"
    ]
    
    # Preprocess the examples
    processed_examples = preprocess_text(test_examples)
    
    # Transform to TF-IDF features
    examples_tfidf = vectorizer.transform(processed_examples)
    
    # Make predictions
    predictions = model.predict(examples_tfidf)
    prediction_probs = model.predict_proba(examples_tfidf)
    
    # Convert predictions back to original labels
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    print("Sample Test Examples and SVM Predictions:")
    print("-" * 60)
    for i, (text, pred_label, probs) in enumerate(zip(test_examples[:5], predicted_labels[:5], prediction_probs[:5])):
        max_prob = np.max(probs)
        print(f"{i+1:2d}. Text: {text}")
        print(f"    SVM Predicted: {pred_label} (confidence: {max_prob:.3f})")
        print()

def interactive_prediction_loop(model, vectorizer, label_encoder):
    """
    Interactive loop for user to input custom text and get SVM predictions.
    
    Args:
        model: Trained SVM classifier
        vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Label encoder for converting predictions back to original labels
    """
    print("\n" + "="*60)
    print("INTERACTIVE SVM PREDICTION MODE")
    print("="*60)
    print("Enter bug reports or descriptions to classify their severity using SVM.")
    print("Type 'exit' to quit the interactive mode.")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your text: ").strip()
            
            # Check if user wants to exit
            if user_input.lower() == 'exit':
                print("\nExiting SVM interactive mode. Goodbye!")
                break
            
            # Check if input is empty
            if not user_input:
                print("Please enter some text or type 'exit' to quit.")
                continue
            
            # Preprocess the input
            processed_input = preprocess_text([user_input])
            
            # Transform to TF-IDF features
            input_tfidf = vectorizer.transform(processed_input)
            
            # Make prediction
            prediction = model.predict(input_tfidf)[0]
            prediction_probs = model.predict_proba(input_tfidf)[0]
            
            # Convert prediction back to original label
            predicted_label = label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(prediction_probs)
            
            # Get all class probabilities for detailed output
            all_labels = label_encoder.classes_
            prob_dict = {label: prob for label, prob in zip(all_labels, prediction_probs)}
            
            # Display results
            print(f"\n📝 Input: {user_input}")
            print(f"🎯 SVM Predicted Severity: {predicted_label}")
            print(f"📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Show all class probabilities if there are multiple classes
            if len(all_labels) > 1:
                print("📈 SVM probability scores:")
                sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_probs:
                    bar_length = int(prob * 20)  # Scale to 20 characters
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"   {label:12s}: {bar} {prob:.3f}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError processing input: {e}")
            print("Please try again or type 'exit' to quit.")

def main():
    """
    Main function to run the complete SVM text classification pipeline.
    """
    print("="*60)
    print("SVM TEXT CLASSIFICATION WITH HUGGINGFACE DATASET")
    print("="*60)
    
    try:
        # Step 1: Prepare the data
        X_train, X_test, y_train, y_test, vectorizer, label_encoder = prepare_data()
        
        # Step 2: Train the SVM model
        model = train_model(X_train, y_train)
        
        # Step 3: Evaluate the model
        accuracy, predictions = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Step 4: Test on sample examples
        predict_custom_examples(model, vectorizer, label_encoder)
        
        # Step 5: Interactive SVM prediction loop
        interactive_prediction_loop(model, vectorizer, label_encoder)
        
        print("\n" + "="*60)
        print("SVM PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Final SVM Model Accuracy: {accuracy:.4f}")
        print("="*60)
        
    except ValueError as e:
        print(f"\nDataset Error: {e}")
        print("\nPlease check the dataset name or try a different dataset.")
        
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
