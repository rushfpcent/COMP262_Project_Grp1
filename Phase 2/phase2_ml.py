import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Import the data preparation pipeline we just built
from phase2_prep import prepare_phase2_data

SEPARATOR = "=" * 64

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print(f"\n{SEPARATOR}")
    print("PHASE 2: MACHINE LEARNING MODELS (Logistic Regression & SVM)")
    print(SEPARATOR)

    # ================================================================
    # MODEL 1: Logistic Regression
    # ================================================================
    print("\nTraining Logistic Regression Model (70% of data)...")
    # 'balanced' weights help the model handle the imbalanced 5-star heavy dataset
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    log_reg.fit(X_train, y_train)
    
    print("Testing Logistic Regression Model (30% of data)...")
    log_reg_preds = log_reg.predict(X_test)
    
    print("\n" + "-"*40)
    print("LOGISTIC REGRESSION RESULTS")
    print("-"*40)
    print(f"Accuracy: {accuracy_score(y_test, log_reg_preds):.2%}")
    print("\nConfusion Matrix:")
    # Specifying labels ensures the matrix prints in a consistent order
    cm_labels = ["Positive", "Neutral", "Negative"]
    print(pd.DataFrame(confusion_matrix(y_test, log_reg_preds, labels=cm_labels), 
                       index=[f"True {label}" for label in cm_labels], 
                       columns=[f"Pred {label}" for label in cm_labels]))
    print("\nClassification Report (Precision, Recall, F1):")
    print(classification_report(y_test, log_reg_preds))


    # ================================================================
    # MODEL 2: Multinomial Naive Bayes
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("Training Multinomial Naive Bayes Model (70% of data)...")
    # kernel='linear' is highly effective for high-dimensional TF-IDF text data
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    print("Testing Multinomial Naive Bayes Model (30% of data)...")
    nb_preds = nb_model.predict(X_test)

    print("\n" + "-"*40)
    print("MULTINOMIAL NAIVE BAYES RESULTS")
    print("-"*40)
    print(f"Accuracy: {accuracy_score(y_test, nb_preds):.2%}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, nb_preds, labels=cm_labels), 
                       index=[f"True {label}" for label in cm_labels], 
                       columns=[f"Pred {label}" for label in cm_labels]))
    print("\nClassification Report (Precision, Recall, F1):")
    print(classification_report(y_test, nb_preds))
    
    return log_reg, nb_model, log_reg_preds, nb_preds

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    FULL_DATA_PATH = os.path.join(current_dir, "Data", "AMAZON_FASHION.json")
    
    # 1. Get the prepared TF-IDF data from the previous script
    X_train_tfidf, X_test_tfidf, y_train, y_test, df_sample, X_test_text = prepare_phase2_data(FULL_DATA_PATH)
    
    # 2. Train and Evaluate the ML Models
    lr_model, nb_model, lr_preds, nb_preds = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    print(f"\n{SEPARATOR}")
    print("Phase 2 ML Training Complete! Copy these metrics into your Word Report.")
    print(SEPARATOR)