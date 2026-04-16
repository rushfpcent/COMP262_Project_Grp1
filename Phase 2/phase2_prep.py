import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Importing your existing Phase 1 modules
from loader import load_data
from basic_preprocess import preprocess_data

SEPARATOR = "=" * 64

def prepare_phase2_data(file_path):
    print(f"\n{SEPARATOR}")
    print("PHASE 2: DATA PREPARATION & SPLITTING")
    print(SEPARATOR)

    # 1. Load the dataset
    df_raw = load_data(path=file_path)

    # 2. Re-use your Phase 1 preprocessing
    df_clean = preprocess_data(df_raw)

    # 3. Handle the Rubric Size Constraint Safely
    available_rows = len(df_clean)
    print(f"\nNOTE: The cleaned dataset only has {available_rows} unique reviews.")
    print("Using all available reviews instead of sampling 2000+.")
    df_sample = df_clean.copy()

    # 4. Stratified 70/30 Split
    print("\nPerforming Stratified 70/30 Split...")
    X = df_sample['clean_swn'] 
    y = df_sample['sentiment'] 

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=42, 
        stratify=df_sample['overall'] 
    )

    print(f"  -> Training Set: {len(X_train_text)} reviews (70%)")
    print(f"  -> Testing Set:  {len(X_test_text)} reviews (30%)")

    # 5. Requirement 11c: Text Representation (TF-IDF)
    print("\nApplying TF-IDF Text Representation...")
    X_train_joined = X_train_text.apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    X_test_joined = X_test_text.apply(lambda x: " ".join(x) if isinstance(x, list) else x)

    tfidf = TfidfVectorizer(max_features=5000) 
    
    X_train_tfidf = tfidf.fit_transform(X_train_joined)
    X_test_tfidf = tfidf.transform(X_test_joined)

    print("  -> TF-IDF Vectorization Complete.")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, df_sample, X_test_text

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    # Pointed directly to your 5-core file
    FULL_DATA_PATH = os.path.join(current_dir, "Data", "AMAZON_FASHION.json")
    
    # Run the pipeline
    X_train, X_test, y_train, y_test, df_phase2, test_text = prepare_phase2_data(FULL_DATA_PATH)
    
    print(f"\n{SEPARATOR}")
    print("Phase 2 Data Prep Complete. Ready for Machine Learning!")
    print(SEPARATOR)