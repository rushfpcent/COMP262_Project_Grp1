import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing your existing Phase 1 modules
from loader import load_data
from basic_preprocess import preprocess_data, generate_report

SEPARATOR = "=" * 64

def prepare_phase2_data(file_path):
    """
    Handles Deliverables 11a, 11b, 11c, and 11d
    """
    print(f"\n{SEPARATOR}")
    print("PHASE 2: DATA PREPARATION & SPLITTING")
    print(SEPARATOR)

    # 1. Load the FULL dataset (Make sure file_path points to the full JSON)
    df_raw = load_data(path=file_path)

    # 2. Re-use your Phase 1 preprocessing (Handles 11b)
    # This automatically cleans text, flags outliers, and drops duplicates
    df_clean = preprocess_data(df_raw)

    # 3. Requirement 11a: Select a subset of minimum 2000 reviews
    # Let's sample 3000 to be safe and ensure a good training size
    print("\nSampling 3,000 reviews for Phase 2...")
    df_sample = df_clean.sample(n=3000, random_state=42).reset_index(drop=True)

    # 4. Requirement 11d: Stratified 70/30 Split based on 'overall' rating
    print("Performing Stratified 70/30 Split...")
    X = df_sample['clean_swn'] # We use the SentiWordNet clean column (lemmatized, no stopwords) for ML
    y = df_sample['sentiment'] # Target variable

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=42, 
        stratify=df_sample['overall'] # Stratifying based on the raw star rating
    )

    print(f"  -> Training Set: {len(X_train_text)} reviews (70%)")
    print(f"  -> Testing Set:  {len(X_test_text)} reviews (30%)")

    # 5. Requirement 11c: Text Representation (TF-IDF)
    # ML Models need numbers, not words. TF-IDF is standard for this.
    print("\nApplying TF-IDF Text Representation...")
    
    # We join the list of tokens back into a single string for the Vectorizer
    X_train_joined = X_train_text.apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    X_test_joined = X_test_text.apply(lambda x: " ".join(x) if isinstance(x, list) else x)

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000) # Keep top 5000 words to prevent memory crashes
    
    # Fit on training data ONLY to prevent data leakage, then transform both
    X_train_tfidf = tfidf.fit_transform(X_train_joined)
    X_test_tfidf = tfidf.transform(X_test_joined)

    print("  -> TF-IDF Vectorization Complete.")
    
    # Return everything needed for modeling
    return X_train_tfidf, X_test_tfidf, y_train, y_test, df_sample, X_test_text


if __name__ == "__main__":
    # Point this to your new, full dataset file
    FULL_DATA_PATH = "Data/AMAZON_FASHION.json" 
    
    # Run the pipeline
    X_train, X_test, y_train, y_test, df_phase2, test_text = prepare_phase2_data(FULL_DATA_PATH)
    
    # Optional: You can run your existing explore.py logic specifically on df_phase2 
    # to fulfill the "Carry out data exploration on the subset" requirement.
    print(f"\n{SEPARATOR}")
    print("Phase 2 Data Prep Complete. Ready for Machine Learning!")
    print(SEPARATOR)