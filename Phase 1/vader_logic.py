"""
Module for running Lexicon-based sentiment analysis models (VADER and SentiWordNet).
Includes evaluation and comparison metrics for Phase 1 of the COMP 262 Project.

Usage:
    cd "COMP262_PROJECT_GRP1/Phase 1"
    python vader_logic.py
"""

import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score
import os

# Ensure necessary NLTK resources are downloaded for SentiWordNet
# Adding 'omw-1.4' as it is often required by newer versions of wordnet
nltk.download(['sentiwordnet', 'wordnet', 'averaged_perceptron_tagger', 'punkt', 'omw-1.4'], quiet=True)

from loader import load_data
from basic_preprocess import preprocess_data, sample_data

SEPARATOR = "=" * 64

# ================================================================
# Step 6a: VADER Model Implementation
# ================================================================
def run_vader(df):
    """
    VADER is chosen because it is specifically designed for social media 
    and product reviews, handling emojis, capitalization, and punctuation well.
    """
    print(f"\n{SEPARATOR}")
    print("Running VADER Lexicon analysis...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    def get_vader_label(text):
        # Ensure text is parsed as a string to avoid float errors
        scores = analyzer.polarity_scores(str(text))
        compound = scores['compound']
        
        # Compound score thresholds: Positive >= 0.05, Negative <= -0.05
        if compound >= 0.05: 
            return "Positive"
        elif compound <= -0.05: 
            return "Negative"
        else: 
            return "Neutral"

    # Applying to the 'clean_vader' column which preserved punctuation/casing
    df['vader_pred'] = df['clean_vader'].apply(get_vader_label)
    print("  -> VADER predictions complete.")
    return df

# ================================================================
# Step 6b: SentiWordNet Model Implementation
# ================================================================
def run_swn(df):
    """
    SentiWordNet is chosen for its deep linguistic coverage and 
    synset-based scoring of lemmatized text.
    """
    print("Running SentiWordNet Lexicon analysis...")
    
    def get_swn_label(tokens):
        sentiment_score = 0
        # Check if tokens is a valid list (it should be from preprocess_for_swn)
        if isinstance(tokens, list):
            for token in tokens:
                synsets = list(swn.senti_synsets(token))
                if synsets:
                    # Using the first (most common) synset score
                    sentiment_score += synsets[0].pos_score() - synsets[0].neg_score()
        
        # Scoring logic
        if sentiment_score > 0: 
            return "Positive"
        elif sentiment_score < 0: 
            return "Negative"
        else: 
            return "Neutral"

    # Applying to the 'clean_swn' column which contains lemmatized lists of tokens
    df['swn_pred'] = df['clean_swn'].apply(get_swn_label)
    print("  -> SentiWordNet predictions complete.")
    return df

# ================================================================
# Step 7: Validation and Comparison Table
# ================================================================
def generate_comparison(df):
    """
    Validates results of both models against the ground truth 'sentiment' column
    and provides a comparison table.
    """
    print(f"\n{SEPARATOR}")
    print("PHASE 1: LEXICON COMPARISON TABLE")
    print(SEPARATOR)
    
    metrics = []
    
    for model_name, pred_col in [("VADER", "vader_pred"), ("SentiWordNet", "swn_pred")]:
        acc = accuracy_score(df['sentiment'], df[pred_col])
        metrics.append({
            "Model": model_name,
            "Accuracy": f"{acc:.2%}",
            "Sample Size": len(df)
        })
    
    comparison_table = pd.DataFrame(metrics)
    print(comparison_table.to_string(index=False))
    
    print(f"\n{SEPARATOR}")
    print("DETAILED CLASSIFICATION REPORTS")
    print(SEPARATOR)
    
    print("\nDetailed VADER Report:\n")
    print(classification_report(df['sentiment'], df['vader_pred']))
    
    print("\nDetailed SentiWordNet Report:\n")
    print(classification_report(df['sentiment'], df['swn_pred']))
    
    return comparison_table

# ================================================================
# Main Execution Block
# ================================================================
if __name__ == "__main__":
    # 1. Load the raw data
    df_raw = load_data()
    
    # 2. Pre-process the entire dataset (cleaning, outlier flagging, target labeling)
    df_processed = preprocess_data(df_raw)
    
    # 3. Randomly sample exactly 1000 reviews for Phase 1 Lexicon Models
    df_sampled = sample_data(df_processed, n=1000, random_seed=1)
    
    # 4. Run Both Lexicon Models
    df_results = run_vader(df_sampled)
    df_results = run_swn(df_results)
    
    # 5. Generate Step 7 Comparison Table and Detailed Reports
    generate_comparison(df_results)
    
    print(f"\n{SEPARATOR}")
    print("Phase 1 Modeling Complete!")
    print(SEPARATOR)