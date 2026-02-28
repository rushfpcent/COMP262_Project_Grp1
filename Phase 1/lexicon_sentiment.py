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
from Models.swn_model import run_swn_model
from Models.vader_model import run_vader
from sklearn.metrics import classification_report, accuracy_score
import os

# Ensure necessary NLTK resources are downloaded for SentiWordNet
# Adding 'omw-1.4' as it is often required by newer versions of wordnet
nltk.download(['sentiwordnet', 'wordnet', 'averaged_perceptron_tagger', 'punkt', 'omw-1.4'], quiet=True)

# Importing your custom project modules
from loader import load_data
from basic_preprocess import preprocess_data, sample_data

SEPARATOR = "=" * 64

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
    
    for model_name, pred_col in [("VADER", "vader_pred"), ("SentiWordNet", "swn_prediction")]:
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
    print(classification_report(df['sentiment'], df['swn_prediction']))
    
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
    df_results = run_swn_model(df_sampled)

    # 5. Generate Step 7 Comparison Table and Detailed Reports
    generate_comparison(df_results)
    
    print(df_results[['clean_swn', 'sentiment', 'swn_prediction', 'swn_score', 'vader_pred', 'vader_score']].head())

    df_results.to_csv("phase1_lexicon_results.csv", index=False)
    print("\nResults exported to 'phase1_lexicon_results.csv'")

    print(f"\n{SEPARATOR}")
    print("Phase 1 Modeling Complete!")
    print(SEPARATOR)