import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
    
    def get_vader_score(text):
        # Ensure text is parsed as a string to avoid float errors
        scores = analyzer.polarity_scores(str(text))
        return scores['compound']

    def get_vader_label(score):        
        # Compound score thresholds: Positive >= 0.05, Negative <= -0.05
        if score >= 0.05: 
            return "Positive"
        elif score <= -0.05: 
            return "Negative"
        else: 
            return "Neutral"

    # Saving numerical score
    df['vader_score'] = df['clean_vader'].apply(get_vader_score)

    # Applying to the 'clean_vader' column which preserved punctuation/casing

    df['vader_pred'] = df['vader_score'].apply(get_vader_label)
    print("  -> VADER predictions complete.")
    return df
