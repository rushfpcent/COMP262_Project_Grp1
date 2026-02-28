"""
Script to preprocess data in preparation for sentiment analysis.

Imports cleaned DataFrame from loader.py.

Usage by future components:
    from basic_preprocess import preprocess_data
    df = preprocess_data()

Usage as standalone to see output and generate figures:
    cd "COMP262_PROJECT_GRP1/Phase 1"
    python basic_preprocess.py
"""

#Importing Libraries
import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Importing loader function
from loader import load_data

#================================================================
#Set Up
#================================================================
#Creating folder to hold figures
FIGURES_FOLDER = os.path.join(os.path.dirname(__file__), "figures")
if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER)

#Defining reusable constants
PALETTE = "viridis"
SEPARATOR = "-" * 64

#Setting plt style
plt.style.use("seaborn-v0_8-whitegrid")

#Defining colors for sentiment categories
SENTIMENT_COLORS = {
    "Positive": "#4C72B0",
    "Neutral" : "#DD8452",
    "Negative": "#C44E52",
}
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def basic_text_clean(text: str) -> str:
    """
    Light text cleaning safe for all lexicon pipelines.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_vader(text: str) -> str:
    """
    Minimal preprocessing for VADER, which relies on punctuation and casing cues.
    """
    return basic_text_clean(text)


def preprocess_for_swn(text: str) -> list[str]:
    """
    Full linguistic normalization is required for lexicon matching in SentiWordNet pipelines.
    """
    cleaned = basic_text_clean(text).lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(cleaned)
    processed_tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOP_WORDS]
    return processed_tokens


#================================================================
#Preprocessing Function
#================================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies sentiment labeling, selects columns, and flags outliers.
    """

    print(f"\n{SEPARATOR}")
    print("Starting preprocessing...")
    print(SEPARATOR)

    df = df.copy()
    reviewer_ids = df["reviewerID"].copy() if "reviewerID" in df.columns else None

    #Sentiment Labeling
    def label_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"
        
    df["sentiment"] = df["overall"].apply(label_sentiment)

    #Selecting relevant columns
    #Combining summary and reviewText into single column
    df["combined_text"] = (
        df["summary"].fillna("") + " " + df["reviewText"].fillna("")
    ).str.strip()

    #Defining selected columns
    SELECTED_COLUMNS = [
        "combined_text", #Merged text - primary model input
        "sentiment",     #Labeled sentiment - target variable
        "overall",       #Raw star rating for reference
        "verified",      #Trust indicator - verified = more genuine sentiment
        "vote",          #Community signal - higher voted = stronger signal
        "style_Size",    #Fit/size prevalent in negative reviews
        "style_Color",   #Sentiment can vary by product variant
    ]

    existing_columns = [col for col in SELECTED_COLUMNS if col in df.columns]
    df = df[existing_columns].copy()
    if reviewer_ids is not None:
        df["reviewerID"] = reviewer_ids

    #Flagging outliers
    #Calculating lengths of new combined text
    df["textLen"] = df["combined_text"].str.len()
    df["wordCount"] = df["combined_text"].str.split().str.len()

    #Dropping empty reviews
    before = len(df)
    df = df[df["textLen"] > 0].reset_index(drop=True)
    dropped = before - len(df)

    if dropped > 0:
        print(f"Dropped {dropped} empty reviews")

    #Identifying outliers using IQR method on word count
    wc = df["wordCount"]
    Q1 = wc.quantile(0.25)
    Q3 = wc.quantile(0.75)
    IQR = Q3 - Q1
    df["is_outlier"] = (wc < (Q1 - 1.5 * IQR)) | (wc > (Q3 + 1.5 * IQR))

    print("\nApplying lexicon-specific preprocessing...")
    before_dedup = len(df)
    if "reviewerID" in df.columns:
        df = df.drop_duplicates(subset=["reviewerID", "combined_text"]).reset_index(drop=True)
        df = df.drop(columns=["reviewerID"])
    else:
        df = df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    dropped_dupes = before_dedup - len(df)
    if dropped_dupes > 0:
        print(f"Dropped {dropped_dupes} duplicate reviews")

    df["clean_vader"] = df["combined_text"].apply(preprocess_for_vader)
    df["clean_swn"] = df["combined_text"].apply(preprocess_for_swn)

    print(f"\nPreprocessing complete. {len(df)} rows remaining.")
    print(SEPARATOR)

    return df

#================================================================
#Reporting & Visualization (for standalone use)
#================================================================
def generate_report(df: pd.DataFrame) -> None:
    """
    Generates summary report.
    """

    print(f"\n{SEPARATOR}")
    print("Generating report...")
    print(SEPARATOR)

    #Labeling report
    print("\nSentiment Distribution:")
    sentiment_counts = df["sentiment"].value_counts()
    sent_percentages = (df["sentiment"].value_counts(normalize=True) * 100).round(2)
    for sentiment in ["Positive", "Neutral", "Negative"]:
        count = sentiment_counts.get(sentiment, 0)
        percent = sent_percentages.get(sentiment, 0.0)
        bar = "â–ˆ" * int(percent / 2)
        print(f"   {sentiment:<10}: {count:>5,} ({percent:5.1f}%)  {bar}")


    #Column selection rationale
    print(f"\n{SEPARATOR}")
    print("Column Selection Rationale:")

    rationale = {
        "combined_text" : "Primary model input  summary + reviewText merged into one field.",
        "sentiment"     : "Model target derived from star rating.",
        "overall"       : "Raw star rating kept for reference and validation.",
        "verified"      : "Trust indicator verified purchases likely have more genuine sentiment.",
        "vote"          : "Community signal higher voted reviews carry stronger sentiment.",
        "style_Size"    : "Fit/size complaints are a primary driver of negative fashion reviews.",
        "style_Color"   : "Sentiment can differ across product colour variants.",
    }

    for col in df.columns:
        reason = rationale.get(col, "Derived column")
        print(f"\n    {col}")
        print(f"      {reason}")

    print(f"\nWorking DataFrame shape: {df.shape}")

    #Outlier summary
    print(f"\n{SEPARATOR}")
    print("Outlier Summary:")

    for col_label, col in [("Combined Text Length (chars)", df["textLen"]),
                        ("Word Count",                   df["wordCount"])]:
        mu  = col.mean()
        sig = col.std()
        q1  = col.quantile(0.25)
        q3  = col.quantile(0.75)
        iqr = q3 - q1

        z_thresh   = mu + 3 * sig
        iqr_upper  = q3 + 1.5 * iqr
        iqr_lower  = q1 - 1.5 * iqr
        z_out      = (col > z_thresh).sum()
        iqr_out    = ((col > iqr_upper) | (col < iqr_lower)).sum()

        print(f"\n  {col_label}:")
        print(f"    Mean              : {mu:.1f}")
        print(f"    Std Dev           : {sig:.1f}")
        print(f"    Z-score threshold : {z_thresh:.1f}    {z_out:,} outliers  ({z_out/len(col)*100:.2f}%)")
        print(f"    IQR upper bound   : {iqr_upper:.1f}    {iqr_out:,} outliers  ({iqr_out/len(col)*100:.2f}%)")

    print(f"\n  Total flagged outliers (IQR, word count): {df['is_outlier'].sum():,}")
    print(f"\n  Outlier breakdown by sentiment:")
    print(df.groupby("sentiment")["is_outlier"].sum().to_string())

    print(f"\n  Top 5 longest reviews:")
    top5 = df.nlargest(5, "wordCount")[["sentiment", "wordCount", "combined_text"]]
    for _, row in top5.iterrows():
        snippet = row["combined_text"][:80].replace("\n", " ")
        print(f"    [{row['sentiment']:<10}]  {row['wordCount']:>4} words  |  {snippet}â€¦")

    #Final Summary
    print(f"\n{SEPARATOR}")
    print("Final Preprocessed DataFrame:")
    print(f"  Shape          : {df.shape}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Outliers flagged (not dropped): {df['is_outlier'].sum():,}")
    print(f"\n  Sentiment counts:")
    print(df["sentiment"].value_counts().to_string())


def generate_figures(df: pd.DataFrame) -> None:
    """
    Generates and saves all preprocessing visualizations.
    """

    print(f"\n{SEPARATOR}")
    print("Generating figures...")
    print(SEPARATOR)

    sentiment_order = ["Positive", "Neutral", "Negative"]
    colors = [SENTIMENT_COLORS[sent] for sent in sentiment_order]

    #Sentiment Distribution (bar + pie)
    counts = df["sentiment"].value_counts().reindex(sentiment_order)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.7)
    axes[0].bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
    axes[0].set_title("Review Count by Sentiment", fontsize=11)
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Reviews")

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[1].set_title("Sentiment Proportion", fontsize=11)

    plt.suptitle("Sentiment Label Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "2.01_sentiment_distribution.png"), dpi=150)
    plt.close()
    print("  2.01_sentiment_distribution.png")

    #Word count boxplot by sentiment (raw and clipped)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    grouped = [df.loc[df["sentiment"] == s, "wordCount"].values
               for s in sentiment_order]

    for ax, clipped, title in zip(
        axes,
        [False, True],
        ["With Outliers", "Clipped at 95th Percentile"]
    ):
        cap  = df["wordCount"].quantile(0.95) if clipped else None
        data = [g.clip(max=cap) if clipped else g for g in grouped]
        bp   = ax.boxplot(data, labels=sentiment_order, patch_artist=True,
                          medianprops=dict(color="black", linewidth=1.5),
                          flierprops=dict(marker="o", markerfacecolor="red",
                                         markersize=4, alpha=0.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Word Count")

    plt.suptitle("Outlier Detection â€” Word Count by Sentiment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "2.02_word_count_by_sentiment.png"), dpi=150)
    plt.close()
    print("  2.02_word_count_by_sentiment.png")

    #Word count hist with outlier threshold
    fig, ax = plt.subplots(figsize=(9, 4))
    wc  = df["wordCount"]
    q1  = wc.quantile(0.25)
    q3  = wc.quantile(0.75)
    iqr = q3 - q1

    ax.hist(wc.clip(upper=wc.quantile(0.99)), bins=60,
            color="#8172B2", edgecolor="white", linewidth=0.5)
    ax.axvline(q3 + 1.5 * iqr, color="red", linestyle="--", linewidth=1.4,
               label=f"IQR upper bound ({q3 + 1.5*iqr:.0f} words)")
    ax.axvline(wc.mean() + 3 * wc.std(), color="orange", linestyle="--", linewidth=1.4,
               label=f"Z-score threshold ({wc.mean() + 3*wc.std():.0f} words)")
    ax.set_title("Word Count Distribution with Outlier Thresholds",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Reviews")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "2.03_word_count_thresholds.png"), dpi=150)
    plt.close()
    print("  2.03_word_count_thresholds.png")

    print(f"\nAll figures saved to {os.path.abspath(FIGURES_FOLDER)}")

#================================================================
#Sampling
#================================================================
def sample_data(df: pd.DataFrame, n: int = 1000, random_seed: int = 1) -> pd.DataFrame:
    """
    Randomly selects a subset of reviews (default 1000) from dataset
    """
    n_samples = min(n, len(df))

    if n_samples < n:
        print(f"NOTE: Dataset has only {n_samples} rows. Returning all rows.")
    else:
        print(f"Sampling {n_samples} random reviews.")
    
    sampled_df = df.sample(n=n_samples, random_state=1).reset_index(drop=True)

    print(f"Sampled DF shape: {sampled_df.shape}")

    return sampled_df
    

#================================================================
#Entry for standalone execution
#================================================================
if __name__ == "__main__":
    df = load_data()
    preprocessed_df = preprocess_data(df)
    generate_report(preprocessed_df)
    generate_figures(preprocessed_df)
    print(f"\n{SEPARATOR}")
    print("Preprocessing complete. Report generated and figures saved.")
    print(SEPARATOR)
