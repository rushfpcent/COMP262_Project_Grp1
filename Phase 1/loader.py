"""
Script to load and clean the phase 1 data.
Returns DataFrame
Usage:
    from loader import load_data
    df = load_data()
"""

#Importing libraries
import json
import pandas as pd
import numpy as np
import os

#Defining data path
DATA_PATH = os.path.join(os.path.dirname(__file__), "Data", "AMAZON_FASHION_5.json")

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loading data into a cleaned DataFrame.
    """

    print(f"Loading data from: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    #Creating DataFrame from records list
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")

    #Cleaning data
    df = _clean(df)
    print(f"  Cleaning complete. Final shape: {df.shape}\n")

    return df

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applying type conversions and adding derived columns.
    """
    #Converting ratings to numeric
    df["overall"] = pd.to_numeric(df.get("overall"), errors="coerce")

    #Converting time / date to datetime
    if "unixReviewTime" in df.columns:
        df["reviewDate"] = pd.to_datetime(
            df["unixReviewTime"], unit="s", errors="coerce"
        )

    if "reviewTime" in df.columns:
        df["reviewTime"] = pd.to_datetime(
            df["reviewTime"], format="%m %d, %Y", errors="coerce"
        )

    #Calculating review text lengths
    if "reviewText" in df.columns:
        df["reviewLen"] = df["reviewText"].fillna("").str.len()
        df["wordCount"] = df["reviewText"].fillna("").str.split().str.len()

    #Summary lengths
    if "summary" in df.columns:
        df["summaryLen"] = df["summary"].fillna("").str.len()

    #Vote column to numeric
    if "vote" in df.columns:
        df["vote"] = pd.to_numeric(df["vote"].astype(str).str.replace(",", ""), errors="coerce")

    #Unpacking style into flat columns
    if "style" in df.columns:
        style_df = df["style"].apply(
            lambda x: {k.strip().rstrip(":"): v.strip()
                       for k, v in x.items()}
            if isinstance(x, dict) else {}
        )
        expanded_style = pd.json_normalize(style_df)
        expanded_style.index = df.index

        #Prefixing style columns to avoid name clashes
        expanded_style.columns = [
            f"style_{col}" for col in expanded_style.columns
        ]
        df = pd.concat([df, expanded_style], axis=1)

    return df


##Test block
"""if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.dtypes)
    print(df.columns.tolist())"""