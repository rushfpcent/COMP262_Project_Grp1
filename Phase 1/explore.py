"""
This module contains code for exploartion of the data.
Usage: 
    cd "COMP262_PROJECT_GRP1/Phase1
    python explore.py
"""

#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

#Importing loader function
from loader import load_data

#Creating folder to hold figures
FIGURES_FOLDER = os.path.join(os.path.dirname(__file__), "figures")
if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER)

#Defining reusable constants
PALETTE = "viridis"
SEPARATOR = "-" * 64

#Setting plt style
plt.style.use("seaborn-v0_8-whitegrid")


#Loading data
df = load_data()

#================================================================
#Conducting basic exploration
#================================================================

print(SEPARATOR)
print("Basic Information about the dataset:")
print(SEPARATOR)

print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())
print(f"\nShape: {df.shape}")

#================================================================
#Counts and averages
#================================================================

print(f"\n{SEPARATOR}")
print("Counts and Averages:")
print(SEPARATOR)

#Initializing variables to hold counts, averages, etc
total_reviews = len(df)
unique_products = df["asin"].nunique()
unique_users = df["reviewerID"].nunique()
average_rating = df["overall"].mean()
median_rating = df["overall"].median()
mode_rating = df["overall"].mode()[0]
std_rating = df["overall"].std()

#Displaying information
print(f"   Total Reviews: {total_reviews}")
print(f"   Unique Products: {unique_products}")
print(f"   Unique Users: {unique_users}")
print(f"   Average Rating: {average_rating}")
print(f"   Median Rating: {median_rating}")
print(f"   Mode Rating: {mode_rating}")
print(f"   Std Dev of Ratings: {std_rating}")

#Percentage of verified purchases
print(f"\n   Percentage of Verified Purchases: {(df['verified'].mean() * 100)}%")

#Rating distribution
print("\n   Rating Distribution:")
rating_dist = (df["overall"].value_counts(normalize=True).sort_index() * 100).round(2)
for star, pct in rating_dist.items():
    bar = "█" * int(pct / 2)
    print(f"   {int(star)}★ {pct:5.1f}% {bar}")


#================================================================
#Missing Value Analysis
#================================================================
print(f"\n{SEPARATOR}")
print("Missing Value Analysis:")
print(SEPARATOR)

missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage": missing_percent
})
missing_df = missing_df[missing_df["Missing Values"] > 0]
print("   No missing values found." if missing_df.empty else missing_df.to_string())


#================================================================
#Checking for duplicates
#================================================================
print(f"\n{SEPARATOR}")
print("Duplicate Analysis:")
print(SEPARATOR)

#Defining columns to check to avoid dicts and lists
cols_to_check = [
    col for col in df.columns
    if not df[col].apply(lambda x: isinstance(x, (dict, list))).any()
]

duplicate_count = df.duplicated(subset=cols_to_check).sum()
print(f"   Total Duplicate Rows: {duplicate_count}")

#Checking same reviewer reviewing same product multiple times
duplicate_reviews = df.duplicated(subset=["reviewerID", "asin"], keep=False).sum()
print(f"   Duplicate Reviews (same reviewer reviewing same product): {duplicate_reviews}")


#================================================================
#Reviews per product and per user
#================================================================
print(f"\n{SEPARATOR}")
print("Reviews per Product and per User:")
print(SEPARATOR)

reviews_per_product = df["asin"].value_counts()
reviews_per_user = df["reviewerID"].value_counts()

#Average number of reviews
print(f"   Average Reviews per Product: {reviews_per_product.mean():.2f}")
print(f"   Average Reviews per User: {reviews_per_user.mean():.2f}")

#Most reviewed products
print("\n   Top 5 Most Reviewed Products:")
print(reviews_per_product.head().to_string())

#Most active users
print("\n   Top 5 Most Active Users:")
print(reviews_per_user.head().to_string())

#Number of users who reviewed only one product
single_review_users = (reviews_per_user == 1).sum()
print(f"\n   Number of Users Who Reviewed Only One Product: {single_review_users}")


#================================================================
#Length of reviews analysis
#================================================================
print(f"\n{SEPARATOR}")
print("Length of Reviews Analysis:")
print(SEPARATOR)

#Defining length, mean and std for review text
review_length = df["reviewLen"]
mean_length = review_length.mean()
std_length = review_length.std()

#Defining threshold for long reviews (mean + 3*std)
threshold = mean_length + 3 * std_length
long_reviews = (review_length > threshold).sum()

#Displaying information about review lengths
print(review_length.describe().round(1).to_string())
print(f"\n   1st percentile: {review_length.quantile(0.01):.1f}")
print(f"   99th percentile: {review_length.quantile(0.99):.1f}")
print(f"\n   Threshold for Long Reviews (mean + 3*std): {threshold:.1f}")
print(f"   Number of Long Reviews: {long_reviews}")

#Word count analysis
print("\n   Word Count Analysis:")
print(df["wordCount"].describe().round(1).to_string())

#Mean review length by rating
print("\n   Mean Review Length by Rating:")
mean_length_by_rating = df.groupby("overall")["reviewLen"].mean().round(1)
print(mean_length_by_rating.to_string())

#Summary length analysis
print("\n   Summary Length Analysis:")
print(df["summaryLen"].describe().round(1).to_string())

#Longest review
longest_review = df.loc[review_length.idxmax()]
print(f"\n   Snippet of Longest Review (Length: {longest_review['reviewLen']} characters):")
print(longest_review["reviewText"][:100] + "...")


#================================================================
#Temporal analysis of reviews
#================================================================
print(f"\n{SEPARATOR}")
print("Temporal Analysis of Reviews:")
print(SEPARATOR)

#Calculating review window
review_window = (df["reviewDate"].max() - df["reviewDate"].min()).days
print(f"   Review Window: {review_window} days | (~{review_window / 365:.1f} years)")

#Reviews per year
reviews_per_year = df["reviewDate"].dt.year.value_counts().sort_index().to_string()
print("\n   Reviews per Year:")
print(reviews_per_year)


#================================================================
#Vote count analysis
#================================================================
print(f"\n{SEPARATOR}")
print("Vote Count Analysis:")
print(SEPARATOR)

#Filtering columns that have votes
has_votes = df["vote"].notna().sum()
print(f"   Reviews with Votes: {has_votes} ({(has_votes / len(df) * 100):.2f}%)")
print(f"   Reviews without Votes: {len(df) - has_votes} ({((len(df) - has_votes) / len(df) * 100):.2f}%)")

print("\n   Vote Count Statistics:")
print(df["vote"].describe().round(1).to_string())

#Average votes by rating
print("\n   Average Votes by Rating:")
avg_votes_by_rating = df.groupby("overall")["vote"].mean().round(1)
print(avg_votes_by_rating.to_string())

#Most voted reviews
top_voted = df.nlargest(5, "vote")[["reviewerName", "asin", "overall", "vote", "reviewText"]]
top_voted["reviewText"] = top_voted["reviewText"].str[:100]
print("\n   Top 5 Most Voted Reviews:")
print(top_voted.to_string(index=False))


#================================================================
#Style Analysis
#================================================================
print(f"\n{SEPARATOR}")
print("Style Analysis:")
print(SEPARATOR)

#style_Size
#Number of reviews with style_Size information
has_size = df["style_Size"].notna().sum()
print(f"   Reviews with style_Size: {has_size} ({(has_size / len(df) * 100):.2f}%)")
print(f"   Reviews without style_Size: {len(df) - has_size} ({((len(df) - has_size) / len(df) * 100):.2f}%)")

#Unique style_Size values
unique_sizes = df["style_Size"].nunique()
print(f"\n   Unique style_Size values: {unique_sizes}")

#Most common style_Size values
print("\n   Top 5 Most Common style_Size values:")
print(df["style_Size"].value_counts().head().to_string())

#style_Color
#Number of reviews with style_Color information
has_color = df["style_Color"].notna().sum()
print(f"\n   Reviews with style_Color: {has_color} ({(has_color / len(df) * 100):.2f}%)")
print(f"   Reviews without style_Color: {len(df) - has_color} ({((len(df) - has_color) / len(df) * 100):.2f}%)")

#Unique style_Color values
unique_colors = df["style_Color"].nunique()
print(f"\n   Unique style_Color values: {unique_colors}")

#Most common style_Color values
print("\n   Top 5 Most Common style_Color values:")
print(df["style_Color"].value_counts().head().to_string())

#Mean rating by style_Size and style_Color
print("\n   Mean Rating by style_Size:")
mean_rating_by_size = df.groupby("style_Size")["overall"].mean().round(2).sort_values(ascending=False)
print(mean_rating_by_size.head().to_string())

print("\n   Mean Rating by style_Color:")
mean_rating_by_color = df.groupby("style_Color")["overall"].mean().round(2).sort_values(ascending=False)
print(mean_rating_by_color.head().to_string())


#================================================================
#Visualizations
#================================================================
print(f"\n{SEPARATOR}")
print("Generating Visualizations...")
print(SEPARATOR)

#Rating distribution plot
fig, ax = plt.subplots(figsize=(8, 5))
rc = df["overall"].value_counts().sort_index()
bars = ax.bar(rc.index.astype(int).astype(str), rc.values,
              color=sns.color_palette(PALETTE, n_colors=len(rc)),
              edgecolor="black", linewidth=0.7)
ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Number of Reviews", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "01_rating_distribution.png"), dpi=150)
plt.close()
print("   Saved: figures/01_rating_distribution.png")

#Reviews per product
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vals = reviews_per_product.values
for ax, log, color, title in zip(axes, [False, True], ["#4c72b0", "#dd8452"], ["Products clustering at review counts", "Small Number of products dominating review counts"]):
    ax.hist(vals, bins=50, color=color, edgecolor="black", linewidth=0.7, log=log)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Reviews per Product")
    ax.set_ylabel("Number of Products")
plt.suptitle("Distribution of Reviews per Product", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "02_reviews_per_product.png"), dpi=150)
plt.close()
print("   Saved: figures/02_reviews_per_product.png")

#Reviews per user
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
uvals = reviews_per_user.values
for ax, log, color, title in zip(axes, [False, True], ["#c4c653", "#3eb4ac"], ["Linear y-axis", "Log y-axis"]):
    ax.hist(uvals, bins=50, color=color, edgecolor="black", linewidth=0.7, log=log)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Reviews per User")
    ax.set_ylabel("Number of Users")
plt.suptitle("Distribution of Reviews per User", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "03_reviews_per_user.png"), dpi=150)
plt.close()
print("   Saved: figures/03_reviews_per_user.png")

#Review length histogram boxed by rating
cap = df["reviewLen"].quantile(0.99)
fig, ax = plt.subplots(figsize=(10, 6))
axes[0].hist(df["reviewLen"].clip(upper=cap), bins=50, color="#4c72b0", edgecolor="black", linewidth=0.7)
axes[0].axvline(df["reviewLen"].mean(), color="red", linestyle="--", label=f"Mean {df['reviewLen'].mean():.0f}")
axes[0].axvline(df["reviewLen"].median(), color="green", linestyle="--", label=f"Median {df['reviewLen'].median():.0f}")
axes[0].set_title("Distribution of Review Lengths (Clipped at 99th percentile)", fontsize=12)
axes[0].set_xlabel("Review Length (characters)")
axes[0].set_ylabel("Number of Reviews")
axes[0].legend(fontsize=9)

stars = sorted(df["overall"].dropna().unique())
grouped = [df.loc[df["overall"] == s, "reviewLen"].clip(upper=cap) for s in stars]
bp = axes[1].boxplot(grouped, labels=[str(int(s)) for s in stars], patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5))
for patch, col in zip(bp["boxes"], sns.color_palette(PALETTE, len(stars))):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
axes[1].set_title("Review Length by Rating", fontsize=12)
axes[1].set_xlabel("Rating")
axes[1].set_ylabel("Review Length (characters)")

plt.suptitle("Analysis of Review Lengths", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "04_review_length_analysis.png"), dpi=150)
plt.close()
print("   Saved: figures/04_review_length_analysis.png")

#Word count distribution
cap_wc = df["wordCount"].quantile(0.99)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["wordCount"].clip(upper=cap_wc), bins=60, color="#c653b7", edgecolor="black", linewidth=0.7)
ax.axvline(df["wordCount"].mean(), color="red", linestyle="--", label=f"Mean {df['wordCount'].mean():.0f}")
ax.axvline(df["wordCount"].median(), color="green", linestyle="--", label=f"Median {df['wordCount'].median():.0f}")
ax.set_title("Distribution of Word Counts (Clipped at 99th percentile)", fontsize=12)
ax.set_xlabel("Word Count")
ax.set_ylabel("Number of Reviews")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "05_word_count_distribution.png"), dpi=150)
plt.close()
print("   Saved: figures/05_word_count_distribution.png")

#Reviews by time of year
df["month"] = df["reviewDate"].dt.month
reviews_by_month = df["month"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(reviews_by_month.index, reviews_by_month.values, color="#4c72b0", edgecolor="black", linewidth=0.7)
ax.set_title("Number of Reviews by Month", fontsize=12)
ax.set_xlabel("Month")
ax.set_ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "06_reviews_by_month.png"), dpi=150)
plt.close()
print("   Saved: figures/06_reviews_by_month.png")

#Mean rating per year
reviews_by_year = df["reviewDate"].dt.year.value_counts().sort_index()
mean_rating_by_year = df.groupby(df["reviewDate"].dt.year)["overall"].mean()
fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = "#4c72b0"
ax1.bar(reviews_by_year.index, reviews_by_year.values, color=color1, edgecolor="black", linewidth=0.7)
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Reviews", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax2 = ax1.twinx()
color2 = "#dd8452"
ax2.plot(mean_rating_by_year.index, mean_rating_by_year.values, color=color2, marker="o", label="Mean Rating")
ax2.set_ylabel("Mean Rating", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(1, 5)
ax1.set_title("Number of Reviews and Mean Rating by Year", fontsize=12)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "07_reviews_and_rating_by_year.png"), dpi=150)
plt.close()
print("   Saved: figures/07_reviews_and_rating_by_year.png")

#Verified vs Unverified purchases
fig, ax = plt.subplots(figsize=(8, 5))
verified_counts = df["verified"].value_counts()
ax.pie(verified_counts.values, labels=["Verified" if v else "Unverified" for v in verified_counts.index],
       autopct="%1.1f%%", colors=["#4c72b0", "#dd8452"], startangle=140, wedgeprops=dict(edgecolor="black", linewidth=0.7))
ax.set_title("Verified vs Unverified Purchases", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "08_verified_vs_unverified.png"), dpi=150)
plt.close()
print("   Saved: figures/08_verified_vs_unverified.png")

#Top 10 products
top10 = reviews_per_product.head(10)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top10.index[::-1], top10.values[::-1], color=sns.color_palette(PALETTE, len(top10))[::-1], edgecolor="black", linewidth=0.7)
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_title("Top 10 Most Reviewed Products", fontsize=12)
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Product ASIN")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "09_top10_products.png"), dpi=150)
plt.close()
print("   Saved: figures/09_top10_products.png")

#Top sizes
fig, ax = plt.subplots(figsize=(9, 4))
size_counts = df["style_Size"].value_counts().head(12)
bars = ax.bar(size_counts.index, size_counts.values,
                color=sns.color_palette(PALETTE, len(size_counts)), edgecolor="white")
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
ax.set_title("Review Count by Size", fontsize=13, fontweight="bold")
ax.set_xlabel("Size")
ax.set_ylabel("Number of Reviews")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "10_reviews_by_size.png"), dpi=150)
plt.close()
print("   Saved: figures/10_reviews_by_size.png")

#Top colors
fig, ax = plt.subplots(figsize=(11, 4))
color_counts = df["style_Color"].value_counts().head(12)
bars = ax.bar(color_counts.index, color_counts.values,
                color=sns.color_palette(PALETTE, len(color_counts)), edgecolor="white")
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
ax.set_title("Review Count by Color", fontsize=13, fontweight="bold")
ax.set_xlabel("Color")
ax.set_ylabel("Number of Reviews")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, "11_reviews_by_color.png"), dpi=150)
plt.close()
print("   Saved: figures/11_reviews_by_color.png")

print(f"\n{SEPARATOR}")
print("Exploration Completed!")
print(SEPARATOR)