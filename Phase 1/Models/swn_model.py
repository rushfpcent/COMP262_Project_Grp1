import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

#documentation
#https://www.nltk.org/api/nltk.corpus.reader.sentiwordnet.html

nltk.download(['sentiwordnet', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'], quiet=True)

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Maps standard POS tag to WordNet POS Tag
    """
    if treebank_tag.startswith("J"):
        return wn.ADJ
    elif treebank_tag.startswith("V"):
        return wn.VERB
    elif treebank_tag.startswith("N"):
        return wn.NOUN
    elif treebank_tag.startswith("R"):
        return wn.ADV
    else:
        return None

def swn_polarity(tokens: list) -> float:
    """
    Calculates the SWN polarity score for a list of tokens
    """
    if not tokens:
        return 0.0
    
    sentiment_score_swn = 0.0
    tags = nltk.pos_tag(tokens)

    for word, tag in tags:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag not in (wn.ADJ, wn.VERB, wn.NOUN, wn.ADV):
            continue

        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            continue

        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        sentiment_score_swn += (swn_synset.pos_score() - swn_synset.neg_score())

    return sentiment_score_swn

def predict_swn_sentiment(score: float) -> str:
    """
    Classifies numerical sentiment score to a categorical one
    """
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"
    
def run_swn_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies SWN scoring to a dataframe
    """
    print("\nRunning SentiWordNet model.")
    df = df.copy()
    df['swn_score'] = df['clean_swn'].apply(swn_polarity)
    df['swn_prediction'] = df["swn_score"].apply(predict_swn_sentiment)
    print("\nSentiWordNet scoring finished.")
    return df


