from loader import load_data
from basic_preprocess import preprocess_data, sample_data
from Models.swn_model import run_swn_model

#loading the raw data
raw_df = load_data()

#preprocessing the data
preprocessed_df = preprocess_data(raw_df)

#creating a 1000-row sample (or however many is possible)
sampled_df = sample_data(preprocessed_df, n=1000)

#running SWN model
scored_df = run_swn_model(sampled_df)

#output check
print(scored_df[['clean_swn', 'sentiment', 'swn_prediction', 'swn_score']].head())