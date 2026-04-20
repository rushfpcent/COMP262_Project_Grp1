import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import your Phase 2 prep function
from phase2_prep import prepare_phase2_data

SEPARATOR = "=" * 64

def run_llm_tasks(df):
    print(f"\n{SEPARATOR}")
    print("PHASE 2: LLM TASKS (Summarization & Customer Support)")
    print(SEPARATOR)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running models locally on: {device.upper()}")

    # ================================================================
    # TASK 16: Summarize 10 long reviews 
    # ================================================================
    print("\n" + "-"*40)
    print("TASK 16: LLM SUMMARIZATION")
    print("-" * 40)
    
    long_reviews = df[df['wordCount'] > 100].head(10)
    
    if len(long_reviews) < 10:
        print("Note: Could not find 10 reviews over 100 words. Using the longest available.")
        long_reviews = df.nlargest(10, 'wordCount')

    print("\nLoading Hugging Face Summarization Model (facebook/bart-large-cnn)...")
    sum_model_name = "facebook/bart-large-cnn"
    sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name).to(device)

    count = 1
    for idx, row in long_reviews.iterrows():
        text = row['combined_text']
        
        # Tokenize, Generate, Decode
        inputs = sum_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
        summary_ids = sum_model.generate(inputs["input_ids"], max_new_tokens=80, min_new_tokens=20, num_beams=6, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
        summary_text = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        if count <= 2: 
            print(f"\nReview #{count} (Original length: {row['wordCount']} words)")
            print(f"ORIGINAL SNIPPET: {text[:150]}...")
            print(f"LLM 50-WORD SUMMARY:  {summary_text}")
        count += 1
    
    # ================================================================
    # TASK 17: Customer Service Response
    # ================================================================
    print("\n" + "-"*40)
    print("TASK 17: CUSTOMER SERVICE RESPONSE")
    print("-" * 40)
    
    question_reviews = df[df['combined_text'].fillna("").str.contains(r'\?', regex=True)]
    
    if not question_reviews.empty:
        target_review = question_reviews.iloc[0]['combined_text']
    else:
        print("No question mark found in text. Using a default negative review.")
        target_review = df[df['sentiment'] == 'Negative'].iloc[0]['combined_text']

    print(f"CUSTOMER REVIEW:\n'{target_review}'\n")

    print("Loading Hugging Face Text Generation Model (google/flan-t5-base)...")
    gen_model_name = "google/flan-t5-base"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(device)

    prompt = (
        f"You are a customer service representative. "
        f"Write a polite and professional response to the following customer review. "
        f"Acknowledge their experience and offer help if needed.\n\n"
        f"Customer review: {target_review[:300]}\n\n"  
        f"Customer service response:"
    )

    print("Generating AI Response...")
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    # Generate the response
    response_ids = gen_model.generate(inputs["input_ids"], max_new_tokens=150, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
    response_text = gen_tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    print(f"\nAI CUSTOMER SERVICE REPLY:\n{response_text}")
    print(f"\n{SEPARATOR}")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    FULL_DATA_PATH = os.path.join(current_dir, "Data", "AMAZON_FASHION.json")
    
    # Load our Phase 2 dataset
    _, _, _, _, df_sample, _ = prepare_phase2_data(FULL_DATA_PATH)
    
    # Execute the LLM tasks
    run_llm_tasks(df_sample)
