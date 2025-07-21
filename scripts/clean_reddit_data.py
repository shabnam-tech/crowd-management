import pandas as pd
import re
import os

def remove_quran_verses(text):
    # Remove posts with mostly Quran-style Arabic numbering or short meaningless text
    if not isinstance(text, str):
        return False
    if len(text.strip()) < 30:
        return False
    if re.search(r'\[\d{1,3}\]', text):
        return False
    return True

def clean_reddit_data(input_file="data/reddit_hajj_data.csv", output_file="data/cleaned_hajj_reddit_data.csv"):
    print("🧪 Running clean_reddit_data function...")

    df = pd.read_csv(input_file)

    # Drop duplicates
    df.drop_duplicates(subset=["title", "text"], inplace=True)

    # Remove posts with empty text or meaningless titles
    df = df[df["text"].notna() & df["title"].notna()]
    df = df[df["text"].str.strip().str.len() > 30]

    # Remove posts that look like Quran verses or empty quotes
    df = df[df["text"].apply(remove_quran_verses)]

    # Reset index and save
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_reddit_data()

