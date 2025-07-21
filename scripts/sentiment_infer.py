import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.nn.functional import softmax

def predict_sentiment(text, model, tokenizer, device):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    scores = softmax(output.logits, dim=1).cpu().numpy()[0]
    return scores.argmax() + 1  # Sentiment: 1 (negative) to 5 (positive)

def run_sentiment_inference(input_file="data/cleaned_hajj_reddit_data.csv", output_file="data/hajj_sentiment_results.csv"):
    df = pd.read_csv(input_file)

    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("🔍 Running sentiment analysis...")
    df["sentiment_score"] = df["text"].apply(lambda x: predict_sentiment(str(x), model, tokenizer, device))
    df.to_csv(output_file, index=False)
    print(f"✅ Sentiment results saved to {output_file}")

if __name__ == "__main__":
    run_sentiment_inference()

S