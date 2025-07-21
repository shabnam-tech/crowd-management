import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# 🧠 Custom Attention Layer for loading the model
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.attention_dense = tf.keras.layers.Dense(feature_dim, activation="softmax")

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        return inputs * attention_scores

# 🔧 Load the trained .keras model
print("🔧 Loading model...")
model = load_model("models/crowd_lstm_attention_model.keras", custom_objects={"Attention": Attention})
print("✅ Model loaded successfully!")

# 📂 Load visual feature dictionary
features_dict = np.load("data/grouped_video_features.npy", allow_pickle=True).item()
print(f"📦 Loaded visual features for {len(features_dict)} videos")

# 📰 Load Reddit sentiment scores
sentiment_df = pd.read_csv("data/hajj_sentiment_results.csv")
avg_sentiment_score = sentiment_df["sentiment_score"].mean()
print(f"🧠 Average Sentiment Score from Reddit: {avg_sentiment_score:.2f}")

# 🔁 Risk adjustment rule
def adjust_risk(original_class, sentiment_score):
    if sentiment_score <= 2:
        return min(original_class + 1, 3)
    elif sentiment_score >= 4:
        return max(original_class - 1, 0)
    return original_class

# 🎯 Risk labels
risk_labels = ["Low", "Medium", "High", "Critical"]

# 🔍 Run inference and adjust risk
results = []
print("🔄 Running model inference on all video frames...")
for vid, feature_seq in tqdm(features_dict.items(), desc="🎥 Videos"):
    for frame_no, feature_vec in enumerate(feature_seq):
        x_input = np.expand_dims(np.expand_dims(feature_vec, axis=0), axis=0)  # (1, 1, 3328)
        preds = model.predict(x_input, verbose=0)
        pred_class = np.argmax(preds)
        adjusted = adjust_risk(pred_class, avg_sentiment_score)

        results.append({
            "Video_ID": vid,
            "Frame_No": frame_no,
            "Predicted_Risk": risk_labels[pred_class],
            "Adjusted_Risk": risk_labels[adjusted],
            "Sentiment_Score": avg_sentiment_score
        })

# 💾 Save results to CSV
output_path = "data/combined_risk_results.csv"
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"✅ Combined results saved → {output_path}")
