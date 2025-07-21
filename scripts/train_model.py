import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Lambda
import tensorflow as tf

# ✅ Custom Attention Layer (with support for model save/load)
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        self.attention_dense = Dense(self.feature_dim, activation="softmax")

    def call(self, inputs):
        attention_scores = self.attention_dense(inputs)
        return inputs * attention_scores

# ✅ Load dataset
df = pd.read_pickle("data/final_dataset.pkl")

# 🔄 Prepare features
X = np.stack(df["Visual_Features"].values)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # Shape: (samples, timesteps=1, features)

# 🔄 Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(df["density_label"])
y = to_categorical(y_encoded)

# 🔀 Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build the model
inputs = Input(shape=(1, X.shape[2]))
x = Attention()(inputs)
x = LSTM(64)(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs, output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 🏋️ Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 💾 Save model in modern format
os.makedirs("models", exist_ok=True)
model.save("models/crowd_lstm_attention_model.keras")
print("✅ Model saved to models/crowd_lstm_attention_model.keras")
