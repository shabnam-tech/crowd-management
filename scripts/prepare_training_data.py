import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def load_data_safe(labels_dir, frames_dir, output_dir, batch_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    X_batch, y_batch = [], []
    batch_num = 0
    label_encoder = LabelEncoder()
    all_labels = []

    print("🔎 Scanning for all labels...")
    # Collect all labels for consistent encoding
    for file in os.listdir(labels_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(labels_dir, file))
            all_labels.extend(df["Classes"].astype(str).str.strip().tolist())

    label_encoder.fit(all_labels)

    for file in os.listdir(labels_dir):
        if not file.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(labels_dir, file))
        print(f"\n📄 Processing: {file} | {len(df)} rows")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            frame_file = f"{row['Video_No']}_frame_{row['Frame_No']}.npy"
            path = os.path.join(frames_dir, frame_file)

            if os.path.exists(path) and os.path.getsize(path) > 1000:
                try:
                    frame = np.load(path)
                    label = str(row["Classes"]).strip()
                    X_batch.append(frame)
                    y_batch.append(label_encoder.transform([label])[0])
                except Exception as e:
                    print(f"❌ Failed: {frame_file} — {e}")
            else:
                print(f"⚠️ Skipped: {frame_file}")

            # Save in batches
            if len(X_batch) >= batch_size:
                save_path = os.path.join(output_dir, f"chunk_{batch_num}.npz")
                np.savez_compressed(save_path, X=np.array(X_batch), y=np.array(y_batch))
                print(f"💾 Saved batch → {save_path} [{len(X_batch)} samples]")
                X_batch, y_batch = [], []
                batch_num += 1

    # Save any leftover samples
    if X_batch:
        save_path = os.path.join(output_dir, f"chunk_{batch_num}.npz")
        np.savez_compressed(save_path, X=np.array(X_batch), y=np.array(y_batch))
        print(f"💾 Saved final batch → {save_path} [{len(X_batch)} samples]")

    print("\n✅ All batches saved in:", output_dir)
    print("🏷️ Classes:", list(label_encoder.classes_))

if __name__ == "__main__":
    load_data_safe(
        labels_dir="data/labels_filtered",
        frames_dir="data/processed_frames",
        output_dir="data/dataset_chunks",
        batch_size=1000  # You can reduce this to 500 if still laggy
    )
