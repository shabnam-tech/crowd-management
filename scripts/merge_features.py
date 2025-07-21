import os
import re
import numpy as np
import pandas as pd
from glob import glob

def assign_density_label(count):
    if count <= 25:
        return "Low"
    elif count <= 50:
        return "Medium"
    elif count <= 75:
        return "High"
    else:
        return "Critical"

def merge_features(
    csv_dir="data/labels_filtered",
    visual_feature_path="data/grouped_video_features.npy",
    output_path="data/final_dataset.pkl"
):
    # Load visual features
    print("📥 Loading visual features...")
    visual_data = np.load(visual_feature_path, allow_pickle=True).item()
    visual_rows = []

    for vid, features in visual_data.items():
        for frame_no, feat in enumerate(features):
            visual_rows.append({
                "Video_ID": vid,
                "Frame_No": frame_no,
                "Visual_Features": feat
            })

    visual_df = pd.DataFrame(visual_rows)
    print(f"✅ Visual features loaded: {visual_df.shape}")

    # Load and process raw CSV label files
    print("📥 Loading and processing raw CSVs...")
    csv_files = glob(os.path.join(csv_dir, "filtered_HajjDataset_V*_Train.csv"))
    csv_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)

        # Extract video ID
        match = re.search(r"V(\d+)", os.path.basename(file))
        if not match:
            print(f"⚠️ Skipping {file} — couldn't extract Video ID")
            continue

        video_id = int(match.group(1))
        df["Video_ID"] = video_id

        # Compute bbox size
        df["bbox_size"] = df["Width"] * df["Height"]

        # Filter out unwanted classes (optional)
        valid_classes = ["sitting", "standing", "sleeping", "moving_in_opposite"]
        df = df[df["Classes"].isin(valid_classes)].copy()

        if df.empty:
            print(f"⚠️ Skipping {file} — no valid class rows after filtering")
            continue

        # Compute per-frame people count and bbox size
        people_count = df.groupby("Frame_No").size().reset_index(name="people_count")
        avg_bbox = df.groupby("Frame_No")["bbox_size"].mean().reset_index(name="avg_bbox_size")

        frame_stats = pd.merge(people_count, avg_bbox, on="Frame_No")
        frame_stats["Video_ID"] = video_id
        frame_stats["density_label"] = frame_stats["people_count"].apply(assign_density_label)

        csv_dfs.append(frame_stats)

    if not csv_dfs:
        print("❌ No valid CSVs processed.")
        return

    csv_df = pd.concat(csv_dfs, ignore_index=True)
    print(f"✅ CSV features ready: {csv_df.shape}")

    # Convert types
    csv_df["Video_ID"] = csv_df["Video_ID"].astype(int)
    csv_df["Frame_No"] = csv_df["Frame_No"].astype(int)
    visual_df["Video_ID"] = visual_df["Video_ID"].astype(int)
    visual_df["Frame_No"] = visual_df["Frame_No"].astype(int)

    # Merge
    print("🔗 Merging visual + CSV features...")
    final_df = pd.merge(csv_df, visual_df, on=["Video_ID", "Frame_No"], how="inner")
    print(f"🎯 Final merged dataset shape: {final_df.shape}")

    # Save
    final_df.to_pickle(output_path)
    print(f"✅ Final dataset saved at → {output_path}")

if __name__ == "__main__":
    merge_features()
