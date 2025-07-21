import os
import pandas as pd

def filter_labels(label_dir, output_dir, frame_interval=5):
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.csv'):
            path = os.path.join(label_dir, label_file)
            df = pd.read_csv(path)

            # Filter only rows where Frame_No is a multiple of frame_interval
            filtered_df = df[df["Frame_No"] % frame_interval == 0]

            out_path = os.path.join(output_dir, f"filtered_{label_file}")
            filtered_df.to_csv(out_path, index=False)

            print(f"✅ Filtered {label_file} → {out_path}")

if __name__ == "__main__":
    filter_labels("videos/Training/Labels", "data/labels_filtered", frame_interval=5)
