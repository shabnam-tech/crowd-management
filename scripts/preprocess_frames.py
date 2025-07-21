import os
import cv2
import numpy as np

def preprocess_frames(input_dir="data/frames", output_dir="data/processed_frames", size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(input_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(input_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Skipped unreadable: {img_file}")
                continue
            resized = cv2.resize(image, size)
            normalized = resized / 255.0  # float32 [0,1]
            save_path = os.path.join(output_dir, img_file.replace(".jpg", ".npy"))
            np.save(save_path, normalized)

    print("✅ All frames preprocessed.")

if __name__ == "__main__":
    preprocess_frames()
