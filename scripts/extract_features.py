import os
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.models import Model

def extract_features(frame_dir="data/frames", output_path="data/grouped_video_features.npy", batch_size=32):
    # Load pretrained models (without classification head)
    print("🔄 Loading pretrained models...")
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    print(f"📦 Found {len(frame_files)} frames")

    feature_dict = {}

    for i in range(0, len(frame_files), batch_size):
        batch = frame_files[i:i+batch_size]
        images = []
        video_ids = []
        frame_ids = []

        for fname in batch:
            path = os.path.join(frame_dir, fname)
            image = cv2.imread(path)
            if image is None:
                print(f"❌ Skipped unreadable: {fname}")
                continue

            img_resized = cv2.resize(image, (224, 224))
            images.append(img_resized)
            video_id = int(fname.split("_")[0])
            frame_no = int(fname.split("_")[2].split(".")[0])
            video_ids.append(video_id)
            frame_ids.append(frame_no)

        if not images:
            continue

        images_np = np.array(images)
        # Extract features
        resnet_feat = resnet.predict(resnet_preprocess(images_np), verbose=0)
        mobile_feat = mobilenet.predict(mobilenet_preprocess(images_np), verbose=0)
        combined = np.concatenate((resnet_feat, mobile_feat), axis=1)

        # Group features
        for idx in range(len(combined)):
            vid = video_ids[idx]
            fno = frame_ids[idx]
            if vid not in feature_dict:
                feature_dict[vid] = {}
            feature_dict[vid][fno] = combined[idx]

        print(f"✅ Processed batch {i//batch_size + 1}/{(len(frame_files) + batch_size - 1)//batch_size}")

    # Convert inner dicts to sorted lists
    for vid in feature_dict:
        ordered = [feature_dict[vid][f] for f in sorted(feature_dict[vid].keys())]
        feature_dict[vid] = np.stack(ordered)

    # Save grouped feature dictionary
    np.save(output_path, feature_dict)
    print(f"📦 Features saved → {output_path}")
    print(f"🎥 Total videos processed: {len(feature_dict)}")

if __name__ == "__main__":
    extract_features()
