import cv2
import os

def extract_frames(video_dir, output_dir, frame_interval=5):
    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi')):
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()
            frame_no = 0

            while success:
                if frame_no % frame_interval == 0:
                    frame_filename = f"{video_name}_frame_{frame_no}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                success, frame = cap.read()
                frame_no += 1

            cap.release()
            print(f"✅ Extracted frames from {video_file}")

if __name__ == "__main__":
    extract_frames("videos/Training/Videos", "data/frames")
