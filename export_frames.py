import cv2
import os

video_path = 'timecr1_gameplay.mp4'
output_folder = 'frames'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 5)  # Extract every Nth frame for 5 FPS
frame_count = 0
extracted_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        cv2.imwrite(f'{output_folder}/frame_{extracted_count:05d}.jpg', frame)
        extracted_count += 1
    frame_count += 1

cap.release()
