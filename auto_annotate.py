from ultralytics import YOLO
import os

model = YOLO('best.pt')  # Your pretrained model

frames_folder = 'frames'
output_folder = 'annotations'
os.makedirs(output_folder, exist_ok=True)

for file_name in sorted(os.listdir(frames_folder)):
    frame_path = os.path.join(frames_folder, file_name)
    results = model(frame_path)

    # Save as YOLO format (.txt files)
    results[0].save_txt(txt_file=f'{output_folder}/{file_name[:-4]}.txt')
