import cv2
from rfdetr import RFDETRBase
from tqdm import tqdm
import numpy as np
import os

CHECKPOINT_PATH = r"D:\Pycharm\rf_detr\output_model\checkpoint_best_regular.pth"
VIDEO_PATH = r"D:\Pycharm\rf_detr\video_42.mp4"
OUTPUT_VIDEO_PATH = r"D:\Pycharm\rf_detr\wyjscie_video.avi"

CLASS_NAMES = ['Potato']

os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH, num_classes=1)


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Nie można otworzyć wideo: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video params: fps={fps}, width={width}, height={height}, total_frames={total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError(f"Nie udało się zainicjalizować VideoWriter: {OUTPUT_VIDEO_PATH}")

pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Przetwarzanie klatek")
frame_num = 0

try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = model.predict(frame_rgb, threshold=0.3)

        if len(detections.xyxy) > 0:
            labels = []
            for cls, conf in zip(detections.class_id, detections.confidence):
                cls_i = int(cls) - 1
                if 0 <= cls_i < len(CLASS_NAMES):
                    labels.append(f"{CLASS_NAMES[cls_i]} {conf:.2f}")
                else:
                    labels.append(f"Nieznana klasa ({cls_i}) {conf:.2f}")

            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 5)
                label = labels[i] if i < len(labels) else "Błąd etykiety"
                cv2.putText(frame_bgr, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

        out.write(frame_bgr)

        if frame_num % 10 == 0:
            if len(detections.xyxy) > 0:
                print(f"Klatka {frame_num}: {len(detections.xyxy)} detekcji, conf przykładowe: {detections.confidence[:3]}")
            else:
                print(f"Klatka {frame_num}: Brak detekcji")

        frame_num += 1
        pbar.update(1)

finally:
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Zakończono. Wynik zapisany w: {OUTPUT_VIDEO_PATH}")
