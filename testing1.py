from ultralytics import YOLO
import cv2
import os

# === 1. Load model dasar YOLOv8 nano ===
# Jika file belum ada, YOLO akan otomatis download dari repo Ultralytics
model = YOLO("yolov8n.pt")

# === 2. Training model pakai dataset pepaya ===
model.train(
    data="data.yaml",   # path ke data.yaml
    epochs=50,          # jumlah epoch training
    imgsz=640,          # resolusi gambar input
    batch=16,           # batch size
    workers=2           # jumlah worker (ubah sesuai CPU kamu)
)
