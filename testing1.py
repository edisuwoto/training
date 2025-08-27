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

# === 3. Evaluasi model ===
metrics = model.val()  # hasil evaluasi mAP, precision, recall
print("📊 Hasil evaluasi:", metrics)

# === 4. Inference / Testing ke 1 gambar ===
# Pastikan file best.pt ada di folder hasil training
best_model_path = "runs/detect/train/weights/best.pt"
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"❌ File model {best_model_path} tidak ditemukan!")

best_model = YOLO(best_model_path)

# Tes ke gambar pepaya dari folder test
test_image = "data/pepaya1.jpg"  # ganti sesuai dataset kamu
if not os.path.exists(test_image):
    raise FileNotFoundError(f"❌ File test {test_image} tidak ditemukan!")

results = best_model(test_image)

# Tampilkan hasil dengan OpenCV
for r in results:
    im = r.plot()  # gambar hasil deteksi
    cv2.imshow("Pepaya Detection", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
