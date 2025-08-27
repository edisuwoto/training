from ultralytics import YOLO
import cv2

# Load model terbaik hasil training
#model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("weights/best.pt")

# Tes gambar dengan setting threshold
results = model.predict(
    source="data/p2.jpg",
    conf=0.5,   # confidence minimal 50%
    iou=0.45    # NMS threshold (semakin kecil makin ketat)
)

# Tampilkan hasil
for r in results:
    im = r.plot()
    cv2.imshow("Pepaya Detection", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
