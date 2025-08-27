from ultralytics import YOLO
import cv2
import datetime

# === Load model YOLO terbaik hasil training ===
model = YOLO("weights/best.pt")

# === Buka file video MP4 ===
cap = cv2.VideoCapture("pepaya_demo.mp4")  # ganti dengan path video kamu

# Ambil info video (biar output sama ukuran dan fps)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Writer untuk simpan video hasil deteksi
out = cv2.VideoWriter("5pepaya_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Set untuk menyimpan ID pepaya unik
unique_ids = set()

# Buka file log untuk simpan hasil
log_file = open("pepaya_log.txt", "a")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Deteksi + Tracking ===
    results = model.track(
        source=frame,
        conf=0.2,            # confidence threshold
        iou=0.4,             # NMS threshold
        persist=True,        # pertahankan ID antar frame
        tracker="bytetrack.yaml"
    )

    # Ambil ID unik dari hasil tracking
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is not None:   # pastikan ada ID
                obj_id = int(box.id.item())
                unique_ids.add(obj_id)

    # Gambar hasil deteksi + tracking di frame
    annotated_frame = results[0].plot()

    # Tampilkan jumlah pepaya unik di layar
    cv2.putText(
        annotated_frame,
        f"Pepaya unik: {len(unique_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Simpan frame ke video output
    out.write(annotated_frame)

    # Simpan log tiap update jumlah pepaya
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] Pepaya unik: {len(unique_ids)}\n")

    # Tampilkan preview video
    cv2.imshow("Pepaya Video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
