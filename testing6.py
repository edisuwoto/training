from ultralytics import YOLO
import cv2
import datetime
import mysql.connector

# === Koneksi ke MySQL ===
db = mysql.connector.connect(
    host="localhost",     # ganti dengan IP server MySQL
    user="root",          # username MySQL
    password="",  # password MySQL
    database="pepaya_db"  # nama database
)
cursor = db.cursor()

# === Load model YOLO ===
model = YOLO("weights/best.pt")

# === Buka CCTV / Webcam ===
# ganti dengan RTSP CCTV jika ada
cap = cv2.VideoCapture(0)

# Set untuk menyimpan ID pepaya unik
unique_ids = set()
last_logged_count = -1  # biar tidak spam log kalau jumlah sama

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Deteksi + Tracking ===
    results = model.track(
        source=frame,
        conf=0.2,
        iou=0.4,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # Ambil ID unik dari hasil tracking
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is not None:
                obj_id = int(box.id.item())
                unique_ids.add(obj_id)

    # Gambar hasil deteksi
    annotated_frame = results[0].plot()

    # Hitung jumlah pepaya unik
    total_pepaya = len(unique_ids)
    cv2.putText(
        annotated_frame,
        f"Pepaya unik: {total_pepaya}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # === Simpan log ke MySQL hanya jika ada perubahan jumlah ===
    if total_pepaya != last_logged_count:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = "INSERT INTO deteksi_pepaya (timestamp, jumlah_pepaya) VALUES (%s, %s)"
        cursor.execute(sql, (timestamp, total_pepaya))
        db.commit()
        print(f"[{timestamp}] ✅ Log masuk DB: {total_pepaya} pepaya unik")
        last_logged_count = total_pepaya

    # Tampilkan video
    cv2.imshow("Pepaya Live CCTV", annotated_frame)

    # Keluar dengan 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cursor.close()
db.close()
cv2.destroyAllWindows()
