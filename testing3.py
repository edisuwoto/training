from ultralytics import YOLO

# === 1. Load model terbaik hasil training ===
model = YOLO("weights/best.pt")

# === 2. Batch testing ke semua gambar di folder test ===
results = model.predict(
    source="images",   # folder berisi semua gambar test
    conf=0.2,               # confidence minimal 50%
    iou=0.4,                # threshold NMS (supaya tidak double box)
    save=True,              # simpan hasil deteksi ke folder runs/detect/
    save_txt=True,          # simpan koordinat bbox ke file .txt
    project="runs/detect",  # lokasi output
    name="pepaya_test",     # nama folder hasil
    exist_ok=True           # overwrite kalau folder sudah ada
)

print("✅ Batch testing selesai! Hasil disimpan di: runs/detect/pepaya_test/")
