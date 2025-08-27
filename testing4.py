from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")

cap = cv2.VideoCapture("pepaya_demo.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("pepaya_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(
        source=frame, 
        conf=0.2, 
        iou=0.4
        )[0]
    annotated = results.plot()
    out.write(annotated)
    cv2.imshow("Deteksi Pepaya", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Deteksi selesai, hasil simpan di pepaya_result.mp4")
