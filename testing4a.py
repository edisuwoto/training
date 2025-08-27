import cv2
import glob

images = sorted(glob.glob("images/*.jpg"))
if not images:
    raise FileNotFoundError("Folder gambar pepaya kosong!")

frame = cv2.imread(images[0])
h, w, _ = frame.shape
out = cv2.VideoWriter(
    "pepaya_demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h)
)

for img in images:
    frame = cv2.imread(img)
    out.write(frame)

out.release()
print("Video pepaya_demo.mp4 berhasil dibuat.")
