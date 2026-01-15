import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO(r"E:\AI\Project\Project_Ai\weights\face_gender_detector3\weights\best.pt")

# Load image
img = cv2.imread("E:/AI/Project/Project_Ai/test12.jpg")
h, w, _ = img.shape

# Run inference
results = model(img, conf=0.3)[0]

print("Model class names:", model.names)

for box in results.boxes:
    cls_id = int(box.cls.item())
    cls_name = model.names[cls_id].lower()

    # DEBUG PRINT (keep this once to confirm)
    print(f"Detected: {cls_id} -> {cls_name}")

    # ✅ BLUR ONLY FEMALE
    if cls_name != "women":
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Clamp coordinates
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        continue

    # Strong adaptive blur
    k = int(min(x2 - x1, y2 - y1) / 2)
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(face, (k, k), 50)
    img[y1:y2, x1:x2] = blurred

cv2.imwrite("female_faces_only_blurred.jpg", img)
print("✅ ONLY female faces blurred")
