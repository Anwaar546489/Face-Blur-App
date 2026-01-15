# test_single_image.py
from ultralytics import YOLO
import cv2

# Path to your trained model
model_path = "E:/AI/Project/Project_Ai/weights/face_gender_detector3/weights/best.pt"

# Load YOLO model
model = YOLO(model_path)

# Path to the image you want to test
image_path = "E:/AI/Project/Project_Ai/test11.jpg"

# Run detection
results = model(image_path)

# Get the first (and usually only) result
result = results[0]

# Load image using OpenCV
img = cv2.imread(image_path)

# Draw bounding boxes
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])  # class id

    label = f"{model.names[cls]}: {conf:.2f}"

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)

    # Put text
    cv2.putText(img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Resize for display (540×540)
display_img = cv2.resize(img, (540, 540))

# Show image
cv2.imshow("Detection", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image with detections
save_path = "detection_output.jpg"
cv2.imwrite(save_path, img)
print(f"Saved {save_path}")



# from ultralytics import YOLO
# import cv2

# # Path to your trained model
# model_path = "C:/YOLO_FaceGender_Project/weights/face_gender_detector3/weights/best.pt"
# model = YOLO(model_path)

# # Path to the image
# image_path = "C:/YOLO_FaceGender_Project/test.jpg"
# img = cv2.imread(image_path)

# # Run detection
# results = model(image_path)
# result = results[0]

# # Iterate over detections
# for box in result.boxes:
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     cls = int(box.cls[0])
#     label = model.names[cls]  # class name (ex: 'man', 'woman')

#     # If detection is Woman -> blur it
#     if label.lower() == "woman" or label.lower() == "female":
#         face_region = img[y1:y2, x1:x2]
        
#         # Apply blur (Gaussian)
#         blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)

#         # Replace original region with blurred one
#         img[y1:y2, x1:x2] = blurred_face

# # Resize for display
# display_img = cv2.resize(img, (540, 540))

# # Show image
# cv2.imshow("Blurred Output", display_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save blurred image
# save_path = "blurred_women_output.jpg"
# cv2.imwrite(save_path, img)
# print(f"Saved {save_path}")
