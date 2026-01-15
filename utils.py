import cv2
from ultralytics import YOLO

# -------------------------------------------------
# Load your TRAINED gender detection model
# Class mapping:
# 0 -> women
# 1 -> men
# -------------------------------------------------
model = YOLO(r"E:\AI\Project\Project_Ai\weights\face_gender_detector3\weights\best.pt")

# -------------------------------------------------
# IMAGE FUNCTION
# -------------------------------------------------
def blur_faces_image(image):
    """
    Blur ONLY female faces (class = 0) in an image
    """
    results = model(image, verbose=False)[0]
    h, w = image.shape[:2]

    for box in results.boxes:
        cls_id = int(box.cls.item())

        # Blur ONLY women
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Clamp coordinates safely
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Strong blur (dynamic size)
        blur_size = max(51, ((x2 - x1) // 2) | 1)
        if blur_size % 2 == 0:
            blur_size += 1

        blurred = cv2.GaussianBlur(face, (blur_size, blur_size), 0)
        image[y1:y2, x1:x2] = blurred

    return image


# -------------------------------------------------
# VIDEO FUNCTION
# -------------------------------------------------
def blur_faces_video(input_path, output_path, update_callback=None):
    """
    Blur ONLY female faces (class = 0) in a video
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("❌ Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure even dimensions (required for mp4)
    width -= width % 2
    height -= height % 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        results = model(frame, verbose=False)[0]
        h, w = frame.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls.item())

            # Blur ONLY women
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            blur_size = max(55, ((x2 - x1) // 2) | 1)
            if blur_size % 2 == 0:
                blur_size += 1

            blurred = cv2.GaussianBlur(face, (blur_size, blur_size), 0)
            frame[y1:y2, x1:x2] = blurred

        out.write(frame)
        frame_num += 1

        if update_callback and total_frames > 0:
            update_callback(frame_num / total_frames)

    cap.release()
    out.release()
