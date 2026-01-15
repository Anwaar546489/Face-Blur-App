import streamlit as st
import cv2
import numpy as np
import tempfile
from utils import blur_faces_image, blur_faces_video

st.set_page_config(page_title="Face Blur App", layout="centered")
st.title("🕶️ Face Blur Tool (YOLOv8 + Streamlit)")

mode = st.radio("Choose input type:", ["Image", "Video"])

def save_uploaded_file(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    return temp_file.name

if mode == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        result = blur_faces_image(image)

        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Blurred Image")

        # Save to temp file and create download button
        _, temp_path = tempfile.mkstemp(suffix=".jpg")
        cv2.imwrite(temp_path, result)
        with open(temp_path, "rb") as file:
            st.download_button("📥 Download Blurred Image", file, file_name="blurred_image.jpg", mime="image/jpeg")

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)

        if st.button("🔄 Blur Faces in Video"):
            input_path = save_uploaded_file(uploaded_video)
            output_path = "blurred_video.mp4"

            progress_text = st.empty()
            progress_bar = st.progress(0)

            def update_progress(value):
                percent = int(value * 100)
                progress_bar.progress(percent)
                progress_text.text(f"Processing: {percent}%")

            with st.spinner("Processing video..."):
                try:
                    blur_faces_video(input_path, output_path, update_callback=update_progress)
                    st.success("✅ Processing complete!")
                    st.video(output_path)
                    with open(output_path, "rb") as file:
                        st.download_button("📥 Download Blurred Video", file, file_name="blurred_video.mp4", mime="video/mp4")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
