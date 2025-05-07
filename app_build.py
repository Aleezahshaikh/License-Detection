import streamlit as st
import cv2
import easyocr
import os
from pathlib import Path
from ultralytics import YOLO
import tempfile

# Initialize EasyOCR reader (CPU only)
reader = easyocr.Reader(['en'], gpu=False)

# OCR from cropped region
def getOCR(im, coors):
    x1, y1, x2, y2 = map(int, coors)
    crop = im[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    ocr_text = ""

    for result in results:
        if len(results) == 1:
            ocr_text = result[1]
        elif len(results) > 1 and len(result[1]) > 6 and result[2] > 0.2:
            ocr_text = result[1]
    return ocr_text

# Inference function
def run_inference(model_path, source_path, apply_ocr=False):
    model = YOLO(model_path)
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)

    results = model(source_path)
    frame = results[0].orig_img.copy()

    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        name = model.names[cls]

        label = f"{name} {conf:.2f}"

        if apply_ocr:
            ocr_text = getOCR(frame, xyxy)
            if ocr_text:
                label = ocr_text

        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    filename = "result_with_ocr.jpg" if apply_ocr else "result_detect.jpg"
    output_path = os.path.join(save_dir, filename)
    cv2.imwrite(output_path, frame)
    return output_path
# Streamlit UI
st.set_page_config(page_title="License Plate Detection", layout="centered")
st.title("🚘 License Plate Detection and OCR System")
st.write("""
This system is designed to automatically detect license plates from vehicle images and extract the license number using Optical Character Recognition (OCR). It helps in automating surveillance, vehicle identification, and traffic management.

**Advantages**:
- Eliminates manual data entry errors
- Enhances law enforcement and parking management
- Real-time and accurate plate recognition

**🛠️ Tech Stack Used**:
- [Streamlit](https://streamlit.io/): For creating the interactive web application
- [YOLOv8](https://github.com/ultralytics/ultralytics): For real-time object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR): For reading text from license plates
- [OpenCV](https://opencv.org/): For image processing
- [Python](https://www.python.org/): Core programming language powering the system
""")
st.markdown("## 💡 License Plate Detection Only")
image_file_1 = st.file_uploader("📤 Upload Image for Detection Only", type=["jpg", "jpeg", "png"], key="detect_only")

if image_file_1:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file_1.name).suffix) as tmp_image1:
        tmp_image1.write(image_file_1.read())
        image_path1 = tmp_image1.name

    if st.button("🔍 Detect License Plate", key="btn_detect_only"):
        model_path = "best.pt"
        output_image = run_inference(model_path, image_path1, apply_ocr=False)
        st.image(output_image, caption="Detected License Plate", use_container_width=True)

st.markdown("---")
st.markdown("## 🔠 License Plate Detection with OCR")
image_file_2 = st.file_uploader("📤 Upload Image for Detection + OCR", type=["jpg", "jpeg", "png"], key="detect_ocr")

if image_file_2:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file_2.name).suffix) as tmp_image2:
        tmp_image2.write(image_file_2.read())
        image_path2 = tmp_image2.name

    if st.button("🧠 Detect and Read License Plate", key="btn_detect_ocr"):
        model_path = "best.pt"
        output_image = run_inference(model_path, image_path2, apply_ocr=True)
        st.image(output_image, caption="Detected License Plate with OCR", use_container_width=True)
        st.success("✅ OCR processing complete. Extracted license number is shown on the image.")
