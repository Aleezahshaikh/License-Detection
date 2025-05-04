import cv2
import torch
import easyocr
import os
from pathlib import Path
from ultralytics import YOLO

reader = easyocr.Reader(['en'])

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

def run_inference(model_path, source_path):
    model = YOLO(model_path)
    is_image = Path(source_path).suffix.lower() in ['.jpg', '.jpeg', '.png']
    
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    if is_image:
        results = model(source_path)
        frame = results[0].orig_img.copy()

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            name = model.names[cls]

            label = f"{name} {conf:.2f}"
            ocr_text = getOCR(frame, xyxy)
            if ocr_text:
                label = ocr_text

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(save_dir, "result.jpg"), frame)
        print("✅ Image inference done. Saved to: outputs/result.jpg")

    else:
        cap = cv2.VideoCapture(source_path)
        frame_count = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(save_dir, 'output_video.mp4'), fourcc, 30.0, 
                              (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    name = model.names[cls]

                    label = f"{name} {conf:.2f}"
                    ocr_text = getOCR(frame, xyxy)
                    if ocr_text:
                        label = ocr_text

                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        print(f"✅ Video inference completed. Saved to: {save_dir}/output_video.mp4")

if __name__ == "__main__":
    model_path = input("Enter path to YOLO model (.pt): ").strip()
    source_path = input("Enter path to image or video: ").strip()
    run_inference(model_path, source_path)

