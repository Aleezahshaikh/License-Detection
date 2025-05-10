# 🚗 License Plate Detection & OCR-Based System

A powerful web application built with **Streamlit**, powered by **OpenCV**, **EasyOCR**, and **Ultralytics YOLOv8** to detect and read license plates from images in real-time — all in your browser!

![Demo](https://user-images.githubusercontent.com/demo-image.gif)

---

## 🔧 Tech Stack

| Layer        | Technology              |
|--------------|--------------------------|
| Frontend UI  | Streamlit                |
| Computer Vision | OpenCV, Ultralytics YOLOv8 |
| Text Extraction | EasyOCR               |
| Deployment   | Streamlit Cloud          |
| Others       | Python, urllib3, watchdog, tzdata |

---

## 🚀 Features

- 📸 Upload an image or use camera input
- 🧠 Detect license plates using YOLOv8
- 🔍 Read text from plates using EasyOCR
- ⚡ Fast & lightweight — deploys in seconds!
- ☁️ Deployed easily on Streamlit Cloud

---

## 📁 Project Structure

├── app_build.py # Main Streamlit app

├── requirements.txt # Python dependencies

├── packages.txt # System-level dependencies for Streamlit Cloud

├── models/ # YOLOv8 weights and config

├── utils/ # Preprocessing and helper scripts

└── README.md # You're here!

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/license-detection.git
   cd license-detection
   ```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```
3. **Run the app locally**

```bash
streamlit run app_build.py
```
---

## 🌍 Deploy on Streamlit Cloud

Make sure you have the following in your repo:

✅ `requirements.txt`  
✅ `packages.txt` 

Then:

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)  
2. Connect your GitHub repository  
3. Click **Deploy**  

That's it! 🎉

---

### 🔗 Live Demo

👉 [Check out the deployed app here](https://license-detection-with-ocr.streamlit.app/)  


