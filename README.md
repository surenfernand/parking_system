
# 🚗 Smart Parking Detection System using YOLOv8 and OpenCV

This project is a **Smart Parking Detection System** built to explore real-time object detection using **YOLOv8** and **OpenCV**. It can detect vehicles in a video stream, define custom parking slot regions, check occupancy, and dynamically display available parking spots. This is a great entry project into computer vision and smart city applications.

---

## 🧠 Key Features

- ✅ Real-time vehicle **detection and tracking**
- ✅ **Custom parking slot regions** using polygonal shapes
- ✅ **Occupancy detection** for each slot
- ✅ **Dynamic counting** of total and available parking spots
- ✅ Real-time **visual overlays** on the video feed

---

## 🔧 Tech Stack

- **Python 3.x**
- **YOLOv8** (via [Ultralytics](https://github.com/ultralytics/ultralytics))
- **OpenCV**
- **NumPy**
- Optional: Webcam or pre-recorded video input

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-parking-detection.git
cd smart-parking-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the System

```bash
python main.py
```

> ⚠️ Make sure YOLOv8 weights are downloaded or configured in the script.

---

## 📸 How It Works

1. Loads a video stream (live or recorded).
2. Detects vehicles frame-by-frame using YOLOv8.
3. Checks if detected vehicles overlap with user-defined parking slot polygons.
4. Updates and displays real-time availability on the video interface.

---

## 📍 Use Cases

- 🏬 Shopping malls  
- ✈️ Airports  
- 🏢 Office buildings  
- 🌆 Smart city infrastructure  

---

## 🙌 Why I Built This

I created this project to get hands-on experience with **OpenCV**, **YOLOv8**, and **real-time computer vision**. It taught me how to handle live video processing, object detection, region mapping, and data overlays.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributions

Pull requests and feedback are welcome! Let's make parking smarter together.
