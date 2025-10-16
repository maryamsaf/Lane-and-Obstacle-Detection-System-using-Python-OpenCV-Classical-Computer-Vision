# 🚗 Lane and Obstacle Detection System — Python, OpenCV, Classical Computer Vision

A real-time classical computer vision system for lane and obstacle detection using OpenCV and Python.  
It uses ROI masking, Canny + Hough line fitting, and inverse thresholding"" for obstacle detection, displaying an on-screen ""HUD"" with FPS, lane-fit accuracy (R²), lane stability, and STOP/MOVE logic — all ""without deep learning"".

---

## 📊 Results Summary
| Metric | Value |
|:--|:--|
| ""Frames Processed"" | 1,231 |
| ""Average FPS"" | 13.5 |
| ""Lane Fit R² (Left/Right)"" | 0.94 / 0.98 |
| ""Lane Stability (px std-dev)"" | 6.5 |
| ""False-STOP Detections"" | 0 % |
| ""STOP Time"" | 0.0 s |

> 🧠 *All results measured on CPU (540p video) using only OpenCV and NumPy.*

---

## ⚙️ Features
- ""Dynamic ROI Masking"" for focusing on road lanes  
- ""Canny Edge + Hough Transform"" for lane line fitting  
- ""Inverse Thresholding & Morphology"" for obstacle detection  
- ""On-Screen HUD"" displaying FPS, R² accuracy, and STOP/MOVE status  
- ""Pure Classical CV"" — no machine learning or pre-training required  

---

## 🧩 Tech Stack
- ""Language:"" Python 3.10+
- ""Libraries:"" OpenCV, NumPy  
- ""Approach:"" Classical Computer Vision (No ML)

---

## 🚀 Installation
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate    # On Windows
# source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
