# Automated Machine Vision System for Packaging Label Alignment

## 📌 Project Overview
In industrial packaging lines, manual label inspection is slow, inconsistent, and prone to human fatigue. Incorrect labels can reduce product quality and damage customer trust. 

This project is an automated machine vision system designed to inspect product labels accurately and consistently. It locates the label on a packaging bottle, measures how far it deviates from a perfect reference position, and automatically classifies the product as **PASS** or **DEFECTIVE**.

## 🎯 Objectives
* **Develop** a machine vision system for automatic label alignment inspection.
* **Detect** common manufacturing defects: missing labels, translation defects (X/Y shift), and rotation defects (angular tilt).
* **Classify** packaging products using deterministic image processing and decision logic.

## 🗂️ Dataset
A custom dataset was collected to simulate an industrial packaging line using "MyMilk Fresh Laban" bottles. The data covers four critical operational categories:
1. **Pass:** Label is present, centered, and correctly aligned.
2. **Missing Label:** Label is completely absent from the packaging.
3. **Translation Defect:** Label is shifted horizontally or vertically from the center.
4. **Rotation Defect:** Label is tilted from the correct angle.

## ⚙️ Methodology & Vision Pipeline
This system utilizes a rule-based, edge-and-pattern pipeline built with OpenCV.

1. **Image Acquisition:** Captured using a fixed camera setup with standard lighting.
2. **Preprocessing:** Converted to Grayscale and smoothed using Gaussian Blur to reduce noise.
3. **Segmentation & Edge Detection:** 
   * Canny Edge Detection and Dilation isolate the boundaries of the white packaging box.
   * HSV Color Masking extracts the specific red elements of the label.
4. **Feature Measurement:** 
   * `cv2.minAreaRect` is used to calculate the box center `(bx, by)` and the label center `(lx, ly)`.
   * The relative shift in pixels (`dx`, `dy`) and absolute rotation angle (`θ`) are computed.
5. **Decision Logic:** Cascading rules evaluate the measurements against calibrated tolerance thresholds.

## 📊 Thresholds & Classification Logic
The system is highly deterministic. Every verdict traces directly back to a measured sub-pixel value:

* **PASS:** `dx <= 5 px` AND `dy <= 5 px` AND `angle <= 1.0°`.
* **TRANSLATION DEFECT:** `dx > 5 px` OR `dy > 5 px`.
* **ROTATION DEFECT:** `angle > 1.0°`.
* **MISSING LABEL:** No box or no red label detected.

## 📷 Results & Visual Feedback
<img width="462" height="517" alt="Status_Pass" src="https://github.com/user-attachments/assets/3da062c2-565f-411d-86e4-afc86656363b" />
<img width="417" height="417" alt="Status_Fail" src="https://github.com/user-attachments/assets/90c21183-5708-41e4-b57b-3fb5bdfca691" />


* **UI Overlay:** Draws the bounding box (blue), label outline (green/red), centroids, and live shift vector measurements directly onto the frame.

## 🚀 Challenges & Future Improvements
While the current OpenCV operations are lightweight and run in real-time without requiring a GPU, the system faces real-world environmental challenges:
* **Limitations:** Shiny bottle surfaces cause glare, and ambient lighting variations can break fixed HSV thresholding. Curved bottle surfaces also slightly distort the label.
* **Proposed Upgrades:** 
  * Implementing a Deep Learning (CNN) model to replace fixed HSV rules for higher robustness.
  * Upgrading from a standard CMOS rolling shutter to a high-performance industrial Global Shutter camera with a telecentric lens to eliminate optical parallax.
  * Integrating OCR (Optical Character Recognition) and barcode checking.
