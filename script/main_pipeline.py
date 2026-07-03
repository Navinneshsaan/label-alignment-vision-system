import cv2
import numpy as np
import os
import glob

# 1. SYSTEM TOLERANCES (Relative Calibration)
IDEAL_REL_X = 0   
IDEAL_REL_Y = 0   
MAX_SHIFT_PIXELS = 5    # Tightened to 5
MAX_ROTATION_DEG = 1.0  # Tightened to 1.0

def run_inspection(image_path):
    print(f"\n--- Inspecting: {image_path} ---")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("ERROR: Could not find the image.")
        return

    # 2A. FRONTEND: FIND THE WHITE BOX (Grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    repaired_edges = cv2.dilate(edges, kernel, iterations=1)
    
    box_contours_raw, _ = cv2.findContours(repaired_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = [c for c in box_contours_raw if cv2.contourArea(c) > 5000]
    sorted_boxes = sorted(valid_boxes, key=cv2.contourArea, reverse=True)


    # 2B. FRONTEND: FIND THE COLORED LABEL (HSV Color)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    label_mask = cv2.bitwise_or(mask1, mask2)
    
    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_OPEN, kernel)
    
    label_contours_raw, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_labels = [c for c in label_contours_raw if cv2.contourArea(c) > 1000]
    sorted_labels = sorted(valid_labels, key=cv2.contourArea, reverse=True)


    # 3. BACKEND: GEOMETRIC BOUNDING MATH
    if len(sorted_boxes) == 0 or len(sorted_labels) == 0:
        print("STATUS: FAIL - NO LABEL")
        cv2.putText(frame, "FAIL: NO LABEL", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("Inspection Output", frame)
        cv2.waitKey(0)
        return

    box_contour = sorted_boxes[0]
    label_contour = sorted_labels[0]

    box_rect = cv2.minAreaRect(box_contour)
    (bx, by), (bw, bh), b_angle = box_rect
    bx, by = int(bx), int(by)
    box_pts = np.intp(cv2.boxPoints(box_rect)) 

    label_rect = cv2.minAreaRect(label_contour)
    (lx, ly), (lw, lh), angle = label_rect
    lx, ly = int(lx), int(ly)
    label_pts = np.intp(cv2.boxPoints(label_rect)) 

    relative_x = lx - bx
    relative_y = ly - by
        
    print(f"CALIBRATION INFO -> Label is {relative_x}px (X) and {relative_y}px (Y) away from Box Center")

    delta_x = abs(relative_x - IDEAL_REL_X)
    delta_y = abs(relative_y - IDEAL_REL_Y)

    if angle > 45:
        angle = 90 - angle
    elif angle < -45:
        angle = 90 + angle
    angle = abs(angle)

    # 4. DECISION LOGIC & UI OVERLAY
    if angle > MAX_ROTATION_DEG:
        status = "FAIL (Rotation)"
        color = (0, 0, 255)
    elif delta_x > MAX_SHIFT_PIXELS or delta_y > MAX_SHIFT_PIXELS:
        status = "FAIL (Translational)"
        color = (0, 0, 255)
    else:
        status = "PASS (OK)"
        color = (0, 255, 0) 

    cv2.drawContours(frame, [box_pts], 0, (255, 0, 0), 2)
    cv2.drawContours(frame, [label_pts], 0, color, 2)
    
    cv2.circle(frame, (bx, by), 5, (255, 0, 0), -1)
    cv2.circle(frame, (lx, ly), 5, color, -1)
    cv2.line(frame, (bx, by), (lx, ly), (255, 255, 255), 1)     
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"STATUS: {status}", (20, 40), font, 0.8, color, 2)
    cv2.putText(frame, f"Rel Shift: dx={int(delta_x)}, dy={int(delta_y)}", (20, 70), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Angle: {angle:.2f} deg", (20, 95), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Inspection Output", frame)
    print("Press any key to close and move to the next image...")
    cv2.waitKey(0)

# AUTOMATED DATASET SCANNER
if __name__ == "__main__":
    dataset_path = "dataset"
    files = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(images)} images to process.")
    for img_path in images:
        run_inspection(img_path)
    
    cv2.destroyAllWindows()