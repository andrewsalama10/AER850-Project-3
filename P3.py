# ======================================================================
# AER850 – PROJECT 3
# ======================================================================

# ===========================================================
# Step 1: Object Masking - 30 Marks
# ===========================================================
import cv2
import numpy as np

print("===== STEP 1: OBJECT MASKING =====")

# load original motherboard image
image_path = "P3Data/motherboard_image.JPEG"
img = cv2.imread(image_path)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
orig = img.copy()
# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply Gaussian Blur to reduce noise and improve contour detection
blur = cv2.GaussianBlur(gray, (9, 9), 0)

# edge detection
edges = cv2.Canny(blur, 50, 150)

# thresholding
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# contour detection
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# choose largest contour
if contours:
    largest_contour = max(contours, key = cv2.contourArea)
    
    # create mask and extract PCB 
    mask = np.zeros_like(gray, dtype = np.uint8)
    
    # draw the filled largest contour onto the mask
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness = cv2.FILLED)
    
    # extract the PCB using the mask
    extracted = cv2.bitwise_and(orig, orig, mask = mask)

    # save outputs
    cv2.imwrite("Step1_Gray.png", gray)
    cv2.imwrite("Step1_Edges.png", edges)
    cv2.imwrite("Step1_Mask.png", mask)
    cv2.imwrite("Step1_Extracted_PCB.png", extracted)
    
    print("Step 1 complete. Files saved")

# # ===========================================================
# # Step 2: YOLOv11 Training - 30 Marks
# # ===========================================================

# print("\n===== STEP 2: YOLOv11 TRAINING =====")

print("\n===== STEP 2: YOLOv11 TRAINING =====")

from ultralytics import YOLO
import torch

print("CUDA:", torch.cuda.is_available())

# load YOLO model (nano version)
model = YOLO("yolo11n.pt") # Pretrained YOLOv11-nano

# train settings
results = model.train(
  data = "P3Data/data/data.yaml",
  epochs = 175,
  imgsz = 1024,
  batch = 8,
  name = "pcb_yolo11",
  device = "0",
  workers = 2,
  verbose = True
  )

print("Step 2 complete – YOLOv11 model trained.")
print("Results saved in runs/detect/pcb_yolo11/")

# ===========================================================
# Step 3: Evaluation - 10 Marks
# ===========================================================

print("\n===== STEP 3: MODEL EVALUATION =====")

# Load trained weights
model = YOLO("runs/detect/pcb_yolo11/weights/best.pt")

# Image 1 Prediction
results1 = model.predict(
    source = "P3Data/data/evaluation/ardmega.jpg",
    save = True,
    conf = 0.25,
    imgsz = 1024,
    line_width = 6
)
print("Processed test image 1: ardmega.jpg")

# Image 2 Prediction
results2 = model.predict(
    source = "P3Data/data/evaluation/arduno.jpg",
    save = True,
    conf = 0.25,
    imgsz = 1024,
    line_width = 3
)
print("Processed test image 2: arduno.jpg")

# Image 3 Prediction
results3 = model.predict(
    source = "P3Data/data/evaluation/rasppi.jpg",
    save = True,
    conf = 0.25,
    imgsz = 1024,
    line_width = 4
)
print("Processed test image 3: rasppi.jpg")

print("Step 3 complete – Predictions saved in runs/detect/predict/")
print("\n===== PROJECT 3 COMPLETE =====")