import cv2
import matplotlib.pyplot as plt
import datetime
import time
import os
from glob import glob
from ultralytics import YOLO
from pathlib import Path

# input_folder = "/Users/johannes/Code/Work/Test_Data"
output_folder = "/Users/johannes/Code/object-detection/data-management/images"
path_to_best_model = '/Users/johannes/Code/YOLO/YOLOv8/runs/detect/train18/weights/best.pt'

# sometimes source 1 instead of 0
cap = cv2.VideoCapture(0)
counter = 0
os.makedirs(output_folder, exist_ok=True)

# model = YOLO(path_to_best_model)
model = YOLO('yolov8n')

# Check if the webcam is opened successfully
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame!")
        exit()

    # Save the captured image with a timestamp (optional)
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    # filename = f"images/webcam_image_{timestamp}.jpg"
    results = model(source=frame, project="images/detect", name="", save=False, conf=0.5, show=True)
    # to-do: look into results to get bounding box coordinates
    
    # save original image
        # cv2.imwrite(filename, frame)
        # print(f"Image saved successfully as: {filename}")
    # time.sleep(1)
    counter += 1
    if counter >= 0:
        break

# Release the webcam capture
cap.release()
cv2.destroyAllWindows()