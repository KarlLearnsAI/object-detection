import os
from glob import glob
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Define input and output folders
# input_folder = "/Users/johannes/Code/Work/Exports/Random_Test"
# input_folder = "/Users/johannes/Code/Work/Test_Data"
output_folder = "/Users/johannes/Code/object-detection/YOLOv8/inference_output"
path_to_best_model = '/Users/johannes/Code/YOLO/YOLOv8/runs/detect/train18/weights/best.pt'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define supported image extensions
# image_extensions = ["*.png", "*.jpg", "*.jpeg"]

# Find all image files
# image_paths = []
# for extension in image_extensions:
    # image_paths.extend(glob(os.path.join(input_folder, extension)))

model = YOLO(path_to_best_model)

results = model(source=0, project=output_folder, save=False, conf=0.1, show=True)
# results = model('https://ultralytics.com/images/bus.jpg', save=True) # stream=True, visualize=True (shows what the model sees as important?)
# Process each image