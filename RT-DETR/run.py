import os
import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import cv2
import time

label_dict = {"id":0,"name":"person","supercategory":"none"},{"id":1,"name":"trolley","supercategory":"grocery"}

def inference(source, output_folder, model_checkpoint, thrh=0.35, img_size=[640, 640], plot=False):
    if isinstance(source, str):
        # single image
        if re.match(r".*\.(png|jpeg|jpg)$", source):
            inference_image(source, output_folder, model_checkpoint, thrh, img_size, plot)
        # single folder
        else:
            inference_folder(source, output_folder, model_checkpoint, thrh, img_size, plot)
    elif isinstance(source, list):
        # list of folders or images or both
        for path in source:
            # single image
            if re.match(r".*\.(png|jpeg|jpg)$", path):
                inference_image(path, output_folder, model_checkpoint, thrh, img_size, plot)
            else:
                # single folder
                inference_folder(source, output_folder, model_checkpoint, thrh, img_size, plot)
    else:
        inference_image(source, output_folder, model_checkpoint, thrh, img_size, plot)


def inference_folder(input_folder, output_folder, model_checkpoint, thrh = 0.35, img_size=[640, 640], plot=False):
    os.makedirs(output_folder, exist_ok=True)
    img_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    size = torch.tensor([img_size])

    img_counter = 1
    for path in img_paths:
        inference_image(path, output_folder, model_checkpoint, thrh, img_size, plot)
        print(f"Inference Progress {img_counter}/{len(img_paths)} - {np.round(img_counter/len(img_paths), 2) * 100}%")
        img_counter += 1


def inference_image(img_path, output_folder, model_checkpoint, thrh = 0.35, img_size=[640, 640], plot=False):
    os.makedirs(output_folder, exist_ok=True)
    size = torch.tensor([img_size])
    if isinstance(img_path, str):
        im = Image.open(img_path).convert('RGB')
    else:
        im = Image.fromarray(img_path).convert('RGB')
    # im = np.resize(im, (img_size[0], img_size[1], 3))
    im = im.resize((img_size[0], img_size[1]))
    im_data = ToTensor()(im)[None]

    sess = ort.InferenceSession(model_checkpoint) # args.file_name
    output = sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    )
    labels, boxes, scores = output
    draw = ImageDraw.Draw(im)
    print(boxes[:5])
    for i in range(im_data.shape[0]):
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        # print(scr, sum(scr > thrh))
        counter = 0
        for b in box:
            # print(scr[counter])
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(label_dict[lab[counter]]['name']), fill=(127, 0, 255), )
            draw.text((b[0], b[1]+10), text=str(np.round(scr[counter], 4)), fill=(127, 0, 255), )
            counter += 1

    if plot:
        plt.imshow(im)
        plt.show()
    # print(f"Inference on image finished, saved in: {os.path.join(output_folder, os.path.basename(img_path))}")
    # im.save(os.path.join(output_folder, os.path.basename(img_path)))
    im.save(os.path.join(output_folder, f"test{np.random.randint(1,999)}.png"))


output_folder = "/Users/johannes/Code/object-detection/RT-DETR/inference_output"
# model_checkpoint = '/Users/johannes/Code/YOLO/RT-DETR/RT-DETR/rtdetr_pytorch/model.onnx'
# model_checkpoint = '/Users/johannes/Code/YOLO/RT-DETR/RT-DETR/rtdetr_pytorch/train_01_epoch_30.onnx' # rv18
# model_checkpoint = '/Users/johannes/Code/YOLO/RT-DETR/RT-DETR/rtdetr_pytorch/rv50-1-ep55.onnx' # rv50-ep55-1
model_checkpoint = '/Users/johannes/Code/YOLO/RT-DETR/RT-DETR/rtdetr_pytorch/rv50-2-full-ep02.onnx' # rv50-2

cap = cv2.VideoCapture(0)
counter = 0
os.makedirs(output_folder, exist_ok=True)

# Check if the webcam is opened successfully
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame!")
        exit()

    # Save the captured image with a timestamp (optional)
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    # filename = f"images/webcam_image_{timestamp}.jpg"
    inference(frame, output_folder, model_checkpoint, thrh=0.5, img_size=[640, 640], plot=False)
    # to-do: look into results to get bounding box coordinates
    
    # save original image
        # cv2.imwrite(filename, frame)
        # print(f"Image saved successfully as: {filename}")
    # time.sleep(1)
    counter += 1
    time.sleep(1)
    if counter >= 5:
        break

# Release the webcam capture
cap.release()
cv2.destroyAllWindows()