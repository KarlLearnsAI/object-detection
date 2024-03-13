import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


model_checkpoint = '/Users/johannes/Code/YOLO/RT-DETR/Abgabge/rv50-2-combined-ep13.onnx'
thrh = 0.55
resized_video_path = '/Users/johannes/Code/YOLO/RT-DETR/Abgabge/test-video-640x640.mp4'
inferred_path = '/Users/johannes/Code/YOLO/RT-DETR/Abgabge/test-video-640x640_inferred.mp4'

label_dict = {"id":0,"name":"person","supercategory":"none"},{"id":1,"name":"trolley","supercategory":"grocery"}


def inference_frame(frame, model_checkpoint, thrh=0.35, line_thickness=4, img_size=[640, 640], plot=False):
    size = torch.tensor([img_size])
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb_frame)
    im = im.resize((img_size[0], img_size[1]))
    im_data = ToTensor()(im)[None]

    sess = ort.InferenceSession(model_checkpoint)  # args.file_name
    output = sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    )

    labels, boxes, scores = output
    draw = ImageDraw.Draw(im)
    line_thickness = line_thickness

    for i in range(im_data.shape[0]):
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        counter = 0
        for b in box:
            x_min, y_min, x_max, y_max = b

            if label_dict[lab[counter]]['name'] == "person":
                color = (0, 0, 255)
            elif label_dict[lab[counter]]['name'] == "trolley":
                color = (255, 0, 0)
            else:
                color = (127, 0, 255)

            draw.rectangle(list(b), outline=color, width=line_thickness)

            text_x = int(x_min + (x_max - x_min) * 0.05)
            text_y = int(y_min + (y_max - y_min) * 0.1)

            draw.text((text_x, text_y), text=str(label_dict[lab[counter]]['name']), fill=(127, 0, 255))
            draw.text((text_x, text_y + 10), text=str(np.round(scr[counter], 4)), fill=(127, 0, 255))
            counter += 1

    if plot:
        plt.imshow(im)
        plt.show()

    im = np.array(im)
    bgr_frame = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    return np.array(bgr_frame)


cap = cv2.VideoCapture(resized_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Error opening video!")
    exit()

writer = cv2.VideoWriter(inferred_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, frame = cap.read()
    if not success:
        print("No more frames to process.")
        break
    
    inferenced_frame = inference_frame(frame, model_checkpoint, thrh=thrh, line_thickness=5, img_size=[640, 640], plot=False)

    writer.write(inferenced_frame)

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Video processing and concatenation complete!")