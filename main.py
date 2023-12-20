from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import easyocr
from matplotlib import pyplot as plt

def get_boxes_xyxy_numpy(result, conf_threshold=0.7):

    # Create a mask of booleans where the confidence is greater than the given threshold
    mask = result.boxes.conf > conf_threshold

    # Use this mask to select the rows from xyxy and cls
    selected_xyxy = result.boxes.xyxy[mask].tolist()
    selected_cls = result.boxes.cls[mask].tolist()

    # Combine selected_xyxy and selected_cls into a list of lists
    # Ensure that selected_xyxy and selected_cls have the same length
    assert len(selected_xyxy) == len(selected_cls), "Lengths of selected bounding boxes and classes do not match."

    # Combine bounding boxes and their corresponding classes
    combined = [[xyxy_val, "", cls_val] for xyxy_val, cls_val in zip(selected_xyxy, selected_cls)]

    return combined



def convert_images_to_cv2_values(folder_path):
    image_list = []
    
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            image_list.append(img)
    
    return image_list


# choose the trained model
model = YOLO('ms_question_detection.pt')

# get the cv2 array of image
cv2image = convert_images_to_cv2_values("hw")
modified_cv2image = cv2image

# detect the questions
results = model.predict(source=cv2image, show = False, device='mps')

# create a list of arrays, using indext to find page and question number, eg.[0][0] stand for the first quesion on first page
list_boxes_xy = []
for result in results:
    xyxy = get_boxes_xyxy_numpy(result)
    list_boxes_xy.append(xyxy)

counter = 0
reader = easyocr.Reader(['en'])
for i in list_boxes_xy:
    for j in i:
        x1 = j[0][0]
        y1 = j[0][1] 
        x2 = j[0][2] 
        y2 = j[0][3] 
        image = cv2image[counter]
        cropped_image = image[ round(y1):round(y2),round(x1):round(x2)]
        j[1] = " ".join(reader.readtext(cropped_image, detail = 0))
    counter += 1

print(list_boxes_xy[0][0])


for page in range(len(list_boxes_xy)):
    for question in list_boxes_xy[page]:
        x1, y1, x2, y2 = question[0]
        image = cv2image[page]
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
        cv2.imshow('Image with Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


