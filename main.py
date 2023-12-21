from ultralytics import YOLO
import cv2
import os
import easyocr
from sentence_transformers import SentenceTransformer

##############################################################################
################################# INITIATION ################################# 
##############################################################################

# convert image to cv2 for easier reading and cropping
def convert_images_to_cv2_values(folder_path):
    image_list = []
    
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            image_list.append(img)
    
    return image_list


# load the models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
detection_model = YOLO('ms_question_detection.pt')

# load datas
question_path = "hw"
cv2image = convert_images_to_cv2_values(question_path)

##############################################################################
############################# question analysis ############################## 
##############################################################################

# filter the question by confidence
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


def q_data_const(detection_result):
    # create container to store the data
    q_data = []
    for result in detect_results:
        xyxy = get_boxes_xyxy_numpy(result)
        q_data.append(xyxy)

    # write data into the container by iteration
    counter = 0
    reader = easyocr.Reader(['en'])
    for page in q_data:
        for question in page:
            x1 = question[0][0]
            y1 = question[0][1] 
            x2 = question[0][2] 
            y2 = question[0][3] 
            image = cv2image[counter]
            cropped_image = image[ round(y1):round(y2),round(x1):round(x2)]
            question[1] = " ".join(reader.readtext(cropped_image, detail = 0))
            question[2] = embedding_model.encode(question[1])
        counter += 1
        for page in range(len(q_data)):
            q_data[page] = sorted_list = sorted(q_data[page], key=lambda x: x[0][1])

    return q_data

def mark_question(q_data):
    for page in range(len(question_data)):
        for question in question_data[page]:
            x1, y1, x2, y2 = question[0]
            image = cv2image[page]
            cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
            cv2.imshow('Image with Boxes', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# detect the questions
detect_results = detection_model.predict(source=cv2image, show = False, device='mps')
question_data = q_data_const(detect_results)
mark_question(question_data)