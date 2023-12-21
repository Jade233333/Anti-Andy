from ultralytics import YOLO
import cv2
import os
import easyocr
from sentence_transformers import SentenceTransformer
import mysql.connector
from tqdm import tqdm
import pickle

##############################################################################
################################# INITIATION ################################# 
##############################################################################

# convert image to cv2 for easier reading and cropping
def convert_images_to_cv2_values(folder_path):
    image_list = [] 
    
    for filename in tqdm(os.listdir(folder_path), desc="Loading images"):
        img = cv2.imread(os.path.join(folder_path, filename))
        paper_id = filename.split('.')[0]
        if img is not None:
            image_list.append([paper_id, img])

    return image_list


# load the models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
detection_model = YOLO('ms_question_detection.pt')

# load datas
question_path = "raw_question_bank/multi/image_test"
ls_img = convert_images_to_cv2_values(question_path) 
cv2image = [question[1] for question in ls_img]
cv2image_names =  [question[0] for question in ls_img]


##############################################################################
############################# question analysis ############################## 
##############################################################################

# filter the question by confidence
def get_boxes_xyxy_numpy(result, conf_threshold=0.69):

    # Create a mask of booleans where the confidence is greater than the given threshold
    mask = result.boxes.conf > conf_threshold

    # Use this mask to select the rows from xyxy and cls
    selected_xyxy = result.boxes.xyxy[mask].tolist()
    selected_cls = result.boxes.cls[mask].tolist()

    # Combine selected_xyxy and selected_cls into a list of lists
    # Ensure that selected_xyxy and selected_cls have the same length
    assert len(selected_xyxy) == len(selected_cls), "Lengths of selected bounding boxes and classes do not match."

    # Combine bounding boxes and their corresponding classes
    combined = [[xyxy_val, "", cls_val, ""] for xyxy_val, cls_val in zip(selected_xyxy, selected_cls)]

    return combined


def q_data_const(detection_result):
    # create container to store the data
    q_data = []
    for result in tqdm(detect_results, desc="Processing OCR and generating embeddings"):
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
            name = cv2image_names[counter]
            cropped_image = image[ round(y1):round(y2),round(x1):round(x2)]
            question[1] = " ".join(reader.readtext(cropped_image, detail = 0))
            question[2] = embedding_model.encode(question[1])
            question[3] = name
        counter += 1
        for page in range(len(q_data)):
            q_data[page] = sorted(q_data[page], key=lambda x: x[0][1])

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
print("Object detection with YOLOv8...")
print(len(cv2image))
detect_results = detection_model.predict(source=cv2image, show = False, device='mps')
print("Processing OCR and generating embeddings...")
question_data = q_data_const(detect_results)
# mark_question(question_data)


##############################################################################
############################# insert to database ############################# 
##############################################################################
def insert_data(question_data):

    # create a connection to the database
    cnx = mysql.connector.connect(
        host='localhost',
        user='root',
        password='hello',
        database='anti_andy'
    )
    # create a cursor object
    cursor = cnx.cursor() 

    for page in tqdm(range(len(question_data)), desc="Inserting data into database"):
        for question in range(len(question_data[page])):
            # data to insert
            data = {
                'text': question_data[page][question][1],
                'embedding': pickle.dumps(question_data[page][question][2]),
                'paper': question_data[page][question][3],
                'page': page,
                'q_number': question 
            }

            # prepare insert statement
            add_data = ("INSERT INTO question_bank"
                        "(text, embedding, paper, page, q_number) "
                        "VALUES (%(text)s, %(embedding)s, %(paper)s, %(page)s, %(q_number)s)")

            # execute the statement
            cursor.execute(add_data, data)

            # commit the changes
            cnx.commit()

    # close the cursor and connection
    cursor.close()
    cnx.close()

insert_data(question_data)