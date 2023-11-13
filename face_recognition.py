import numpy as np
import pandas as pd
import cv2

import redis

# insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise  # for cosine similarity calculation

# connect to redis cloud
host = 'redis-12830.c302.asia-northeast1-1.gce.cloud.redislabs.com'
port_no = 12830
password = '0k137l1gcAW3TQMjUeQ06oc7vQTMKsAq'

r = redis.StrictRedis(host=host, port=port_no, password=password)

# buffalo_l model face analysis configuration
model_l = FaceAnalysis(name='buffalo_l',
                       root='/Users/boss/Desktop/Notes/4_attendance_app/insightface_model/models/buffalo_l',
                       providers=['CPUExecutionProvider '])  # CUDAExecutionProvider incase GPU

model_l.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)  # input size

# using cosine similarity function to compare vectors and identify the person
def identification(df, test_vector, name_role=['Name', 'Role'], threshold=0.5):
    data = df.copy()

    X_list = df['Facial_Features'].tolist()
    X = np.asarray(X_list)

    y = test_vector.reshape(1, -1)

    cosine_similarity = pairwise.cosine_similarity(X, y)
    cosine_similarity = np.array(cosine_similarity).flatten() # for any kind of numpy array
    data['cosine_similarity'] = cosine_similarity

    data_filter_cosine = data.query(f'cosine_similarity > {threshold}') # keep an eye out for thresholds
    data_filter_cosine.reset_index(drop=True, inplace=True)

    if len(data_filter_cosine) > 0:
        argmax = data_filter_cosine['cosine_similarity'].argmax()
        name, role = data_filter_cosine.loc[argmax][name_role]

    else:
        name = 'Unknown'
        role = 'Unknown'

    return name, role

def face_prediction(image, df, name_role=['Name', 'Role'], threshold=0.5):
    results = model_l.get(image)
    img_copy = image.copy() # good practice

    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = identification(image,
                                                  df,
                                                  name_role,
                                                  threshold)

        if person_name == 'Unknown':
            color = (0, 0, 255)  # bgr

        else:
            color = (0, 255, 0)

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color)

        rect_text = person_name
        cv2.putText(img_copy, rect_text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    return img_copy
