# Face Recognition Attendance System

This project is a mock-up of a facial recognition attendance system. It uses Redis and InsightFace models to simulate a state-of-the-art facial recognition system. This project is intended for demonstration and learning purposes only.

## Overview

The system uses InsightFace models for face detection and recognition. InsightFace is a deep learning toolkit that provides several pre-trained models for face detection, face recognition, and facial attribute analysis. The system uses these models to detect faces in real-time and recognize them based on a database of known faces.

The recognized faces are then used to mark attendance in a Redis database. Redis is an in-memory data structure store that is used for its high performance and flexibility. In this system, it is used to store the attendance records, which can be accessed and updated in real-time.

Here's a snippet from the `fr.py` file that shows how the face recognition is done:

```python
def face_prediction(image, df):
    faces = model_l.get(image)
    for face in faces:
        box, prob, landmarks = face.bbox.astype(np.int), face.prob, face.landmark.astype(np.int)
        face = np.squeeze(cv2.resize(image[box[1]:box[3], box[0]:box[2]], (112, 112)))
        emb = model.get_embedding(face).flatten()
        name = 'Unknown'
        role = 'Unknown'
        if len(df) > 0:
            sim = df['Embedding'].apply(lambda x: cosine_similarity([emb], [x])[0][0])
            idx = np.argmax(sim)
            if sim[idx] > 0.4:
                name = df.loc[idx, 'Name']
                role = df.loc[idx, 'Role']
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(image, name + ', ' + role, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image
```
This function takes an image and a dataframe as input. It performs face detection on the image using a model (model_l), and then for each detected face, it computes the bounding box coordinates. If the cosine similarity of the detected face and the faces in the dataframe is above a certain threshold, it assigns the name and role from the dataframe to the detected face. If no match is found, it assigns 'Unknown' to both the name and role.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
Before you begin, ensure you have met the following requirements:

- You have installed Python 3.8 or higher.
- You have a Windows/Linux/Mac machine.
- You have installed Redis.
- You have installed the InsightFace library.

## Installing
To install the project, follow these steps:

Clone the repository to your local machine.
Navigate to the project directory.
Install the required Python packages using pip:
`pip install -r requirements.txt`

# Behind the Scenes
In the parent directory you will find different notebooks and the backbone of this project.
- [Insightface](understanding_insightface.ipynb)
- [Face Recogntion workings](fast_face_recognition.ipynb)
- [Predictions](predictions.ipynb)
- [Redis](save_and_fetch_from_redis.ipynb)


