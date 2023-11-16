import numpy as np
import pandas as pd
import cv2
import datetime
import redis
import os

# insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise  # for cosine similarity calculation

# connect to redis cloud
host = 'redis-12830.c302.asia-northeast1-1.gce.cloud.redislabs.com'
port_no = 12830
password = '0k137l1gcAW3TQMjUeQ06oc7vQTMKsAq'

r = redis.StrictRedis(host=host, port=port_no, password=password)

def redis_connect(keyname):
    key_name = keyname
    redis_hash = r.hgetall(key_name)
    # print(redis_hash)
    redis_series = pd.Series(redis_hash)

    # Convert bytes to array
    redis_series = redis_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = redis_series.index
    index = list(map(lambda x: x.decode(), index)) # converting keys to strings

    redis_series.index = index

    redis_df = redis_series.to_frame().reset_index() # converting to df and resetting index
    redis_df.columns = ['name_role', 'facial_features']
    redis_df[['Name' , 'Role']] = redis_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series) # splitting name and role to different columns
    
    return redis_df


# buffalo_l model face analysis configuration
model_l = FaceAnalysis(name='buffalo_l',
                        root='/Users/boss/Desktop/Notes/4_attendance_app/insightface_model/models/buffalo_l', 
                        providers=['CPUExecutionProvider']) # CUDAExecutionProvider incase GPU

model_l.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)  # input size

# using cosine similarity function to compare vectors and identify the person
def identification(df, test_vector, name_role=['Name', 'Role'], threshold=0.5):
    data = df.copy()

    X_list = df['facial_features'].tolist()
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

### Real-time face recognition
# save logs every minute

class RealTimeFaceRecognition:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])
        
    def reset_logs(self):
        self.logs = dict(name=[], role=[], current_time=[])
        
    def save_logs_redis(self):
        # create a log df
        df = pd.DataFrame(self.logs)
        
        # drop duplicates
        df.drop_duplicates('name', inplace=True)
        
        # push to redis
        # concat name, role and current_time
        name_list = df['name'].tolist()
        role_list = df['role'].tolist()
        current_time_list = df['current_time'].tolist()
        encoded_list = []
        
        for name, role, current_time in zip(name_list, role_list, current_time_list):
            if name != 'Unknown':
                concat_string = f'{name}@{role}@{current_time}'
                encoded_list.append(concat_string)
            
        if len(encoded_list) > 0:
            r.lpush('attendance:logs', *encoded_list)
            
        self.reset_logs()

    def face_prediction(self, image, df, name_role=['Name', 'Role'], threshold=0.5):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")  # get current time
        results = model_l.get(image)
        img_copy = image.copy() # good practice
        

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = identification(df=df,
                                                    test_vector=embeddings,
                                                    name_role=name_role,
                                                    threshold=threshold)

            if person_name == 'Unknown':
                color = (0, 0, 255)  # bgr

            else:
                color = (0, 255, 0)

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color)

            rect_text = person_name
            cv2.putText(img_copy, rect_text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(img_copy, str(current_time), (x2+10, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
            
            # save logs
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
            
        return img_copy


#### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
        
    def get_embedding(self,frame):
        # get results from insightface model
        results = model_l.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            # put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            # facial features
            embeddings = res['embedding']
            
        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):
        # validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        # if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        
        
        # step-1: load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32) # flatten array            
        
        # step-2: convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)       
        
        # step-3: cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        
        # step-4: save this into redis database
        # redis hashes
        r.hset(name='academy:register',key=key,value=x_mean_bytes)
        
        # 
        os.remove('face_embedding.txt')
        self.reset()
        
        return True