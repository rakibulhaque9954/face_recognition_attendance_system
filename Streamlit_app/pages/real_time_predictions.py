from home import st, fr # fecthing modules from home.py
from streamlit_webrtc import webrtc_streamer
import av
import pandas as pd
import numpy as np


st.subheader('Real-Time Attendance System')


# redis db key-name
key_name = 'academy:register'

# retrieving data from redis cloud
with st.spinner('Retrieving data from Redis Cloud'):
    redis_df = fr.redis_connect(key_name)
    if redis_df.items():
        st.success('Retrieval Successful')
    
    else:
        st.error('System Down')

st.dataframe(redis_df)

# real_time_pred = fr.face_prediction()

# Real time prediction
# streamlit-webrtc

# callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format='bgr24') # 3d np array
    
    # face detection
    predicted_img = fr.face_prediction(img, redis_df)
    
    return av.VideoFrame.from_ndarray(predicted_img, format='bgr24')

webrtc_streamer(key='realtimePrediction', video_frame_callback=video_frame_callback)