from home import st, fr # fecthing modules from home.py
from streamlit_webrtc import webrtc_streamer
import av
import time

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

# set time
wait_time = 10 # seconds
set_time = time.time()

real_time_fr = fr.RealTimeFaceRecognition()

# Real time prediction
# streamlit-webrtc

# callback function
def video_frame_callback(frame):
    global set_time

    img = frame.to_ndarray(format='bgr24') # 3d np array
    
    # face detection
    predicted_img = real_time_fr.face_prediction(img, redis_df)
    
    time_now = time.time()
    difference = time_now - set_time
    
    if difference >= wait_time:
        real_time_fr.save_logs_redis()
        set_time = time.time() # reset time
        
        # st.success('Attendance Saved')
    
    return av.VideoFrame.from_ndarray(predicted_img, format='bgr24')

webrtc_streamer(key='realtimePrediction', video_frame_callback=video_frame_callback)