from home import st 
from home import fr
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np




st.set_page_config(page_title='Registration', layout='wide')
st.subheader('Registration Form')


## init registration form
registration_form = fr.RegistrationForm()

# Step-1: Collect person name and role
person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select your Role',options=('Student',
                                                      'Teacher'))


# step-2: Collect facial embedding of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') # 3d array bgr
    reg_img, embedding = registration_form.get_embedding(img)
    # two step process
    # 1st step save data into local computer txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func)


# step-3: save the data in redis database
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f"{person_name} registered sucessfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')
        
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again.')
        
