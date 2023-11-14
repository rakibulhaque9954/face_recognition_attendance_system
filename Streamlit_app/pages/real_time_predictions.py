from home import st, fr # fecthing modules from home.py
import pandas as pd
import numpy as np






st.set_page_config(page_title='Predicitons', layout='wide')
st.subheader('Real-Time Attendance System')


# redis db key-name
key_name = 'academy:register'

# retrieving data from redis cloud
with st.spinner('Retrieving data from Redis Cloud'):
    redis_df = fr.redis_connect(key_name)
    if redis_df.items():
        st.success('Retrieval Successful')
    
    else:
        st.success('System Down')

st.dataframe(redis_df)





# Real time prediction