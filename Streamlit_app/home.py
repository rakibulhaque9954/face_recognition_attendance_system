import streamlit as st


st.set_page_config(page_title='Attendance System', layout='wide')
st.header('Attendance System using Facial Recognition')

with st.spinner('Loading Models and Connecting to Redis Cloud...'):
    import fr
   
st.success('Models Loaded Successfully')
st.success('Redis Cloud Configured')