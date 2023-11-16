from home import st 
from home import fr


st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reports')

# retrieving logs data from redis cloud
# extract data from redis
name = 'attendance:logs'
key_name = 'academy:register'

def load_logs(name, end=-1):
    logs = fr.r.lrange(name, start=0, end=end) # retrieve all logs
    return logs

# create tab to show info
tab1, tab2 = st.tabs(['Logs', 'Data'])
 
with tab1:
    if st.button('Refresh Logs'):
        for i in range(len(load_logs(name))):
            st.write(i + 1, load_logs(name)[i].decode('utf-8'))

with tab2:
    if st.button('Refresh Data'):
        # retrieving data from redis cloud
        with st.spinner('Retrieving data from Redis Cloud'):
            redis_df = fr.redis_connect(key_name)
            st.dataframe(redis_df[['Name', 'Role']])