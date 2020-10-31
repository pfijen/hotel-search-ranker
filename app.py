# %%
import streamlit as st
import pandas as pd



def load_data(nrows):
    data = pd.read_csv('data/stored/test_sample.csv', nrows=nrows)
    return data

st.title('Hotel search ranker webapp')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')
    
    



# %%
