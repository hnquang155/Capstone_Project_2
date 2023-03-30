import streamlit as st
import datetime

st.sidebar.write("Select location, time frame and model")

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Model Selection: ',
    ('Sentinel-2', 'Landsat-8')
)

# Longtitude
add_slider = st.sidebar.slider(
    'Longtitude:',
    104.00000000, 106.00000000
)

# Latitude
add_slider = st.sidebar.slider(
    'Latitude:',
    9.00000000, 11.00000000
)

# Time selection
d = st.sidebar.date_input(
    "Time selection:",
    datetime.date(2022, 1, 1))
