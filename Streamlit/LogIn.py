import streamlit as st
from streamlit_extras.switch_page_button import switch_page
st.title ("Welcome to the portal")

st.subheader ("Log In")
form1 = st.form ('form1')
username = form1.text_input ('Username')
password = form1.text_input ('Password', type = 'password')

submit1 = form1.form_submit_button("Login")
if submit1:
    switch_page("Mainpage")



