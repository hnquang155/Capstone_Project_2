import streamlit as st
form3 = st.form('form3')
form3.markdown('All * fields are mandatory')
username = form3.text_input ('User Name *')
email = form3.text_input ('Email *')
feedback = form3.text_area ('Feedback *')
rating = form3.slider ('Rating', min_value = 0, max_value = 5)
file_upload = form3.file_uploader ('Upload file')
submit3 = form3.form_submit_button("Submit feedback")
if submit3:
    if len(username) == 0 or len (email) ==0 or len (feedback) == 0:
        st.warning ('Please review the review form. The * parts are mandatory')