import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import streamlit_authenticator as stauth

# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

# import yaml
# from yaml.loader import SafeLoader
# with open('credential.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

# name, authentication_status, username = authenticator.login('Login', 'main')

# st.title ("Welcome to the portal")

# st.subheader ("Log In")
# # form1 = st.form ('form1')
# # username = form1.text_input ('Username')
# # password = form1.text_input ('Password', type = 'password')
# # submit1 = form1.form_submit_button("Login")


# # if submit1:
# #     switch_page("Mainpage")
# if authentication_status:
#     authenticator.logout('Logout', 'main')
#     # st.write(f'Welcome *{name}*')
#     # st.title('Some content')
#     switch_page("Mainpage")
# elif authentication_status is False:
#     st.error('Username/password is incorrect')
# elif authentication_status is None:
#     st.warning('Please enter your username and password')

import yaml
import streamlit as st
from yaml.loader import SafeLoader
import streamlit.components.v1 as components


_RELEASE = False

if not _RELEASE:
    # hashed_passwords = Hasher(['abc', 'def']).generate()

    # Loading config file
    with open('/Users/sinhthanhnguyen/GitHub/GitHub/Capstone_Project_2/Streamlit/credential.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Creating the authenticator object
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'], 
        config['cookie']['key'], 
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    # creating a login widget
    name, authentication_status, username = authenticator.login('Login', 'main')
    if authentication_status:
        _RELEASE = True
        # authenticator.logout('Logout', 'main')
        # st.write(f'Welcome *{name}*')
        # st.title('Some content')
        switch_page("Mainpage")
    elif authentication_status is False:
        st.error('Username/password is incorrect')
         # Creating a forgot password widget
        try:
            username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
            if username_forgot_pw:
                st.success('New password sent securely')
                # Random password to be transferred to user securely
            else:
                st.error('Username not found')
        except Exception as e:
            st.error(e)

        # Creating a forgot username widget
        try:
            username_forgot_username, email_forgot_username = authenticator.forgot_username('Forgot username')
            if username_forgot_username:
                st.success('Username sent securely')
                # Username to be transferred to user securely
            else:
                st.error('Email not found')
        except Exception as e:
            st.error(e)   
    elif authentication_status is None:
        st.warning('Please enter your username and password')

    # Creating a password reset widget
    if authentication_status:
        try:
            if authenticator.reset_password(username, 'Reset password'):
                st.success('Password modified successfully')
        except Exception as e:
            st.error(e)

    # Creating a new user registration widget
    try:
        if authenticator.register_user('Register user', preauthorization=False):
            st.success('User registered successfully')
            

    except Exception as e:
        st.error(e)

    with open('credential.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

    # # Creating an update user details widget
    # if authentication_status:
    #     try:
    #         if authenticator.update_user_details(username, 'Update user details'):
    #             st.success('Entries updated successfully')
    #     except Exception as e:
    #         st.error(e)

    # Saving config file
    
