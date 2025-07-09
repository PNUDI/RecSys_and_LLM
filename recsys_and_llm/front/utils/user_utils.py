import os
import time

import requests
import streamlit as st


@st.dialog("Sign up")
def signup():
    user_name = st.text_input("Name", label_visibility="visible", key="signup_name")
    reviewer_id = st.text_input("Id", label_visibility="visible", key="signup_id")
    password = st.text_input(
        "Password", type="password", label_visibility="visible", key="signup_pw"
    )

    if st.button("Submit"):
        response = requests.post(
            f'{os.environ.get("BACK_URL")}/sign-up',
            json={"reviewer_id": reviewer_id, "password": password, "name": user_name},
        ).json()

        if response["success"]:
            st.text("Your registration is complete!")
            time.sleep(2)
            st.session_state.reviewer_id = reviewer_id
            st.session_state.user_name = user_name
            st.switch_page(
                "./pages/main_page.py",
            )
        else:
            st.error(response["message"])
