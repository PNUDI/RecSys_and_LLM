import json
import os
import sys

import requests

# 현재 스크립트의 상위 디렉토리 경로를 가져와 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import streamlit as st
from front.utils.session_utils import init_session_state
from front.utils.user_utils import signup
from pyparsing import empty

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
empty1, header_text, empty2 = st.columns([0.2, 1.2, 0.2])  # title
empty1, login_input, empty2 = st.columns([0.2, 1.2, 0.2])
empty1, login_btn, signup_btn, empty2 = st.columns([0.725, 0.08, 0.09, 0.725])
empty1, error_con, empty2 = st.columns([0.2, 1.2, 0.2])  # text input

# os.environ["FRONT_URL"] = "http://localhost:8503"
# os.environ["BACK_URL"] = "http://localhost:8000"


def main():
    init_session_state()
    error_msg = None
    print(st.session_state)

    with empty1:
        empty()
    with header_text:
        st.header("Login")
    with login_input:
        id = st.text_input("Username", label_visibility="visible", key="login_id")
        password = st.text_input(
            "Password", type="password", label_visibility="visible", key="login_pw"
        )
    with login_btn:
        if st.button("Login"):
            login_response = requests.post(
                f'{os.environ.get("BACK_URL")}/sign-in',
                json={"reviewer_id": id, "password": password},
            ).json()

            if login_response["success"]:
                st.session_state.reviewer_id = id
                st.session_state.user_name = login_response["data"]["name"]

                st.switch_page(
                    "./pages/main_page.py",
                )
            else:
                error_msg = login_response["message"]

    with signup_btn:
        if st.button("Sign up"):
            signup()

    if error_msg is not None:
        with error_con:
            st.error(error_msg)
    with empty2:
        empty()


if __name__ == "__main__":
    main()
