import os
import sys

# 현재 스크립트의 상위 디렉토리 경로를 가져와 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import streamlit as st
from front.utils.session_utils import init_session_state
from pyparsing import empty

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
empty1, con1, empty2 = st.columns([0.2, 1.2, 0.2])  # title
empty1, con2, empty2 = st.columns([0.2, 1.2, 0.2])  # text input
empty1, con3, empty2 = st.columns([0.2, 1.2, 0.2])  # text input
empty1, con4, empty2 = st.columns([0.2, 1.2, 0.2])  # text input

sample_user = {"id": "USER", "password": "123"}  # dummy data


def main():
    init_session_state()
    print(st.session_state)

    with empty1:
        empty()
    with con1:
        st.header("Login")
    with con2:
        st.session_state.user_id = st.text_input("Username", label_visibility="visible")
        password = st.text_input(
            "Password", type="password", label_visibility="visible"
        )
    with con3:
        if st.button("Login"):
            if (
                st.session_state.user_id == sample_user["id"]
                and password == sample_user["password"]
            ):
                st.switch_page("./pages/main_page.py")
            else:
                st.error("Username or Password is incorrect. Try Again!")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
