import os

import requests
import streamlit as st  # type: ignore
from front.components.recommender import rec_line
from front.utils.item_utils import show_info
from pyparsing import empty

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")


def main():
    print("########### item page in ###########")
    st.session_state.reviewer_id = st.query_params["user"]
    st.session_state.selected = st.query_params["item"]
    st.session_state.user_name = st.query_params["name"]
    st.session_state.predictions = requests.post(
        f'{os.environ.get("BACK_URL")}/detail',
        json={
            "reviewer_id": st.session_state.reviewer_id,
            "item_id": st.session_state.selected,
        },
    ).json()["data"]["predictions"]

    if st.session_state.reviewer_id == "":
        st.switch_page("./app.py")

    empty1, con, empty2 = st.columns([0.05, 0.9, 0.05])
    with empty1:
        empty()

    if st.session_state.selected is not None:
        with con:
            item_title = show_info(st.session_state.selected)

            st.header("")
            rec_line(
                f"Movies similar to {item_title}",
                st.session_state.predictions,
            )

            _, logout_btn, _ = st.columns([0.775, 0.2, 0.775])

            with logout_btn:
                st.header("")

                if st.button("Logout"):
                    st.switch_page("./app.py")

    with empty2:
        empty()


if __name__ == "__main__":
    main()
