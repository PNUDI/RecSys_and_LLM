import os

import requests
import streamlit as st  # type: ignore
from front.components.recommender import rec_line, rec_main, search
from pyparsing import empty
from streamlit_searchbox import st_searchbox

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")


# @st.cache_data
def main():
    if "user" in st.query_params.keys():
        st.session_state.reviewer_id = st.query_params["user"]
        st.session_state.user_name = st.query_params["name"]
    else:
        st.query_params["user"] = st.session_state.reviewer_id
        st.query_params["name"] = st.session_state.user_name

    st.session_state.predictions = requests.post(
        f'{os.environ.get("BACK_URL")}/rec-load',
        json={"reviewer_id": st.session_state.reviewer_id},
    ).json()["data"]["predictions"]

    if st.session_state.reviewer_id == "":
        st.switch_page("./app.py")

    empty1, _, empty2 = st.columns([0.1, 2.0, 0.1])  # empty line

    with empty1:
        empty()

    if st.session_state.predictions is not None:
        st_searchbox(
            search,
            placeholder="Search ... ",
            key="my_key",
        )

        rec_main(st.session_state.predictions["prediction-1"])
        rec_line(
            f"Recommendations for {st.session_state.user_name}",
            st.session_state.predictions["prediction-2"],
        )
        rec_line(
            f"Similar movies to what you've watched recently",
            st.session_state.predictions["prediction-3"],
        )
        rec_line(
            f"Movies in the {st.session_state.predictions['prediction-4']['genre']} genre that {st.session_state.user_name} likes",
            st.session_state.predictions["prediction-4"]["movies"],
        )

        _, logout_btn, _ = st.columns([0.725, 0.15, 0.725])

        with logout_btn:
            st.header("")

            if st.button("Logout"):
                st.switch_page("./app.py")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
