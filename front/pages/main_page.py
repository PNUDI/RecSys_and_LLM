import streamlit as st
from front.components.recommend_main import rec_line, rec_main  # 수정된 부분
from pyparsing import empty

st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

# dummy data
rec_data_1 = 2
rec_data_8 = [0 for _ in range(4)] + [1 for _ in range(4)]


# @st.cache_data
def main():
    if st.session_state.user_id == "":
        st.switch_page("./app.py")

    empty1, _, empty2 = st.columns([0.1, 2.0, 0.1])  # empty line
    with empty1:
        empty()

    if st.session_state.rec_results is None:
        rec_results = {"main": rec_data_1, "line": rec_data_8}
        st.session_state.rec_results = rec_results
    else:
        rec_results = st.session_state.rec_results

    if rec_results is not None:
        rec_main(st.session_state.user_id, rec_results["main"])
        rec_line("추천1", rec_results["line"])
        rec_line("추천2", rec_results["line"])
        rec_line("추천3", rec_results["line"])

        _, logout_btn, _ = st.columns([0.725, 0.15, 0.725])  # logout button

        with logout_btn:
            st.header("")

            if st.button("Logout"):
                st.switch_page("./app.py")
    with empty2:
        empty()


if __name__ == "__main__":
    main()
