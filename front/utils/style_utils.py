import streamlit as st

def load_styles(file_path: str = "front/assets/styles.css"):
    """
    주어진 경로의 CSS 파일을 UTF-8로 로드하여 Streamlit에 반영.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except UnicodeDecodeError:
        st.error(f"CSS 파일을 읽는 중 인코딩 문제가 발생했습니다: {file_path}")
