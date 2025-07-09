# front/utils/style_utils.py

import streamlit as st


def load_styles(file_path: str = "recsys_and_llm/front/assets/styles.css"):
    """
    주어진 경로의 CSS 파일을 UTF-8로 로드하여 Streamlit에 반영합니다.
    파일이 없거나 인코딩 오류가 발생하면 에러 메시지를 표시합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as css_file:
            css = css_file.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except (UnicodeDecodeError, FileNotFoundError) as e:
        st.error(f"CSS 파일을 로드하는 중 문제가 발생했습니다: {file_path}\n{e}")
