# app/utils/session_utils.py
import streamlit as st


def check_login():
    """
    세션에서 로그인 여부 확인
    """

    if "reviewer_id" not in st.session_state or st.session_state.reviewer_id == "":
        st.warning("로그인 상태가 아닙니다. 로그인 페이지로 이동합니다.")
        st.switch_page("app.py")
        return False
    return True


def initialize_conversations():
    """
    세션에서 대화(conversations) 초기화
    """
    if "conversations" not in st.session_state:
        st.session_state.conversations = []


def initialize_saved_conversations():
    """
    세션에서 저장된 대화(saved_conversations) 초기화
    """
    if "saved_conversations" not in st.session_state:
        st.session_state.saved_conversations = {}


def initialize_pipeline(default_pipeline):
    """
    세션에서 파이프라인(pipeline) 초기화
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = default_pipeline


def initialize_feedback_submitted():
    """
    세션에서 메시지 피드백(feedback_submitted) 상태 초기화
    """
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = {}


def init_session_state():
    """
    Streamlit 세션 상태 초기화
    """
    if "reviewer_id" not in st.session_state:
        st.session_state.reviewer_id = ""
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "selected" not in st.session_state:
        st.session_state.selected = None
