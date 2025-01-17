# app/components/conversation_manager.py
import streamlit as st

def save_conversation(title):
    """
    대화를 저장하는 함수
    """
    if title.strip():
        st.session_state.saved_conversations[title] = (
            st.session_state.conversations.copy()
        )
        return f"'{title}' 대화를 저장했습니다!"
    return "세션 제목이 비어있어 저장을 취소했습니다."

def load_conversation(title):
    """
    저장된 대화 불러오기
    """
    st.session_state.conversations = st.session_state.saved_conversations[title].copy()
    return f"'{title}' 대화를 불러왔습니다."
