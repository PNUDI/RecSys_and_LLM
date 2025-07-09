# pages/chat_page.py

from datetime import datetime

import streamlit as st  # type: ignore
from front.components.conversation_manager import (
    load_conversation,
    retrieve_all_conversations,
    save_conversation,
)
from front.components.pipeline_manager import set_pipeline
from front.components.response_generator import get_unicrs_response
from front.utils.session_utils import (
    check_login,
    initialize_conversations,
    initialize_feedback_submitted,
    initialize_pipeline,
    initialize_saved_conversations,
)
from front.utils.style_utils import load_styles


def main():
    # 사이드바를 기본으로 열도록 변경 (layout은 wide)
    st.set_page_config(initial_sidebar_state="expanded", layout="wide")

    # 0. 로그인 여부 확인
    if not check_login():
        return

    # 세션 초기화
    initialize_conversations()
    initialize_saved_conversations()
    # 파이프라인 설정 (모델 선택 전 초기값)
    initialize_pipeline(set_pipeline("", "UniCRS_REC", "UniCRS_GEN"))
    initialize_feedback_submitted()

    if "dialog" not in st.session_state:
        st.session_state.dialog = []

    reviewer_id = st.session_state.reviewer_id

    # ----------------------------------------------------------------
    # (A) CSS 스타일 로드: 기본 스타일 외에 사용자 메시지 정렬을 반전
    # ----------------------------------------------------------------
    load_styles()

    # ----------------------------------------------------------------
    # (B) 사이드바 - 대화 관리 (불러오기 & 저장하기)
    # ----------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 대화 관리")
        st.markdown("---")
        st.markdown("### 저장된 대화 불러오기 📂")
        saved_conversations = retrieve_all_conversations(reviewer_id)

        if not saved_conversations:
            st.write("저장된 대화 세션이 없습니다.")
        else:
            conversation_dict = {
                f"{conv['conversation_title']} ({conv['conversation_id']})": conv[
                    "conversation_id"
                ]
                for conv in saved_conversations
            }
            chosen_session = st.selectbox(
                "세션 선택", list(conversation_dict.keys()), key="load_session_select"
            )

            if st.button("불러오기", key="load_convo_button"):
                selected_conversation_id = conversation_dict[chosen_session]
                st.info(load_conversation(reviewer_id, selected_conversation_id))

        st.markdown("---")

        # 대화 저장하기
        st.markdown("### 대화 저장하기 💾")
        with st.form("save_form", clear_on_submit=True):
            title = st.text_input("대화 제목 입력")
            submitted = st.form_submit_button("저장하기")
            if submitted:
                st.info(save_conversation(title, reviewer_id))

    # ----------------------------------------------------------------
    # (C) 메인 영역: 모델 선택 및 채팅 인터페이스
    # ----------------------------------------------------------------
    st.markdown("## UniCRS 대화형 추천 테스트")

    # (C1) 모델 선택 (상단)
    model_options = {
        "UniCRS (기본)": "default",
        "FillBlank": "blank",
        "Expansion": "expansion",
        "GPT": "gpt",
    }
    selected_model_label = st.selectbox(
        label="모델 선택", options=list(model_options.keys()), index=0
    )
    chosen_flag = model_options[selected_model_label]
    st.session_state.pipeline = set_pipeline(chosen_flag)

    st.write("---")

    # (C2) 채팅 진행 (입력)
    user_message = st.chat_input("원하는 영화를 찾지 못했나요? UniCRS와 대화해보세요!")
    if user_message:

        # Show loading indicator while processing
        with st.spinner("처리 중입니다..."):
            response = get_unicrs_response(
                user_message, st.session_state.dialog, st.session_state.pipeline
            )
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conversations.append(
            {"role": "user", "content": user_message, "date_time": current_time_str}
        )
        st.session_state.conversations.append(
            {
                "role": "assistant",
                "content": response,
                "date_time": current_time_str,
                "feedback": "None",
            }
        )

    # ----------------------------------------------------------------
    # (D) 채팅 기록 표시 (메시지 출력 및 피드백)
    # ----------------------------------------------------------------
    for idx, msg in enumerate(st.session_state.conversations):
        if msg["role"] == "assistant":
            # 시스템 메시지: 왼쪽에 아바타 (기본)
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["content"])
                # 다양한 피드백 이모티콘 (한 번 제출 시 모두 숨김)
                if not st.session_state.feedback_submitted.get(idx, False):
                    feedback_options = [
                        ("👍 good recommendation", "good recommendation"),
                        ("🌟 interesting", "interesting"),
                        ("🤝 realistic", "realistic"),
                        ("👎 bad recommendation", "bad recommendation"),
                        ("😴 boring", "boring"),
                        ("🤖 unnatural", "unnatural"),
                    ]
                    feedback_cols = st.columns(len(feedback_options), gap="small")
                    for col, (icon, feedback_type) in zip(
                        feedback_cols, feedback_options
                    ):
                        with col:
                            if st.button(
                                icon,
                                key=f"feedback_{feedback_type}_{idx}",
                                use_container_width=True,
                            ):
                                st.toast(f"'{feedback_type}' 피드백을 기록했습니다.")
                                st.session_state.feedback_submitted[idx] = True
                                # 피드백 기록 (default: None)
                                st.session_state.conversations[idx][
                                    "feedback"
                                ] = feedback_type

        else:
            # 사용자 메시지: 오른쪽에 아바타 (CSS로 정렬 반전)
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])


if __name__ == "__main__":
    main()
