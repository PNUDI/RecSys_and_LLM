# pages/chat_page.py

import streamlit as st

########################################
# (예시) set_pipeline, 더미 모듈/객체 설정
########################################
unicrs_rec = "UniCRS_REC"  # Dummy
unicrs_gen = "UniCRS_GEN"  # Dummy
gpt_gen = "GPT_GEN"  # Dummy


def set_pipeline(flag, rec_module, gen_module):
    """flag, rec_module, gen_module에 따라 적절한 Pipeline 객체를 생성 (더미)."""
    if flag.lower() == "blank":
        pipeline = "FillBlankPipeline"
    elif flag.lower() == "expansion":
        pipeline = "ExpansionPipeline"
    elif flag.lower() == "gpt":
        pipeline = "GptPipeline"
    else:
        pipeline = "DefaultUniCRS"
    return pipeline


############################################
# 1) UniCRS/GPT 등으로 응답을 만드는 로직
############################################
def get_unicrs_response(user_message: str, pipeline) -> str:
    """
    파이프라인을 통해 더미 응답을 생성하는 함수.
    (실제로는 pipeline을 호출해 결과를 반환)
    """
    return f"안녕하세요! 요청하신 내용에 대한 추천을 드릴게요. (파이프라인: {pipeline})"


########################
# 2) 메인 UI (페이지)  #
########################
def main():
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

    # 0. 로그인 여부 확인
    if "user_id" not in st.session_state or st.session_state.user_id == "":
        st.warning("로그인 상태가 아닙니다. 로그인 페이지로 이동합니다.")
        st.switch_page("app.py")
        return

    # 1. 세션 초기화
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "saved_conversations" not in st.session_state:
        # 예: {"세션제목": [ {role:..., content:...}, ... ], ...}
        st.session_state.saved_conversations = {}
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = set_pipeline("", unicrs_rec, unicrs_gen)
    if "feedback_submitted" not in st.session_state:
        # 각 메시지 인덱스별로 좋아요/싫어요가 이미 눌렸는지 기록
        st.session_state.feedback_submitted = {}

    # -------------------------------------------------------
    # (A) 상단 - "모델 선택"을 가장 먼저(최상단) 배치
    # -------------------------------------------------------
    st.markdown("## UniCRS 대화형 추천 테스트")  # 혹은 st.header("모델 선택")
    model_options = {
        "UniCRS (기본)": "",
        "FillBlank": "blank",
        "Expansion": "expansion",
        "GPT": "gpt",
    }
    selected_model_label = st.selectbox(
        label="", options=list(model_options.keys()), index=0
    )
    chosen_flag = model_options[selected_model_label]
    st.session_state.pipeline = set_pipeline(
        chosen_flag, unicrs_rec, gpt_gen if chosen_flag == "gpt" else unicrs_gen
    )

    # -------------------------------------------------------
    # 상단 버튼들 (오른쪽 정렬)
    # -------------------------------------------------------
    st.markdown(
        """
        <style>
        .top-right-buttons {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 10px;
            margin-top: -3rem; /* 모델 선택과 약간 겹치지 않게 조정 */
            margin-bottom: 1rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="top-right-buttons">', unsafe_allow_html=True)

    # (A1) 메인 페이지 버튼
    # if st.button("메인 페이지로 돌아가기"):
    #     st.switch_page("pages/main_page.py")

    # (A2) "저장된 대화 세션 불러오기"
    st.markdown("#### 저장된 대화 불러오기")
    saved_titles = list(st.session_state.saved_conversations.keys())
    if not saved_titles:
        st.write("저장된 대화 세션이 없습니다.")
    else:
        chosen_session = st.selectbox(
            "불러올 세션", saved_titles, key="load_session_select"
        )
        if st.button("불러오기", key="load_convo_button"):
            st.session_state.conversations = st.session_state.saved_conversations[
                chosen_session
            ].copy()
            st.success(f"'{chosen_session}' 대화를 불러왔습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # (B) "대화 저장하기" (Form) - 대화가 하나 이상일 때
    # -------------------------------------------------------
    if len(st.session_state.conversations) > 0:
        with st.form("save_form", clear_on_submit=True):
            st.write("### 대화 저장하기")
            new_title = st.text_input("대화 세션 제목을 입력하세요.")
            submitted = st.form_submit_button("저장하기")
            if submitted:
                if new_title.strip():
                    st.session_state.saved_conversations[new_title] = (
                        st.session_state.conversations.copy()
                    )
                    st.success(f"'{new_title}' 대화를 저장했습니다!")
                else:
                    st.info("세션 제목이 비어있어 저장을 취소했습니다.")

    st.write("---")

    # -------------------------------------------------------
    # (C) 채팅 진행
    # -------------------------------------------------------
    user_message = st.chat_input("원하는 영화를 찾지 못했나요? UniCRS와 대화해보세요!")
    if user_message:
        # 사용자 메시지
        st.session_state.conversations.append({"role": "user", "content": user_message})
        # 모델 응답
        response = get_unicrs_response(user_message, st.session_state.pipeline)
        st.session_state.conversations.append(
            {"role": "assistant", "content": response}
        )

    # -------------------------------------------------------
    # (D) 대화 표시
    #   - Like/Dislike는 한 번 누르면 둘 다 안 보이게
    # -------------------------------------------------------
    for idx, msg in enumerate(st.session_state.conversations):
        if msg["role"] == "user":
            # (기본) 왼쪽에 표시
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            # 시스템(assistant)은 오른쪽
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["content"])
                # 좋아요/싫어요 이미 제출됐는지 여부
                if not st.session_state.feedback_submitted.get(idx, False):
                    like_col, unlike_col = st.columns([0.12, 0.12], gap="small")
                    with like_col:
                        if st.button("👍", key=f"like_{idx}"):
                            st.toast("좋아요를 서버에 기록했습니다.")
                            st.session_state.feedback_submitted[idx] = True
                    with unlike_col:
                        if st.button("👎", key=f"unlike_{idx}"):
                            st.toast("별로라는 의견을 서버에 기록했습니다.")
                            st.session_state.feedback_submitted[idx] = True


if __name__ == "__main__":
    main()
