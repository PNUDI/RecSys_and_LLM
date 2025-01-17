import streamlit as st
import streamlit.components.v1 as components
from pyparsing import empty
from front.utils.item_utils import get_detail
from front.utils.image_utils import show_img

# st.set_page_config(initial_sidebar_state="collapsed", layout="wide")


def rec_main(user_id: str = "USER", rec_item=None):
    if not rec_item or rec_item < 0:
        st.warning("추천 결과가 없습니다.")
        return

    item_info = get_detail(rec_item)
    st.header(f"{user_id}님 지금 이 영화 어떠세요?")

    # 이미지와 링크 표시
    st.markdown(
        f"""
    <style>
        .image-container {{
            text-align: center;
            margin-top: 20px;
        }}
        .image-container img {{
            width: 400px;
            height: 600px;
            border-radius: 16px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2), 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .image-container img:hover {{
            transform: scale(1.05);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }}
        .image-container p {{
            margin-top: 10px;
            font-size: 18px;
            color: #444;
            font-weight: bold;
        }}
    </style>
    <div class="image-container">
        <a href="{item_info['item_info']}" target="_blank">
            <img src="{item_info['img_url']}" alt="{item_info['item_title']}">
        </a>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.header("")


def rec_line(head: str = "head", rec_results=None):
    if not rec_results or len(rec_results) == 0:
        st.warning("추천 결과가 없습니다.")
        return

    # 레이아웃 설정
    k = len(rec_results)
    empty1, line_text0, empty2 = st.columns([0.1, 4.0, 0.1])
    columns = st.columns([0.1] + [0.4] * k + [0.1])

    # 제목 표시
    with line_text0:
        st.subheader(head)

    # 이미지를 반복적으로 표시
    for idx, rec_item in enumerate(rec_results):
        with columns[idx + 1]:
            item_info = get_detail(rec_item)
            show_img(
                item_info["item_title"],
                item_info["img_url"],
                item_info["item_info"],
            )
