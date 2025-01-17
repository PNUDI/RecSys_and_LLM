# utils/image_utils.py
import base64
from io import BytesIO

import requests
from PIL import Image
import streamlit as st


def show_img(
    item_title: str = "title",
    img_url: str = "https://ghcci.korcham.net/images/no-image01.gif",
    info_url: str = None,
):
    """
    이미지 URL을 통해 이미지를 로드하고 Streamlit에 표시
    """
    image = Image.open(BytesIO(requests.get(img_url, stream=True).content)).resize(
        (600, 900)
    )
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    html = (
        f"<a href='{info_url}'><img src='data:image/png;base64,{img_str}'></a>"
    )
    st.markdown(html, unsafe_allow_html=True)
