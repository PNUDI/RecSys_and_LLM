# utils/image_utils.py
import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


@st.cache_data
def show_img(
    item_id: str,
    img_url: str,
    item_title: str = "No Title",
    clickable: bool = True,
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

    if not clickable:
        st.markdown(
            f"<img src='data:image/png;base64,{img_str}' alt='image' class='image' style='border-radius: 15px;'>",
            unsafe_allow_html=True,
        )
        return

    html = f"""
        <style>
            .show-image-container {{
                position: relative;
                display: inline-block;
                width: 100%;
                max-width: 500px;
                border-radius: 15px;
                overflow: hidden; 
            }}

            .show-image {{ 
                display: block;
                width: 100%; 
                height: auto;
                border-radius: 15px;
                object-fit: cover;
            }}

            .show-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                opacity: 0;
                transition: opacity 0.3s ease;
                display: flex;
                justify-content: center;
                align-items: center;
                border-radius: 15px; 
            }}

            .show-overlay .title {{
                color: white;
                font-size: 18px;
                text-align: center;
            }}

            .show-image-container:hover .show-overlay {{
                opacity: 1;
            }}
        </style>
        
        """

    html += f"""<a href='{os.environ.get('FRONT_URL')}/item_page/?user={st.session_state.reviewer_id}&name={st.session_state.user_name}&item={item_id}' target = '_self' class='image-link'>
            <div class='show-image-container'>
                <img src='data:image/png;base64,{img_str}' alt='image' class='show-image'>
                <div class='show-overlay'>
                    <div class='title'>{item_title}</div>
                </div>
            </div>
        </a>
    """

    st.markdown(html, unsafe_allow_html=True)
