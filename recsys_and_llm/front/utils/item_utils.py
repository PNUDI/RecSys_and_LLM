# utils/item_utils.py
import os
from typing import AnyStr, Dict, List

import requests
import streamlit as st
from front.components.recommender import rec_line
from streamlit_star_rating import st_star_rating

from .image_utils import show_img


def get_review(item_id: str, rating: str):
    _, _, input_con, btn_con, _ = st.columns([0.32, 0.03, 0.5, 0.05, 0.1])

    with input_con:
        review_input = st.text_input(
            "review",
            label_visibility="collapsed",
            key="review_input",
            placeholder="How was it?",
        )

    with btn_con:
        if st.button("apply") or review_input:
            response = requests.post(
                f'{os.environ.get("BACK_URL")}/review',
                json={
                    "reviewer_id": st.session_state.reviewer_id,
                    "item_id": item_id,
                    "review": review_input,
                    "rating": rating,
                },
            ).json()
            print(response)
            if response["success"]:
                st.rerun()
            else:
                st.error("reveiw Fail")


def get_rating(item_id: str, review: str):
    stars = st_star_rating("", key="rating_input", maxValue=5, defaultValue=0, size=20)
    if stars > 0:
        response = requests.post(
            f'{os.environ.get("BACK_URL")}/review',
            json={
                "reviewer_id": st.session_state.reviewer_id,
                "item_id": item_id,
                "review": review,
                "rating": str(stars),
            },
        ).json()
        print(response)
        if response["success"]:
            st.rerun()
        else:
            st.error("rating Fail")


def show_reviews(reviews: List[dict[AnyStr, AnyStr]], item_id):
    for i, r in enumerate(reviews):
        st.markdown(f":blue-background[**{r['user_name']}**]", unsafe_allow_html=True)
        st_star_rating(
            label="",
            key=f"rating_in_container_{r['user_name']}_{item_id}_{i}",
            maxValue=5,
            defaultValue=r["rating"],
            read_only=True,
            size=15,
        )
        st.markdown(f"{r['review']}", unsafe_allow_html=True)


def show_info(item_id: str):
    """
    아이템 정보를 화면에 표시
    """

    response = requests.post(
        f'{os.environ.get("BACK_URL")}/detail',
        json={"reviewer_id": st.session_state.reviewer_id, "item_id": item_id},
    ).json()
    if not response["success"]:
        st.text("Something Wrong!!!")
        return

    item_info = response["data"]
    img_con, _, info_con, _ = st.columns([0.32, 0.03, 0.55, 0.1])

    with img_con:
        st.text("")
        show_img(item_id, item_info["img"], clickable=False)

    with info_con:
        st.header(item_info["title"])
        st.markdown(
            f"**CAST** : {', '.join(item_info['cast'])}" if item_info["cast"] else ""
        )
        with st.container(height=100, border=False):
            st.markdown(f"**Description** : {item_info['description']}")

        st.subheader("Comment")
        container_height = (
            350
            - (0 if item_info["reviews"]["my_rating"] is None else 25)
            - (0 if item_info["reviews"]["my_review"] is None else 25)
        )
        with st.container(height=container_height, border=False):
            show_reviews(item_info["reviews"]["others"], item_id)

        st.markdown('<hr style="border: 1px solid #ececec;">', unsafe_allow_html=True)

    with info_con:
        if item_info["reviews"]["my_rating"] == "":
            get_rating(item_id, item_info["reviews"]["my_review"])
        else:
            st.markdown(f":grey-background[**My Rating**]", unsafe_allow_html=True)
            st_star_rating(
                "",
                key="rating_show",
                maxValue=5,
                defaultValue=item_info["reviews"]["my_rating"],
                size=15,
                read_only=True,
            )

    if item_info["reviews"]["my_review"] == "":
        get_review(item_id, item_info["reviews"]["my_rating"])
    else:
        with info_con:
            st.markdown(f":grey-background[**My Comment**]", unsafe_allow_html=True)
            st.markdown(f'{item_info["reviews"]["my_review"]}', unsafe_allow_html=True)

    return item_info["title"]
