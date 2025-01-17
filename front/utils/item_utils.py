# utils/item_utils.py
import streamlit as st
from .image_utils import show_img

# 아이템 정보 맵 (데이터)
item_info_map = {
    0: {
        "item_title": "파묘",
        "img_url": "https://image.tmdb.org/t/p/original/ca5UAt0YJAkrcPlcOUftXnEa6C5.jpg",
        "item_info": "https://pedia.watcha.com/ko-KR/contents/m53mNG2",
        "people": ", ".join(["김고은", "최민식", "유해진", "이도현"]),
        "release": "2024.12.26",
        "avg_score": 4.8,
    },
    1: {
        "item_title": "짱구",
        "img_url": "https://i.namu.wiki/i/lT70-Fgm_Iw2bPlvJmpKy1QRqEeZiXvsq3Ckw2nMs2eYYmuesH68oWP4gFsAYEFIQZ8zuCXc8Yd0j4f0tc7TZg.webp",
        "item_info": "https://pedia.watcha.com/ko-KR/contents/mdj2e4q",
        "people": ", ".join(["짱구", "흰둥이", "맹구", "철수"]),
        "release": "2024.12.27",
        "avg_score": 3.0,
    },
    2: {
        "item_title": "친절한 금자씨",
        "img_url": "https://web-cf-image.cjenm.com/crop/660x950/public/share/metamng/programs/sympathyforladyvengeance-movie-poster-ko-001-29.jpg?v=1677040153",
        "item_info": "https://watcha.com/contents/mdRL9gW?search_id=8cc236f3-61e8-43bd-9288-3415cfbc6591",
        "people": ", ".join(["이영애", "최민식", "라미란", "김병옥"]),
        "release": "2025.01.13",
        "avg_score": 4.9,
    },
}


def show_info(item_info):
    """
    아이템 정보를 화면에 표시
    """
    show_img(item_info["item_title"], item_info["img_url"], item_info["item_info"])
    _, temp, _ = st.columns([0.3, 0.5, 0.3])
    with temp:
        st.markdown(f"출연진 : {item_info['people']}")
        st.markdown(f"개봉일 : {item_info['release']}")
        st.markdown(f"평균 평점 : {item_info['avg_score']}")
        st.markdown(f"[더보기]({item_info['item_info']})")


def get_detail(item_id: int = 1):
    """
    ID를 통해 특정 아이템 정보를 반환
    """
    return item_info_map.get(item_id)
