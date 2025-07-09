import html
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from pytz import timezone


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)

    return file_paths


def get_missing(title_id_dict):
    exist_items = [k for k, v in title_id_dict.items() if v != "No Title"]

    return np.setdiff1d(
        np.arange(1, max(exist_items) + 1, dtype=np.int32),
        exist_items,
        assume_unique=True,
    ).tolist()


def find_cold(user_collection, max_len):
    data = {}

    # db data -> dict
    for user_data in user_collection.find():
        user_id = str(user_data["_id"])  # _id를 문자열로 변환
        itemnums = [
            item["itemnum"] for item in user_data.get("items", [])
        ]  # itemnum 리스트 추출

        # 최신 max_len 개수만 유지 (패딩 포함)
        seq = [0] * max_len
        recent_items = itemnums[
            -max_len:
        ]  # 마지막 아이템을 제외한 최신 max_len개만 유지
        seq[-len(recent_items) :] = recent_items  # 패딩 포함하여 우측 정렬

        data[user_id] = seq  # 결과 저장

    item_interactions = []
    for user, items in data.items():
        item_interactions.extend(items)

    # Step 2: Count occurrences of each item
    item_counts = Counter(item_interactions)

    # Step 3: Sort items by interaction count
    sorted_items = sorted(
        item_counts.items(), key=lambda x: x[1], reverse=True
    )  # (item, count)

    # Step 4: Calculate thresholds for warm and cold items
    total_items = len(sorted_items)
    cold_threshold = int(total_items * 0.35)  # Bottom 35%

    # Get cold items (bottom 35% of interactions)
    cold_items = [item for item, count in sorted_items[-cold_threshold:]]

    return cold_items


def get_text_name_dict(item_collection):
    text_name_dict = {"title": {}, "description": {}}

    for item in item_collection.find():
        item_id = int(item["_id"])  # _id를 문자열로 변환
        title = item.get("title", "No Title")  # 기본값 설정
        description = item.get("description", ["No Description"])  # 기본값 설정

        # 리스트 타입이면 문자열로 변환
        if isinstance(description, list):
            description = " ".join(description) if description else "No Description"

        text_name_dict["title"][item_id] = title
        text_name_dict["description"][item_id] = description

    return text_name_dict


def seq_preprocess(maxlen, data):
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for i in reversed(data):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break

    return seq


def clean_text(text):
    text = str(text)  # 문자열 변환
    text = html.unescape(text)  # 🔹 HTML 엔티티 변환 (&amp; → & 등)

    # 🔹 특수 공백 및 제어 문자 제거 (유니코드 포함)
    text = text.replace("\xa0", " ").replace("\t", " ").replace("\n", " ")

    # 🔹 앞뒤 공백 및 큰따옴표 제거
    text = text.strip().strip('"').strip()

    # 🔹 중복 공백 제거 (모든 종류의 공백 포함)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text


def calculate_genre_distribution(item_collection, all_genres):
    """
    전체 영화 데이터에서 장르 분포를 계산하여 전역 장르 빈도수 반환
    """
    genre_counts = Counter()
    total_genre_mappings = 0

    for movie in item_collection.find({}, {"predicted_genre": 1}):
        if "predicted_genre" in movie:
            genre_counts.update(movie["predicted_genre"])
            total_genre_mappings += len(movie["predicted_genre"])

    # 장르별 분포 확률 계산 (P(genre))
    genre_distribution = {
        genre: genre_counts[genre] / total_genre_mappings for genre in all_genres
    }
    return genre_distribution
