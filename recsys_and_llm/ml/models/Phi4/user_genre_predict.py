import random
from collections import Counter

import torch
from pymongo import MongoClient
from transformers import pipeline

from recsys_and_llm.backend.app.config import ALL_GENRES, DB_NAME, MONGO_URI
from recsys_and_llm.ml.utils import calculate_genre_distribution


def predict_user_preferred_genres(
    model,
    candidate_genres,
    sorted_user_genre_counts,
    sorted_watched_genres_by_rarity,
    k=3,
    model_name="unsloth/phi-4-unsloth-bnb-4bit",
):

    prompt = [
        {
            "role": "system",
            "content": f"You are an AI movie genre predictor. Your task is to determine the top {k} genres a user is most likely to prefer based on their past viewing history and genre rarity. Follow these rules: 1. Choose ONLY from the given genres: {', '.join(candidate_genres)}. 2. Output ONLY the predicted genres as a comma-separated list. 3. Do NOT repeat or copy the full genre list. 4. Do NOT add explanations or extra text.",
        },
        {
            "role": "user",
            "content": f"User's Most Frequently Watched Genres (from most to least frequent): {', '.join(sorted_user_genre_counts)}\n"
            f"User's Watched Genres Sorted by Rarity (from rarest to most common): {', '.join(sorted_watched_genres_by_rarity)}\n"
            f"Rarer and more frequent genres should be given priority if they are relevant.\n"
            f"Predicted Genres:",
        },
    ]

    output = model(prompt, max_new_tokens=15)[0]["generated_text"][-1]["content"]

    # 예측된 장르 리스트 변환
    predicted_genres = [genre.strip() for genre in output.split(",")]
    filtered_genres = [genre for genre in predicted_genres if genre in ALL_GENRES]
    print(filtered_genres)
    # 3개 중 랜덤하게 하나 고르기
    genre = random.choice(filtered_genres) if filtered_genres else None

    return genre


def predict_user_preferred_genres_test(
    user_collection,
    item_collection,
    reviewer_id,
    model_name="unsloth/phi-4-unsloth-bnb-4bit",
    k=3,
    alpha=0.7,
    beta=0.5,
):
    """
    유저의 영화 시퀀스를 기반으로 LLM을 활용해 선호 장르를 예측
    """
    user = user_collection.find_one(
        {"reviewerID": reviewer_id}, {"items.predicted_genre": 1}
    )

    if not user or "items" not in user:
        return []

    # 유저가 본 영화에서 `predicted_genre` 가져오기
    watched_genres = [
        genre
        for item in user["items"]
        if "predicted_genre" in item
        for genre in item["predicted_genre"]
    ]

    if not watched_genres:
        return []

    # 유저의 장르 빈도수 계산
    user_genre_counts = Counter(watched_genres)
    candidate_genres = set(watched_genres)
    if len(candidate_genres) < 3:
        k = len(candidate_genres)

    sorted_user_genre_counts = sorted(
        user_genre_counts.keys(),
        key=lambda genre: user_genre_counts[genre],
        reverse=True,
    )

    # 전역 장르 분포 계산 (전체 영화 데이터 기반)
    global_genre_distribution = calculate_genre_distribution(
        item_collection, ALL_GENRES
    )
    sorted_watched_genres_by_rarity = sorted(
        candidate_genres,
        key=lambda genre: global_genre_distribution.get(genre, 1),
        reverse=False,
    )

    # 적게 등장한 장르에 가중치 부여
    # total_watched = sum(user_genre_counts.values())
    # genre_weights = {}
    # for genre in user_genre_counts:
    #     user_freq = user_genre_counts[genre] / total_watched
    #     global_freq = global_genre_distribution.get(genre, 0)

    #     # 적게 등장한 장르는 높은 가중치를 받도록 설계
    #     genre_weights[genre] = (1 - alpha) * user_freq + alpha * ((1 - global_freq) ** beta)

    # # 가중치를 적용한 장르 정렬
    # weighted_genres = sorted(genre_weights, key=genre_weights.get, reverse=True)
    # print(weighted_genres)

    prompt = [
        {
            "role": "system",
            "content": f"You are an AI movie genre predictor. Your task is to determine the top {k} genres a user is most likely to prefer based on their past viewing history and genre rarity. Follow these rules: 1. Choose ONLY from the given genres: {', '.join(candidate_genres)}. 2. Output ONLY the predicted genres as a comma-separated list. 3. Do NOT repeat or copy the full genre list. 4. Do NOT add explanations or extra text.",
        },
        {
            "role": "user",
            "content": f"User's Most Frequently Watched Genres (from most to least frequent): {', '.join(sorted_user_genre_counts)}\n"
            f"User's Watched Genres Sorted by Rarity (from rarest to most common): {', '.join(sorted_watched_genres_by_rarity)}\n"
            f"Rarer and more frequent genres should be given priority if they are relevant.\n"
            f"Predicted Genres:",
        },
    ]

    # LLM 예측 수행
    genre_pipeline = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )
    output = genre_pipeline(prompt, max_new_tokens=15)[0]["generated_text"][-1][
        "content"
    ]
    # breakpoint()

    # 예측된 장르 리스트 변환
    predicted_genres = [genre.strip() for genre in output.split(",")]
    filtered_genres = [genre for genre in predicted_genres if genre in ALL_GENRES]
    print(filtered_genres)
    # 3개 중 랜덤하게 하나 고르기
    genre = random.choice(filtered_genres) if filtered_genres else None

    return genre


def main():
    # MongoDB 연결 설정
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    item_collection = db["item"]
    user_collection = db["user"]

    # 특정 유저에 대한 장르 예측 실행
    reviewer_id = "A2M1CU2IRZG0K9"
    predicted_genre = predict_user_preferred_genres_test(
        user_collection, item_collection, reviewer_id, ALL_GENRES
    )
    print(predicted_genre)
