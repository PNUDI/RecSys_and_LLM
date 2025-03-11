from pymongo import MongoClient
from transformers import pipeline

from recsys_and_llm.backend.app.config import DB_NAME, MONGO_URI


def predict_movie_genres(
    item_collection,
    user_collection,
    all_genres,
    model_name="unsloth/phi-4-unsloth-bnb-4bit",
    output_file="recsys_and_llm/ml/models/Phi-4/llm_predictions.txt",
):
    """
    MongoDB 컬렉션을 순회하면서 영화 제목과 설명을 이용하여 장르를 예측하고, 결과를 DB에 업데이트합니다.
    """
    genre_pipeline = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("LLM Amazon Movie Genre Prediction Results\n")
        f.write("=" * 100 + "\n\n")

    for movie in item_collection.find():
        title = movie.get("title", "")
        description_list = movie.get("description", [])
        description = max(description_list, key=len, default="")  # 가장 긴 설명 선택

        # LLM 프롬프트 생성
        prompt = [
            {
                "role": "system",
                "content": f"You are an AI movie genre classifier. Your task is to assign the most appropriate genres to a movie. Follow these rules: 1. Choose ONLY from the given genres: {', '.join(all_genres)}. 2.Assign the most relevant genres (1, 2, or 3) based on fit. If a movie strongly fits only one genre, assign just one. If two genres are a good fit, assign two. 3. Output ONLY the predicted genres as a comma-separated list. 4. Do NOT repeat or copy the full genre list. 5. Do NOT add explanations or extra text.",
            },
            {
                "role": "user",
                "content": f"Movie Title: {title}, Story: {description}, Predicted Genres:",
            },
        ]

        # LLM을 이용한 예측 수행
        output = genre_pipeline(prompt, max_new_tokens=15)[0]["generated_text"][-1][
            "content"
        ]
        pred_genres = [genre.strip() for genre in output.split(",")]

        # 정해진 장르 목록에 포함된 것만 필터링
        filtered_preds = [genre for genre in pred_genres if genre in all_genres]

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"[Prompt]\n{prompt}\n\n")
            f.write(f"[LLM Predictions]\n{', '.join(filtered_preds)}\n\n")
            f.write("-" * 100 + "\n\n")

        user_collection.update_many(
            {"items.itemnum": movie["_id"]},
            {"$set": {"items.$[elem].predicted_genre": filtered_preds}},
            array_filters=[{"elem.itemnum": movie["_id"]}],
        )

    print("Genre prediction completed and updated in the database.")


# MongoDB 연결 및 실행 예시
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
item_collection = db["item"]
user_collection = db["user"]

# 사용 가능한 장르 리스트 (예시)
all_genres = [
    "Action",
    "Adult",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "Game-Show",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "News",
    "Reality-TV",
    "Romance",
    "Sci-Fi",
    "Short",
    "Sport",
    "Talk-Show",
    "Thriller",
    "War",
    "Western",
]

# 함수 실행
predict_movie_genres(item_collection, user_collection, all_genres)
