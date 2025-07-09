import os

from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import pipeline

from recsys_and_llm.backend.app.config import ALL_GENRES


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
        f.write("LLM Amazon Movie Genre Prediction Results(resume version)\n")
        f.write("=" * 100 + "\n\n")

    # find() -> 커서 열어서 데이터 조회 유지, 서버 메모리가 부족하거나 네트워크 불안정할 때 커서 끊김
    # 커서 타임아웃 비활성화 + 커서를 batch size로 나눠서 처리
    # _id >= 9343인 데이터만 가져오기
    with client.start_session() as session:
        for movie in item_collection.find(
            batch_size=100, no_cursor_timeout=True, session=session
        ):
            print(movie["_id"])

            title = movie.get("title", "")
            description_list = movie.get("description", [])
            description = max(
                description_list, key=len, default=""
            )  # 가장 긴 설명 선택

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

            # 아이템 테이블 정보 저장
            item_collection.update_one(
                {"_id": movie["_id"]}, {"$set": {"predicted_genre": filtered_preds}}
            )

            # 해당 아이템을 가진 유저만 찾아 업데이트 (최적화)
            users_with_item = user_collection.find(
                {"items.itemnum": int(movie["_id"])}, {"_id": 1}
            )
            user_ids = [user["_id"] for user in users_with_item]

            if user_ids:
                user_collection.update_many(
                    {"_id": {"$in": user_ids}, "items.itemnum": int(movie["_id"])},
                    {"$set": {"items.$[elem].predicted_genre": filtered_preds}},
                    array_filters=[{"elem.itemnum": int(movie["_id"])}],
                )

    print("Genre prediction completed and updated in the database.")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Final Evaluation Metrics:\n")
        f.write("=" * 100 + "\n")


# MongoDB 연결 및 실행 예시
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME")]
item_collection = db["item"]
user_collection = db["user"]


# 함수 실행
predict_movie_genres(item_collection, user_collection, ALL_GENRES)
