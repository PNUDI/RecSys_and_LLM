import re
from collections import Counter
from datetime import datetime

import requests

from recsys_and_llm.backend.app.config import DEFAULT_IMAGE_URL
from recsys_and_llm.backend.app.inference import (
    genre_inference,
    inference,
    item_content_inference,
)
from recsys_and_llm.backend.app.schemas import *


def get_item_img(url_lst):
    if not url_lst:  # 리스트가 비어있으면 기본 이미지 반환
        return DEFAULT_IMAGE_URL

    for url in url_lst:
        try:
            response = requests.head(url, timeout=3)  # HEAD 요청으로 빠르게 확인
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue

    return DEFAULT_IMAGE_URL


def sign_up(request: SignUpRequest, user_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if user_data:
        return ApiResponse(success=False, message="이미 존재하는 아이디입니다.")

    existing_user = user_collection.find_one({"userName": request.name})
    if existing_user:
        return ApiResponse(success=False, message="이미 존재하는 닉네임입니다.")

    new_user = {
        "_id": str(user_collection.count_documents({}) + 1),  # UserNum 자동 생성
        "reviewerID": request.reviewer_id,
        "password": request.password,
        "userName": request.name,
        "items": [],
    }

    user_collection.insert_one(new_user)
    return ApiResponse(
        success=True, message="회원가입 성공", data={"user_id": new_user["_id"]}
    )


def sign_in(
    request: SignInRequest,
    model_manager,
    user_collection,
    item_collection,
    rec_collection,
):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 사용자입니다.")

    if user_data["password"] != request.password:
        return ApiResponse(success=False, message="비밀번호가 일치하지 않습니다.")

    user_id = user_data["_id"]
    seq = [item["itemnum"] for item in user_data.get("items", [])]
    seq_time = [
        (item["itemnum"], item["unixReviewTime"]) for item in user_data.get("items", [])
    ]

    if model_manager is None:
        return ApiResponse(success=False, message="모델이 로드되지 않았습니다.")

    watched_genres = [
        genre
        for item in user_data["items"]
        if "predicted_genre" in item
        for genre in item["predicted_genre"]
    ]

    # 유저의 장르 빈도수 계산
    user_genre_counts = Counter(watched_genres)
    genre = genre_inference(model_manager, user_genre_counts)
    genre_movie_ids = [
        int(movie["_id"])
        for movie in item_collection.find({"predicted_genre": genre}, {"_id": 1})
    ]
    print(genre, len(genre_movie_ids))

    result = inference(model_manager, user_id, seq, seq_time, genre_movie_ids)

    # match = re.search(r"\(ID: (\d+)\)", result["allmrec_prediction"])
    # allmrec_ids = [match.group(1)] if match else []

    allmrec_ids = [str(result["allmrec_prediction"])]
    gsasrec_ids = list(map(str, result["gsasrec_prediction"]))
    tisasrec_ids = list(map(str, result["tisasrec_prediction"]))
    genrerec_ids = list(map(str, result["genrerec_prediction"]))

    all_ids = list(
        set(allmrec_ids + gsasrec_ids + tisasrec_ids + genrerec_ids)
    )  # 중복 제거
    items = item_collection.find(
        {"_id": {"$in": all_ids}}, {"_id": 1, "available_images": 1, "title": 1}
    )

    item_map = {
        item["_id"]: {
            "image": get_item_img(item.get("available_images", [])),
            "title": item.get(
                "title", "Unknown"
            ).strip(),  # title 추가, 기본값 "Unknown"
        }
        for item in items
    }

    predictions = {
        "prediction-1": {
            "item_id": allmrec_ids[0],
            "img_url": item_map.get(allmrec_ids[0], {}).get("image"),
            "title": item_map.get(allmrec_ids[0], {}).get("title"),
        },
        "prediction-2": [
            {
                "item_id": _id,
                "img_url": item_map.get(_id, {}).get("image"),
                "title": item_map.get(_id, {}).get("title"),
            }
            for _id in gsasrec_ids
        ],
        "prediction-3": [
            {
                "item_id": _id,
                "img_url": item_map.get(_id, {}).get("image"),
                "title": item_map.get(_id, {}).get("title"),
            }
            for _id in tisasrec_ids
        ],
        "prediction-4": {
            "genre": genre,
            "movies": [
                {
                    "item_id": _id,
                    "img_url": item_map.get(_id, {}).get("image"),
                    "title": item_map.get(_id, {}).get("title"),
                }
                for _id in genrerec_ids
            ],
        },
    }

    # 로그인 할 때마다 추천 테이블 업데이트
    rec_collection.update_one(
        {"reviewerID": user_data["reviewerID"]},
        {
            "$set": {
                "predictions": predictions,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        },
        upsert=True,
    )

    return ApiResponse(
        success=True,
        message="로그인 성공",
        data={
            "user_id": user_id,
            "name": user_data["userName"],
        },
    )


def detail_prediction(
    request: DetailPredictRequest,
    model_manager,
    user_collection,
    item_collection,
    review_collection,
):
    item_data = item_collection.find_one({"_id": request.item_id})

    if not item_data:
        return ApiResponse(success=False, message="존재하지 않는 아이템입니다.")

    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 사용자입니다.")

    img = get_item_img(item_data["available_images"])

    # 필요한 정보 추출
    review_data = review_collection.find_one({"_id": item_data["_id"]})
    reviews_list = review_data.get("review", []) if review_data else []

    review_data = {"my_review": "", "my_rating": "", "others": []}
    for review in reversed(reviews_list):  # 최신 리뷰가 먼저 오도록 역순 정렬
        user_name = review.get("userName", "Unknown User")  # 유저 이름 가져오기
        review_text = review.get("reviewText", "")  # 리뷰 내용 가져오기
        rating = review.get("rating", None)  # 평점 가져오기 (숫자)
        rating_str = str(rating) if rating is not None else ""

        if user_name == user_data["userName"]:
            review_data["my_review"] = review_text
            review_data["my_rating"] = rating_str
        else:
            review_data["others"].append(
                {"user_name": user_name, "review": review_text, "rating": rating_str}
            )

    title = item_data.get("title", "No Title Available")
    cast = item_data.get("cast", [])
    description_list = item_data.get("description", [])
    if isinstance(description_list, list) and description_list:
        description = max(description_list, key=len)
    else:
        description = "No Description Available"

    predictions = item_content_inference(model_manager, item_data["_id"])
    items = item_collection.find(
        {"_id": {"$in": predictions}}, {"_id": 1, "available_images": 1}
    )
    item_map = {
        item["_id"]: get_item_img(item.get("available_images", [])) for item in items
    }
    predictions = [
        {"item_id": _id, "img_url": item_map.get(_id, None)} for _id in predictions
    ]

    # 5. 반환할 데이터 구성
    return ApiResponse(
        success=True,
        message="아이템 정보 조회 성공",
        data={
            "img": img,
            "title": title,
            "cast": cast,
            "description": description,
            "reviews": review_data,
            "predictions": predictions,
        },
    )


def review_post(
    request: ReviewPostRequest, user_collection, item_collection, review_collection
):
    # 존재 여부 체크
    item_data = item_collection.find_one({"_id": request.item_id})

    if not item_data:
        return ApiResponse(success=False, message="존재하지 않는 아이템입니다.")

    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    # 리뷰 업데이트
    user_name = user_data["userName"]
    review_data = review_collection.find_one({"_id": item_data["_id"]})

    new_review = {
        "userName": user_name,  # 유저 이름
        "reviewText": request.review,  # 리뷰 내용
        "rating": request.rating,  # 평점 (숫자 그대로 저장)
    }

    if not review_data:
        new_review_data = {
            "_id": item_data["_id"],  # ItemNum
            "review": [new_review],  # 리스트로 초기화
        }
        review_collection.insert_one(new_review_data)
        seq_update = True
    else:
        existing_reviews = review_data.get("review", [])
        if any(review["userName"] == user_name for review in existing_reviews):
            review_collection.update_one(
                {"_id": item_data["_id"], "review.userName": user_name},
                {"$set": {"review.$": new_review}},
            )
            seq_update = False
        else:
            existing_reviews.append(new_review)
            review_collection.update_one(
                {"_id": item_data["_id"]},
                {
                    "$set": {"review": existing_reviews}
                },  # 리뷰 리스트 업데이트 - 새로운 리뷰 추가
            )
            seq_update = True

    if seq_update:
        # 유저 시퀀스 업데이트
        new_item = {
            "itemnum": item_data["_id"],
            "asin": item_data["asin"],
            "reviewText": request.review,
            "overall": request.rating,
            "summary": "",
            "unixReviewTime": int(datetime.utcnow().timestamp()),  # 현재 시간 기준
        }

        user_collection.update_one(
            {"_id": user_data["_id"]}, {"$push": {"items": new_item}}
        )

    return ApiResponse(success=True, message="리뷰 작성 성공")


def conv_save(request: ConversationSaveRequest, user_collection, conv_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    if conv_collection.find_one({"_id": request.conversation_id}):
        return ApiResponse(success=False, message="이미 존재하는 대화입니다.")

    print(user_data["reviewerID"])
    conversation_data = {
        "_id": request.conversation_id,  # MongoDB _id 필드에 conversation_id 저장
        "conversation_title": request.conversation_title,
        "reviewerID": user_data["reviewerID"],
        "pipeline": request.pipeline,
        "dialog": [
            {
                "text": entry.text,
                "speaker": entry.speaker,
                "feedback": entry.feedback,
                "entity": entry.entity,
                "date_time": (
                    datetime.fromisoformat(entry.date_time) if entry.date_time else None
                ),
            }
            for entry in request.dialog
        ],
    }

    conv_collection.insert_one(conversation_data)

    return ApiResponse(success=True, message="대화 저장 성공")


def conv_load(request: ConversationLoadRequest, user_collection, conv_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    conversation = conv_collection.find_one({"_id": request.conversation_id})

    if not conversation:
        return ApiResponse(success=False, message="존재하지 않는 대화입니다.")

    return ApiResponse(success=True, message="대화 내용 로드 성공", data=conversation)


def conv_list(request: ConversationListRequest, user_collection, conv_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    conversations = conv_collection.find(
        {"reviewerID": request.reviewer_id},
        {"_id": 1, "conversation_title": 1},
    )
    conversations_list = [
        {
            "conversation_id": str(conv["_id"]),  # _id를 conversation_id로 변경
            "conversation_title": conv["conversation_title"],
        }
        for conv in conversations
    ]

    if not conversations_list:
        return ApiResponse(success=False, message="대화가 존재하지 않습니다.")

    return ApiResponse(
        success=True, message="대화 리스트 로드 성공", data=conversations_list
    )


def rec_load(request: RecommendResultRequest, user_collection, rec_collection):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    rec_data = rec_collection.find_one({"reviewerID": request.reviewer_id})

    if not rec_data:
        return ApiResponse(success=False, message="추천 결과가 존재하지 않습니다.")

    return ApiResponse(
        success=True,
        message="추천 결과 로드 성공",
        data={
            "user_id": user_data["_id"],
            "name": user_data["userName"],
            "predictions": rec_data["predictions"],
        },
    )


# for test
def main_prediction(
    request: MainPredictRequest, model_manager, user_collection, item_collection
):
    user_data = user_collection.find_one({"reviewerID": request.reviewer_id})

    if not user_data:
        return ApiResponse(success=False, message="존재하지 않는 유저입니다.")

    seq = [item["itemnum"] for item in user_data.get("items", [])]
    seq_time = [
        (item["itemnum"], item["unixReviewTime"]) for item in user_data.get("items", [])
    ]
    print(seq)
    if model_manager is None:
        return ApiResponse(success=False, message="모델이 로드되지 않았습니다.")

    result = inference(model_manager, user_data["_id"], seq, seq_time)

    # match = re.search(r"\(ID: (\d+)\)", result["allmrec_prediction"])
    # allmrec_ids = [match.group(1)] if match else []

    allmrec_ids = [str(result["allmrec_prediction"])]
    gsasrec_ids = list(map(str, result["gsasrec_prediction"]))
    tisasrec_ids = list(map(str, result["tisasrec_prediction"]))

    all_ids = list(set(allmrec_ids + gsasrec_ids + tisasrec_ids))  # 중복 제거
    items = item_collection.find(
        {"_id": {"$in": all_ids}}, {"_id": 1, "available_images": 1}
    )

    item_map = {
        item["_id"]: get_item_img(item.get("available_images", [])) for item in items
    }

    predictions = {
        "prediction-1": {
            "item_id": allmrec_ids[0],
            "img_url": item_map.get(allmrec_ids[0], None),
        },
        "prediction-2": [
            {"item_id": _id, "img_url": item_map.get(_id, None)} for _id in gsasrec_ids
        ],
        "prediction-3": [
            {"item_id": _id, "img_url": item_map.get(_id, None)} for _id in tisasrec_ids
        ],
        "prediction-4": [],  # 장르 모델 완성 후 추가 예정
    }

    return ApiResponse(
        success=True, message="메인 추천 성공", data={"predictions": predictions}
    )
