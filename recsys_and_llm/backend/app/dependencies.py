import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pymongo import MongoClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app.config import DB_NAME, MONGO_URI
from ml.models.model_manager import ModelManager
from ml.utils import find_cold, get_missing, get_text_name_dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 시 실행할 코드"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    user_collection = db["user"]
    item_collection = db["item"]
    review_collection = db["review"]
    conversation_collection = db["conversation"]

    # 모델 사용 데이터 파싱
    cold_items = find_cold(user_collection, 50)
    text_name_dict = get_text_name_dict(item_collection)
    missing_list = get_missing(text_name_dict["title"])

    data = [cold_items, text_name_dict, missing_list]

    # 모델 로드
    model_manager = ModelManager(data)

    app.state.model_manager = model_manager
    app.state.user = user_collection
    app.state.item = item_collection
    app.state.review = review_collection
    app.state.conversation = conversation_collection

    yield

    # 리소스 정리
    del model_manager
    client.close()
    print("서버 종료: 모델 리소스 해제 및 MongoDB 연결 종료.")


def get_dependencies(request: Request):
    """FastAPI의 `request.app.state`에서 의존성을 가져오는 함수"""
    return {
        "model_manager": request.app.state.model_manager,
        "user": request.app.state.user,
        "item": request.app.state.item,
        "review": request.app.state.review,
        "conversation": request.app.state.conversation,
    }
