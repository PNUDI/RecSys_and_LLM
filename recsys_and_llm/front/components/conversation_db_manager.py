# components/conversation_db_manager.py

import time

import requests
import streamlit as st
from pymongo import MongoClient

from .conversation_manager import generate_conversation_id

url = "http://localhost:8001/"


def get_db_connection():
    """DB 연결을 설정하고 클라이언트 및 컬렉션을 반환"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["items"]  # 데이터베이스 선택
    conversation_coll = db["conversation"]  # 컬렉션 선택
    return client, conversation_coll


def db_save_conversation(title, user_id):
    """
    개별 대화를 DB에 저장하는 함수
    """
    endpoint = "conv-save"

    if title.strip():
        conversation_id = generate_conversation_id(user_id)
        dialog = []

        # 기존 dialog가 있으면 그대로 사용, 없으면 conversations에서 변환
        if hasattr(st.session_state, "dialog") and st.session_state.dialog:
            for conv in st.session_state.conversations:
                dialog_to_save = {
                    "text": conv["content"],
                    "speaker": "usr" if conv["role"] == "user" else "sys",
                    "feedback": (
                        None if conv["feedback"] == "None" else conv["feedback"]
                    ),  # 피드백 부분 추가 개선 필요
                    "entity": [],  # 엔티티 부분 추가 개선 필요
                    "date_time": conv["date_time"],
                }
                dialog.append(dialog_to_save)
        else:
            return "저장할 대화 내용이 없습니다!"

        # DB 저장을 위한 데이터 구조 생성 - ConversationSaveRequest에 맞게 구성
        new_conversation = {
            "conversation_id": conversation_id,
            "conversation_title": title,
            "reviewer_id": user_id,
            "pipeline": "gpt",
            "dialog": dialog,
        }

        try:
            response = requests.post(url + endpoint, json=new_conversation)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            result = response.json()
            return result.get("message", "저장 완료")
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            return f"저장 실패: {str(e)}\n{new_conversation}"


def db_load_conversation(user_id, conversation_id):
    """
    DB에 저장된 대화를 불러오는 함수
    """
    endpoint = "conv-load"
    target_conversation = {"reviewer_id": user_id, "conversation_id": conversation_id}
    post_processed_conversation = []
    try:
        response = requests.post(url + endpoint, json=target_conversation)
        response.raise_for_status()
        result = response.json()
        loaded_conversation = result.get("data")
        if result.get("success"):
            st.info("대화 로드 성공")
            print("----------------------------------------------------")
            print(result.get("data"))
            if loaded_conversation:

                for dialog in loaded_conversation.get("dialog", []):
                    if dialog["speaker"] == "usr":
                        role = "user"
                    else:
                        role = "assistant"
                    post_processed_conversation.append(
                        {
                            "role": role,
                            "content": dialog["text"],
                            "feedback": dialog["feedback"],
                            "date_time": dialog["date_time"],
                        }
                    )

            return post_processed_conversation
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
        return None


def db_retrieve_all_conversations(user_id):
    """
    DB에서 특정 사용자의 모든 대화 목록을 가져오는 함수
    """
    endpoint = "conv-list"
    target_user = {"reviewer_id": user_id}

    try:
        response = requests.post(url + endpoint, json=target_user)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            return result.get("data", [])
        else:
            return []
    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
        return []
