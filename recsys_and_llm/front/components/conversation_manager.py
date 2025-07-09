# components/conversation_manager.py

import json
import os
import time

import streamlit as st


def generate_conversation_id(user_id):
    """현재 시간을 이용해 고유한 conversation ID 생성"""
    timestamp = int(time.time())  # 현재 시간을 UNIX timestamp로 변환
    return f"{user_id}_{timestamp}"


def save_conversation(title, user_id):
    """
    개별 대화를 저장하는 함수
    """
    if title.strip():
        conversation_id = generate_conversation_id(user_id)
        conversation_data = {
            "conversation_id": conversation_id,
            "conversation_title": title,
            "user_id": user_id,
            "conversations": st.session_state.conversations.copy(),
            "dialog": st.session_state.dialog.copy(),
        }

        filename = f"{user_id}_session.json"

        # 기존 JSON 파일에서 데이터를 로드
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    saved_data = json.load(f)
                    if not isinstance(
                        saved_data, list
                    ):  # 만약 리스트가 아니라 dict라면 변환
                        saved_data = [saved_data]
                except json.JSONDecodeError:
                    saved_data = []
        else:
            saved_data = []

        # 새로운 대화를 리스트에 추가 후 저장
        saved_data.append(conversation_data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(saved_data, f, ensure_ascii=False, indent=4)

        return f"'{title}' 대화를 저장했습니다! (ID: {conversation_id})"

    return "세션 제목이 비어있어 저장을 취소했습니다."


def load_conversation(user_id, conversation_id):
    """
    저장된 대화를 불러오는 함수
    """
    filename = f"{user_id}_session.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        # 특정 conversation_id와 user_id가 일치하는 데이터 찾기
        for conversation in saved_data:
            if (
                conversation["conversation_id"] == conversation_id
                and conversation["user_id"] == user_id
            ):
                st.session_state.conversations = conversation["conversations"].copy()
                st.session_state.dialog = conversation["dialog"].copy()
                return f"'{conversation['conversation_title']}' 대화를 불러왔습니다. (ID: {conversation_id})"

    return "해당 대화를 찾을 수 없습니다."


def retrieve_all_conversations(user_id):
    """
    특정 사용자의 모든 저장된 대화 목록을 가져오는 함수
    """
    filename = f"{user_id}_session.json"

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                saved_data = json.load(f)
                if not isinstance(saved_data, list):  # 파일이 리스트 형태가 아닐 경우
                    saved_data = []
            except json.JSONDecodeError:
                saved_data = []  # JSON 파싱 오류가 발생하면 빈 리스트 반환

        return [
            {
                "conversation_id": c.get("conversation_id", "N/A"),
                "conversation_title": c.get(
                    "conversation_title", "Untitled Conversation"
                ),
            }
            for c in saved_data
            if isinstance(c, dict)  # 리스트 내 요소가 dict인지 확인
        ]

    return []  # 저장된 대화가 없을 경우 빈 리스트 반환
