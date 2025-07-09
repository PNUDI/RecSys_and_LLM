from typing import Any, List, Optional

from pydantic import BaseModel


# 요청 schema
class SignUpRequest(BaseModel):
    reviewer_id: str
    password: str
    name: str


class SignInRequest(BaseModel):
    reviewer_id: str
    password: str


class DetailPredictRequest(BaseModel):
    reviewer_id: str
    item_id: str


class ReviewPostRequest(BaseModel):
    item_id: str
    reviewer_id: str
    review: str
    rating: str


# Conversation
class DialogEntry(BaseModel):
    text: str
    speaker: str
    feedback: Optional[str] = None
    entity: Optional[List[str]] = None
    date_time: str  # ISO format string representation


class ConversationSaveRequest(BaseModel):
    conversation_id: str
    conversation_title: str
    reviewer_id: str
    pipeline: str
    dialog: List[DialogEntry]


class ConversationLoadRequest(BaseModel):
    conversation_id: str
    reviewer_id: str


class ConversationListRequest(BaseModel):
    reviewer_id: str


class RecommendResultRequest(BaseModel):
    reviewer_id: str


# for test
class MainPredictRequest(BaseModel):
    reviewer_id: str


# 응답 schema
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None  # 성공 시 추가 데이터 포함 (Optional)
