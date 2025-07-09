import os
import sys

from fastapi import APIRouter, Depends

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recsys_and_llm.backend.app.dependencies import get_dependencies
from recsys_and_llm.backend.app.schemas import *
from recsys_and_llm.backend.app.services import *

router = APIRouter()


@router.post("/sign-up")
def predict(request: SignUpRequest, dependencies: dict = Depends(get_dependencies)):
    return sign_up(request, dependencies["user"])


@router.post("/sign-in")
def predict(request: SignInRequest, dependencies: dict = Depends(get_dependencies)):
    return sign_in(
        request,
        dependencies["model_manager"],
        dependencies["user"],
        dependencies["item"],
        dependencies["recommend"],
    )


@router.post("/detail")
def predict(
    request: DetailPredictRequest, dependencies: dict = Depends(get_dependencies)
):
    return detail_prediction(
        request,
        dependencies["model_manager"],
        dependencies["user"],
        dependencies["item"],
        dependencies["review"],
    )


@router.post("/review")
def predict(request: ReviewPostRequest, dependencies: dict = Depends(get_dependencies)):
    return review_post(
        request,
        dependencies["user"],
        dependencies["item"],
        dependencies["review"],
    )


@router.post("/conv-save")
def predict(
    request: ConversationSaveRequest, dependencies: dict = Depends(get_dependencies)
):
    return conv_save(request, dependencies["user"], dependencies["conversation"])


@router.post("/conv-load")
def predict(
    request: ConversationLoadRequest, dependencies: dict = Depends(get_dependencies)
):
    return conv_load(request, dependencies["user"], dependencies["conversation"])


@router.post("/conv-list")
def predict(
    request: ConversationListRequest, dependencies: dict = Depends(get_dependencies)
):
    return conv_list(request, dependencies["user"], dependencies["conversation"])


# for test
@router.post("/rec-load")
def predict(
    request: RecommendResultRequest, dependencies: dict = Depends(get_dependencies)
):
    return rec_load(
        request,
        dependencies["user"],
        dependencies["recommend"],
    )


# for test
@router.post("/main")
def predict(
    request: MainPredictRequest, dependencies: dict = Depends(get_dependencies)
):
    return main_prediction(
        request,
        dependencies["model_manager"],
        dependencies["user"],
        dependencies["item"],
    )
