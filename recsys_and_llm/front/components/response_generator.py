# app/components/response_generator.py


def get_unicrs_response(user_message: str, dialog: list, pipeline) -> str:
    """
    주어진 파이프라인으로 사용자 메시지에 대한 응답 생성.
    """
    dialog.append(f"User: {user_message}")
    full_context = "<sep>".join(dialog)
    # 원래 응답 객체를 받고
    chat_message = pipeline.response(full_context)
    # ChatCompletionMessage 객체에서 content 속성을 추출
    response = (
        chat_message.content if hasattr(chat_message, "content") else chat_message
    )
    dialog.append(response)
    print("Dialog: ", dialog)
    print("Response: ", response)
    return response
