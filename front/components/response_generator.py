# app/components/response_generator.py

def get_unicrs_response(user_message: str, pipeline) -> str:
    """
    주어진 파이프라인으로 사용자 메시지에 대한 응답 생성.
    """
    return f"안녕하세요! 요청하신 내용에 대한 추천을 드릴게요. (파이프라인: {pipeline})"
