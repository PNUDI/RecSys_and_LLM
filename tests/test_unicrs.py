import os

import pytest
from dotenv import load_dotenv
from recwizard import ChatgptGen, FillBlankConfig, FillBlankPipeline, UnicrsRec

# ------------------------------------------------------------------------------
# 테스트용 파이프라인 fixture
# 실제 모델 로딩을 수행하므로, 네트워크나 외부 API 호출이 발생할 수 있습니다.
# 필요하다면 여기에 mocking을 추가하는 것도 좋습니다.
# ------------------------------------------------------------------------------

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


@pytest.fixture
def pipeline():
    # 추천 모듈: UnicrsRec
    rec_module = UnicrsRec.from_pretrained("recwizard/unicrs-rec-redial")
    # 생성 모듈: ChatgptGen
    gen_module = ChatgptGen.from_pretrained("RecWizard/chatgpt-gen-fillblank")
    # FillBlank 모드에 사용할 config 생성
    config = FillBlankConfig()
    # 파이프라인 생성
    return FillBlankPipeline(
        config=config, rec_module=rec_module, gen_module=gen_module
    )


# ------------------------------------------------------------------------------
# 단일 턴 대화 테스트
# 사용자 입력을 대화 컨텍스트에 하드코딩하여 pipeline.response()의 응답이
# 올바른 형식(비어있지 않은 문자열)으로 나오는지를 확인합니다.
# ------------------------------------------------------------------------------
def test_single_turn_conversation(pipeline):
    # 초기 대화 컨텍스트 (하드코딩)
    dialog = ["System: 안녕하세요! 채팅을 시작합니다."]

    # 1회 턴: 사용자 입력 추가 (여기서는 직접 문자열 추가)
    user_input = "오늘 기분이 어때?"
    dialog.append(f"User: {user_input}")

    # 전체 컨텍스트는 <sep>로 join
    full_context = "<sep>".join(dialog)

    # 파이프라인 응답 생성
    response = pipeline.response(full_context)

    # 응답 출력 (필요에 따라 캡쳐 가능)
    print("System:", response)

    # 응답 검증: 문자열이며 공백만 있는 경우가 아니어야 함
    assert isinstance(response, str)
    assert response.strip() != ""

    # 시스템 응답을 대화 컨텍스트에 추가 (대화 시뮬레이션)
    dialog.append(f"System: {response}")

    # 추가 검증: 대화 내에 System:으로 시작하는 문장이 2개 이상 있는지 확인
    system_responses = [line for line in dialog if line.startswith("System:")]
    assert len(system_responses) >= 2


# ------------------------------------------------------------------------------
# 다중 턴 대화 시나리오 테스트
# monkeypatch를 이용하여 입력을 하드코딩한 후 대화 흐름이 올바르게 진행되는지를 검증합니다.
# ------------------------------------------------------------------------------
def test_multi_turn_conversation(monkeypatch, pipeline):
    # 초기 대화 컨텍스트 (하드코딩)
    dialog = ["System: 안녕하세요! 채팅을 시작합니다."]

    # 테스트용 입력 리스트 (마지막 입력은 'quit'로 종료)
    inputs = iter(
        [
            "안녕",  # 1번째 사용자 입력
            "오늘 뭐해?",  # 2번째 사용자 입력
            "quit",  # 종료 입력
        ]
    )

    # builtins.input을 monkeypatch로 대체
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    # 채팅 시뮬레이션 (while 루프를 하드코딩)
    while True:
        user_input = input("User: ")  # monkeypatch 덕분에 미리 정해진 값 반환
        if user_input.lower() == "quit":
            break

        # 사용자 입력을 대화 컨텍스트에 추가
        dialog.append(f"User: {user_input}")

        # 전체 컨텍스트 연결
        full_context = "<sep>".join(dialog)

        # 파이프라인 응답 생성
        response = pipeline.response(full_context)
        print("System:", response)  # 실제 출력 확인(캡쳐 가능)
        dialog.append(f"System: {response}")

    # 대화 내에 적어도 2번 이상의 시스템 응답이 존재해야 함
    system_responses = [line for line in dialog if line.startswith("System:")]
    assert len(system_responses) >= 2
