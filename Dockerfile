# CUDA 12.2.2 + cuDNN + Ubuntu 22.04
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# python, pip 명령어 링크 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    [ -e /usr/bin/pip ] || ln -s /usr/bin/pip3 /usr/bin/pip

# poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s $HOME/.local/bin/poetry /usr/local/bin/poetry

# 작업 디렉터리 설정
WORKDIR /app

# poetry 환경설정: 가상환경 OFF
ENV POETRY_VIRTUALENVS_CREATE=false

# Python 모듈 import 경로 문제 해결 (ex: recsys_and_llm)
ENV PYTHONPATH="/app"

# 프로젝트 메타파일 복사 (캐시 활용)
COPY pyproject.toml poetry.lock ./

# 의존성 설치
RUN poetry install --no-root

# 설정 파일만 복사
COPY pyproject.toml poetry.lock ./

# 소스 코드 디렉토리만 복사
COPY ./recsys_and_llm ./recsys_and_llm

# entrypoint.sh 복사 및 실행 권한 부여
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 컨테이너 실행 시 기본 명령어 (streamlit 예시, 필요시 변경 가능)
CMD ["/entrypoint.sh"]
