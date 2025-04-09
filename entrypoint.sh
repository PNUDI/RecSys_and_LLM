#!/bin/bash

# FastAPI 백엔드 실행 (백그라운드)
uvicorn recsys_and_llm.backend.app.main:app --host 0.0.0.0 --port 8000 &

# Streamlit 프론트 실행 (포그라운드)
streamlit run recsys_and_llm/front/app.py --server.port=8501
