import os

import requests
from dotenv import load_dotenv

# .env 파일에서 Bearer Token 로드
load_dotenv()
BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")
BASE_URL = "https://api.themoviedb.org/3"


# 영화 검색 함수
def search_movie(query, language="en-US"):
    url = f"{BASE_URL}/search/movie"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "accept": "application/json"}
    params = {"query": query, "language": language}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []


# 포스터 URL 생성 함수
def get_poster_url(poster_path, size="w500"):
    if poster_path:
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"
    else:
        return None


# 실행 예제
if __name__ == "__main__":
    # 테스트용 검색어
    query = "Batman"

    # 영화 검색
    results = search_movie(query)
    if results:
        first_result = results[0]
        title = first_result.get("title", "No Title")
        poster_path = first_result.get("poster_path")
        poster_url = get_poster_url(poster_path)

        print(f"Title: {title}")
        print(f"Poster URL: {poster_url}")
    else:
        print("No results found.")
