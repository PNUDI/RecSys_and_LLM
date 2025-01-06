import os

import requests
from dotenv import load_dotenv

# .env 파일에서 Bearer Token 로드
load_dotenv()
BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")
BASE_URL = "https://api.themoviedb.org/3"


# 영화 검색 함수
def search_movie(query, language="en-US"):
    # query를 문자열 파싱해서 맨 마지막이 2024와 같은 숫자 형식이면 year의 값을 할당
    # 아니면 year를 None으로 설정
    year = None
    if query[-4:].isdigit():
        year = query[-4:]
        query = query[:-4].strip()

    url = f"{BASE_URL}/search/movie"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "accept": "application/json"}
    params = {
        "query": query,
        "language": language,
        "page": "1",
        "include_adult": "true",
        "year": year,
    }
    response = requests.get(url, headers=headers, params=params)
    # print(response.json())

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


# 영화 크레딧(감독, 출연진) 정보 가져오기
def get_movie_credits(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "accept": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        credits = response.json()
        cast = credits.get("cast", [])
        crew = credits.get("crew", [])

        # 감독 정보
        director = next(
            (member for member in crew if member.get("job") == "Director"), None
        )

        # 주요 출연진 (최대 5명)
        top_cast = cast[:5]

        return director, top_cast
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None, []


# 실행 예제
if __name__ == "__main__":
    # 테스트용 검색어
    query = "oppenheimer 2023"

    # 영화 검색
    results = search_movie(query)
    if results:
        first_result = results[0]
        movie_id = first_result.get("id")
        title = first_result.get("title", "No Title")
        overview = first_result.get("overview", "No Overview")
        poster_path = first_result.get("poster_path")
        poster_url = get_poster_url(poster_path)

        director, top_cast = get_movie_credits(movie_id)

        # 출력
        print(f"Title: {title}")
        print(f"Overview: {overview}")
        print(f"Poster URL: {poster_url}")
        if director:
            print(f"Director: {director.get('name')}")
        print("Top Cast:")
        for cast_member in top_cast:
            print(f" - {cast_member.get('name')} as {cast_member.get('character')}")
    else:
        print("No results found.")
