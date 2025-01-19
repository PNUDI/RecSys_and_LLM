import re
import time

import requests
from bs4 import BeautifulSoup


def extract_largest_image_src(img_tag):
    """
    <img> 태그에서 srcset을 정규표현식으로 파싱해,
    가장 큰 해상도(XXXw)를 가진 URL을 반환.
    srcset이 없으면 그냥 src를 반환한다.
    """
    src = img_tag.get("src", "")
    srcset = img_tag.get("srcset", "")
    if not srcset:
        # srcset이 없으면 그냥 src 반환
        return src

    # 정규표현식으로 "URL 숫자w" 형태를 찾는다.
    # 예: ("https://..._UX380_.jpg", "380")
    pattern = r"(\S+)\s+(\d+)w"
    matches = re.findall(pattern, srcset)

    if not matches:
        return src

    largest_url = None
    largest_width = -1

    for url_part, width_str in matches:
        try:
            w = int(width_str)
            if w > largest_width:
                largest_width = w
                largest_url = url_part
        except ValueError:
            pass

    return largest_url if largest_url else src


def find_main_poster_tag(soup):
    """
    '메인 포스터'로 추정되는 <img> 태그를 우선순위대로 찾는다.
      1) div[data-testid="hero-title-block__poster"] img.ipc-image
      2) .ipc-poster__poster-image img.ipc-image
      3) img[data-testid="hero-media__poster"]
      4) img.ipc-image[alt*="poster" i]
    실패 시 None 반환.
    """
    # 1) hero-title-block__poster
    poster = soup.select_one(
        'div[data-testid="hero-title-block__poster"] img.ipc-image'
    )
    if poster:
        return poster

    # 2) .ipc-poster__poster-image
    poster = soup.select_one(".ipc-poster__poster-image img.ipc-image")
    if poster:
        return poster

    # 3) data-testid="hero-media__poster"
    poster = soup.select_one('img[data-testid="hero-media__poster"]')
    if poster:
        return poster

    # 4) alt에 "poster" 단어가 들어가는 img
    poster = soup.select_one('img.ipc-image[alt*="poster" i]')
    if poster:
        return poster

    return None


def search_imdb(title):
    """
    IMDb 검색 페이지에서 해당 title을 검색하고,
    첫 번째 /title/ 링크의 상세 페이지 URL을 반환.
    실패 시 None.
    """
    base_search_url = "https://www.imdb.com/find"
    params = {"q": title, "ref_": "nv_sr_sm"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/87.0.4280.88 Safari/537.36"
        )
    }

    resp = requests.get(base_search_url, params=params, headers=headers)
    print(f"[DEBUG] HTTP 상태 코드: {resp.status_code}, 요청 URL: {resp.url}")
    if resp.status_code != 200:
        print(f"[ERROR] 검색 페이지 접근 실패: {title}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 모든 /title/ 링크
    title_links = soup.select('a[href^="/title/"]')
    if not title_links:
        print(f"[INFO] /title/ 링크 없음: {title}")
        return None

    title_lower = title.lower()
    candidate_link = None

    # 검색어와 유사한 링크부터 시도
    for link in title_links:
        link_text = link.get_text(strip=True).lower()
        parent_text = (
            link.find_parent().get_text(strip=True).lower()
            if link.find_parent()
            else ""
        )
        if title_lower in link_text or title_lower in parent_text:
            candidate_link = link
            break

    # 없으면 첫 번째 /title/ 링크 fallback
    if not candidate_link:
        candidate_link = title_links[0]

    detail_href = candidate_link.get("href", "")
    detail_url = "https://www.imdb.com" + detail_href.split("?")[0]
    return detail_url


def get_imdb_details(detail_url):
    """
    상세 페이지(detail_url)에서
    - 감독(director)
    - 주요 출연진(최대5명)
    - 포스터(가장 큰 해상도) URL
    을 딕셔너리 형태로 반환.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/87.0.4280.88 Safari/537.36"
        )
    }
    resp = requests.get(detail_url, headers=headers)
    if resp.status_code != 200:
        print(f"[ERROR] 상세 페이지 접근 실패: {detail_url}")
        return {"director": None, "cast": [], "poster_url": None}

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1) 감독
    director = None
    credits = soup.select('li[data-testid="title-pc-principal-credit"]')
    for item in credits:
        txt = item.get_text(strip=True)
        if "Director" in txt or "Directed by" in txt:
            link = item.select_one("a[href^='/name/']")
            if link:
                director = link.get_text(strip=True)
            break

    # 2) 출연진(최대 5명)
    cast = []
    cast_tags = soup.select(
        "div[data-testid='title-cast-item'] a[data-testid='title-cast-item__actor']"
    )
    for ctag in cast_tags[:5]:
        cast.append(ctag.get_text(strip=True))

    # 3) 포스터
    poster_tag = find_main_poster_tag(soup)
    if poster_tag:
        poster_url = extract_largest_image_src(poster_tag)
    else:
        poster_url = None

    return {"director": director, "cast": cast, "poster_url": poster_url}


def main():
    titles = ["Kimchi Chronicles", "The Way Back Home", "Frank Lloyd Wright"]

    for title in titles:
        print(f"\n크롤링 중: {title}")
        detail_url = search_imdb(title)
        if not detail_url:
            print(f"[결과] {title}: 검색 실패 또는 결과 없음.")
            print("-" * 80)
            time.sleep(2)
            continue

        print(f"[DEBUG] Detail URL: {detail_url}")
        details = get_imdb_details(detail_url)
        print(f"[결과] {title}")
        print(f"  - 감독: {details['director']}")
        print(f"  - 출연진(최대5명): {details['cast']}")
        print(f"  - 포스터 URL: {details['poster_url']}")
        print("-" * 80)

        # IMDb 요청 간격 (과도한 트래픽 방지)
        time.sleep(5)


if __name__ == "__main__":
    main()
