"""
IMDb 데이터를 크롤링하여 JSON 파일로 저장하는 스크립트입니다.
- 크롤링 결과는 'imdb_parts' 폴더 내 JSON 파일로 저장됩니다.
- 크롤링 진행 상황 및 오류 로그는 'crawler.log' 파일에 기록됩니다.

사용 방법:
1. cleaned_title.txt 파일에 크롤링할 작품 제목을 한 줄씩 작성합니다.
2. 이 스크립트를 실행하면 결과가 JSON 파일로 저장되고, 로그 파일이 생성됩니다.
"""

import json
import logging
import os
import re
import time

import requests
from bs4 import BeautifulSoup

# 로그 설정
logging.basicConfig(
    filename="crawler.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_FOLDER = "imdb_parts"


def extract_largest_image_src(img_tag):
    src = img_tag.get("src", "")
    srcset = img_tag.get("srcset", "")
    if not srcset:
        return src

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
    poster = soup.select_one(
        'div[data-testid="hero-title-block__poster"] img.ipc-image'
    )
    if poster:
        return poster

    poster = soup.select_one(".ipc-poster__poster-image img.ipc-image")
    if poster:
        return poster

    poster = soup.select_one('img[data-testid="hero-media__poster"]')
    if poster:
        return poster

    poster = soup.select_one('img.ipc-image[alt*="poster" i]')
    if poster:
        return poster

    return None


def search_imdb(title):
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
    logging.info(f"Searching for title: {title} | HTTP Status: {resp.status_code}")
    if resp.status_code != 200:
        logging.error(f"Search failed for {title}. HTTP Status: {resp.status_code}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    title_links = soup.select('a[href^="/title/"]')
    if not title_links:
        logging.warning(f"No results found for title: {title}")
        return None

    title_lower = title.lower()
    for link in title_links:
        link_text = link.get_text(strip=True).lower()
        parent_text = (
            link.find_parent().get_text(strip=True).lower()
            if link.find_parent()
            else ""
        )
        if title_lower in link_text or title_lower in parent_text:
            return "https://www.imdb.com" + link.get("href").split("?")[0]

    return "https://www.imdb.com" + title_links[0].get("href").split("?")[0]


def get_imdb_details(detail_url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/87.0.4280.88 Safari/537.36"
        )
    }
    resp = requests.get(detail_url, headers=headers)
    if resp.status_code != 200:
        logging.error(f"Failed to access detail page: {detail_url}")
        return {"director": None, "cast": [], "poster_url": None}

    soup = BeautifulSoup(resp.text, "html.parser")
    director = None
    credits = soup.select('li[data-testid="title-pc-principal-credit"]')
    for item in credits:
        txt = item.get_text(strip=True)
        if "Director" in txt or "Directed by" in txt:
            link = item.select_one("a[href^='/name/']")
            if link:
                director = link.get_text(strip=True)
            break

    cast = []
    cast_tags = soup.select(
        "div[data-testid='title-cast-item'] a[data-testid='title-cast-item__actor']"
    )
    for ctag in cast_tags[:5]:
        cast.append(ctag.get_text(strip=True))

    poster_tag = find_main_poster_tag(soup)
    poster_url = extract_largest_image_src(poster_tag) if poster_tag else None

    return {"director": director, "cast": cast, "poster_url": poster_url}


def get_last_saved_index():
    json_files = [
        f
        for f in os.listdir(DATA_FOLDER)
        if f.startswith("imdb_data_part_") and f.endswith(".json")
    ]
    if not json_files:
        return 0

    last_file = sorted(json_files, key=lambda x: int(re.search(r"\d+", x).group()))[-1]
    last_index = int(re.search(r"\d+", last_file).group()) * 10
    return last_index


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    last_index = get_last_saved_index()
    logging.info(f"Starting from index: {last_index + 1}")

    data = {}
    with open("cleaned_title.txt", "r", encoding="utf-8") as f:
        titles = f.readlines()

    for i, raw_title in enumerate(titles[last_index:], start=last_index + 1):
        raw_title = raw_title.strip()
        logging.info(f"Processing item_number: {i}, title: {raw_title}")

        detail_url = search_imdb(raw_title)
        if not detail_url:
            logging.warning(f"Skipping title: {raw_title}")
            continue

        details = get_imdb_details(detail_url)
        data[i] = {
            "director": details["director"],
            "cast": details["cast"],
            "poster_url": details["poster_url"],
        }
        logging.info(f"Completed item_number: {i}")

        if i % 10 == 0 or i == len(titles):
            part_num = i // 10
            with open(
                f"{DATA_FOLDER}/imdb_data_part_{part_num}.json", "w", encoding="utf-8"
            ) as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            logging.info(f"Saved {DATA_FOLDER}/imdb_data_part_{part_num}.json")
            data = {}

        time.sleep(2)


if __name__ == "__main__":
    main()
