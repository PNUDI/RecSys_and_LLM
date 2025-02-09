import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# IMDb 페이지 내 설명 요소의 XPATH
XPATH_DESCRIPTION = '//*[@id="__next"]/main/div/section[1]/section/div[3]/section/section/div[3]/div[2]/div[1]/section/p/span[2]'

CSV_FILE = "imdb_genre.csv"
UPDATED_CSV_FILE = "updated_imdb_genre.csv"

lock = threading.Lock()  # CSV 파일 쓰기 동기화를 위한 락
thread_local = threading.local()  # 각 스레드별 드라이버 인스턴스 보관

# 각 스레드가 생성한 드라이버 인스턴스를 저장할 집합 (종료시 활용)
driver_set = set()
driver_set_lock = threading.Lock()


def get_driver():
    """
    각 스레드별로 ChromeDriver 인스턴스를 생성하고 재사용.
    생성된 드라이버는 전역 driver_set에 추가하여 나중에 종료할 수 있도록 함.
    """
    if not hasattr(thread_local, "driver"):
        chrome_options = Options()
        # 필요에 따라 headless 모드 활성화: 아래 주석 해제
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_experimental_option("prefs", {"intl.accept_languages": "en"})

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=chrome_options
        )
        driver.maximize_window()
        thread_local.driver = driver

        with driver_set_lock:
            driver_set.add(driver)
    return thread_local.driver


def scrape_imdb_description(tconst, primaryTitle, genres):
    """
    IMDb 페이지에서 제목(tconst)에 해당하는 설명을 크롤링한 후, CSV 파일에 저장.
    각 스레드는 내부에서 자신의 드라이버 인스턴스를 get_driver()를 통해 가져옴.
    """
    driver = get_driver()
    imdb_url = f"https://www.imdb.com/title/{tconst}/"
    description_text = np.nan  # 기본값

    try:
        driver.get(imdb_url)

        # 설명 요소가 로드되고 텍스트가 채워질 때까지 대기
        WebDriverWait(driver, 10).until(
            lambda d: d.find_element(By.XPATH, XPATH_DESCRIPTION)
            .get_attribute("innerText")
            .strip()
            != ""
        )

        element = driver.find_element(By.XPATH, XPATH_DESCRIPTION)
        description_text = element.get_attribute("innerText").strip()

    except Exception as e:
        print(f"⚠️ {tconst}: 설명 요소 없음 또는 크롤링 오류 (NaN 저장) - {e}")
        description_text = np.nan

    save_to_csv(
        {
            "tconst": tconst,
            "primaryTitle": primaryTitle,
            "description": description_text,
            "genres": genres,
        }
    )


def save_to_csv(data):
    """
    크롤링한 데이터를 즉시 CSV 파일에 추가 저장.
    동시 접근 문제를 방지하기 위해 락을 사용.
    """
    with lock:
        with open(UPDATED_CSV_FILE, mode="a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    data["tconst"],
                    data["primaryTitle"],
                    data["description"],
                    data["genres"],
                ]
            )


def main():
    # CSV 파일에서 데이터를 읽어옴
    df = pd.read_csv(CSV_FILE)

    # 결과 저장 CSV 파일에 헤더 작성
    with open(UPDATED_CSV_FILE, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tconst", "primaryTitle", "description", "genres"])

    MAX_THREADS = 5  # 병렬 처리할 스레드 수

    """
    web driver 미리 생성해서 재사용 -> 스레드 & web driver 일대일 매핑 보장 X
    threading.lock()을 통해서 스레드마다 독립적인 저장공간 사용 -> 드라이버 매핑
    스레드가 처음 실행될 때만 드라이버 생성, 드라이버 존재하면 재사용
    """
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        for _, row in df.iterrows():
            futures.append(
                executor.submit(
                    scrape_imdb_description,
                    row["tconst"],
                    row["primaryTitle"],
                    row["genres"],
                )
            )

        # 작업 완료 및 예외 발생 시 체크
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"작업 중 예외 발생: {e}")

    # 모든 스레드가 사용한 드라이버 인스턴스 종료
    with driver_set_lock:
        for driver in driver_set:
            driver.quit()

    print(f"IMDb 크롤링 완료! '{UPDATED_CSV_FILE}' 파일에 실시간 저장됨.")


if __name__ == "__main__":
    main()
