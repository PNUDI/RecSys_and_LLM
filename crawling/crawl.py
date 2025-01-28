import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


def get_chrome_options():
    """
    Returns a configured ChromeOptions object.
    """
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # 브라우저 창 숨기기
    chrome_options.add_argument("--disable-headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_experimental_option("prefs", {"intl.accept_languages": "en"})
    return chrome_options


def fetch_movie_data(
    element_index, keyword_text, driver_path, chrome_options, csv_file
):
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )

    try:
        # 메인 페이지로 이동
        url = "https://www.imdb.com/search/keyword/?s=kw"
        driver.get(url)

        # 모든 a 요소 선택
        parent_xpath = '//*[@id="__next"]/main/div/section/div/section/div/div[1]/section/div[2]/div[2]'
        elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, f"{parent_xpath}/a"))
        )

        # 지정된 인덱스의 요소 선택
        target_element = elements[element_index]
        ActionChains(driver).move_to_element(target_element).perform()
        target_element.click()

        # 목록 확장
        while True:
            try:
                # 'Load More' 버튼 대기
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button',
                        )
                    )
                )

                # 버튼이 화면에 보이도록 스크롤
                ActionChains(driver).move_to_element(load_more_button).perform()

                # 버튼 클릭
                try:
                    load_more_button.click()
                except Exception:
                    # JavaScript를 통한 클릭 (예외 발생 시)
                    driver.execute_script("arguments[0].click();", load_more_button)

                # 데이터 로드 대기
                time.sleep(1)

            except Exception as e:
                print(f"No more 'Load More' button or error occurred: {e}")
                break

        # 영화 목록 가져오기
        movie_list = driver.find_elements(
            By.XPATH,
            '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li',
        )
        for movie in movie_list:
            try:
                title = movie.find_element(
                    By.XPATH, "./div/div/div/div[1]/div[2]/div[1]/a/h3"
                ).text
                description = movie.find_element(
                    By.XPATH, "./div/div/div/div[2]/div/div"
                ).text
                rating = float(
                    movie.find_element(
                        By.XPATH, "./div/div/div/div[1]/div[2]/span/div/span/span[1]"
                    ).text
                )
                title_cleaned = re.sub(r"^\d+\.\s*", "", title).strip()

                # CSV에 바로 쓰기
                with open(csv_file, mode="a", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([keyword_text, title_cleaned, description, rating])

            except Exception:
                continue

    finally:
        driver.quit()


def main():
    driver_path = ChromeDriverManager().install()
    chrome_options = get_chrome_options()
    csv_file = "movies_with_keywords.csv"

    # CSV 파일 초기화 (헤더 작성)
    with open(csv_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Keyword", "Title", "Description", "Rating"])

    # Selenium으로 메인 페이지의 모든 요소 가져오기
    driver = webdriver.Chrome(service=Service(driver_path), options=chrome_options)

    url = "https://www.imdb.com/search/keyword/?s=kw"
    driver.get(url)

    parent_xpath = '//*[@id="__next"]/main/div/section/div/section/div/div[1]/section/div[2]/div[2]'
    elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, f"{parent_xpath}/a"))
    )
    total_elements = len(elements)

    # 각 요소의 텍스트 (키워드) 가져오기
    keywords = [element.text.strip() for element in elements]

    with open("keywords.txt", "w", encoding="utf-8") as file:
        for keyword in keywords:
            file.write(f"{keyword}\n")

    driver.quit()

    # 병렬 처리
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(
                fetch_movie_data,
                index,
                keywords[index],
                driver_path,
                chrome_options,
                csv_file,
            ): index
            for index in range(total_elements)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                future.result()  # 각 작업 완료 대기
            except Exception as e:
                print(f"Error with element {index}: {e}")

    print(f"Movie data saved to {csv_file}")


if __name__ == "__main__":
    main()
