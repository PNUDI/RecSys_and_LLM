import csv
from collections import defaultdict


def clean_csv(input_csv, output_csv):
    """
    Cleans the input CSV by merging rows with the same Title and Description.
    The Keywords are merged into a list.
    """
    # 중복 제거를 위한 딕셔너리
    data_dict = defaultdict(lambda: {"Keywords": set(), "Rating": None})

    # CSV 읽기
    with open(input_csv, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                title = row["Title"]
                description = row["Description"]
                keyword = row["Keyword"]
                rating = float(row["Rating"]) if row["Rating"] else None

                # Title과 Description을 키로 사용하여 중복 체크
                key = (title, description)

                # Keywords 병합
                data_dict[key]["Keywords"].add(keyword)

                # Rating 업데이트 (가장 높은 평점 사용)
                if data_dict[key]["Rating"] is None or (
                    rating and rating > data_dict[key]["Rating"]
                ):
                    data_dict[key]["Rating"] = rating
            except:
                print(
                    f"error: {row["Keyword"]} / {row["Title"]} / {row["Description"]}"
                )
                continue

    # CSV 쓰기
    with open(output_csv, mode="w", encoding="utf-8", newline="") as file:
        fieldnames = ["Title", "Description", "Keywords", "Rating"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for (title, description), values in data_dict.items():
            writer.writerow(
                {
                    "Title": title,
                    "Description": description,
                    "Keywords": list(values["Keywords"]),  # Keywords를 리스트로 변환
                    "Rating": values["Rating"],
                }
            )

    print(f"Cleaned data saved to {output_csv}")


if __name__ == "__main__":
    # 사용 예시
    input_csv = "movies_with_keywords.csv"
    output_csv = "cleaned_movies_with_keywords.csv"
    clean_csv(input_csv, output_csv)
