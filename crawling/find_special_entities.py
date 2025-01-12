import re


def find_special_entities(file_path):
    """
    파일에서 HTML 특수문자(&...;)를 모두 찾아 반환합니다.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # 정규표현식: &로 시작하고 ;로 끝나는 모든 패턴 찾기
    entities = re.findall(r"&[a-zA-Z0-9#]+;", text)

    # 중복 제거 및 정렬
    unique_entities = sorted(set(entities))
    return unique_entities


# 테스트
file_path = "cleaned_title.txt"
special_entities = find_special_entities(file_path)
print("발견된 특수문자:", special_entities)
