import html


def replace_special_entities(file_path):
    """
    파일에서 HTML 특수문자를 원래 문자로 변환한 뒤 다시 저장.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # html.unescape를 사용해 특수문자를 원래 문자로 변환
    restored_content = html.unescape(content)

    # 변환된 내용을 다시 파일에 저장
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(restored_content)

    print(f"[INFO] 특수문자를 변환한 내용이 {file_path}에 저장되었습니다.")


# 실행
file_path = "cleaned_title.txt"
replace_special_entities(file_path)
