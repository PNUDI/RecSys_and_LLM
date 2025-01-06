import pickle

# 데이터 로드
with open("./Movies_and_TV_text_name_dict.json.gz", "rb") as ft:
    text_name_dict = pickle.load(ft)

# title만 추출
titles = []
t = "title"
t_ = "No Title"
items = range(1, 86678 + 1)

for i in items:
    title = text_name_dict[t].get(i, t_)
    titles.append(title)

# 추출한 title을 txt 파일에 저장
with open("titles_only.txt", "w", encoding="utf-8") as f:
    for title in titles:
        f.write(f"{title}\n")

print("Title만 저장 완료!")
