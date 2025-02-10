import json
from sentence_transformers import SentenceTransformer

# 1. 모델 로드
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 2. 메타데이터 파싱
meta_data = []
count = 0
with open("data/amazon/meta_Movies_and_TV.json", "r") as f:
    for line in f:
        meta_data.append(json.loads(line))
        count +=1
        if count==10:
            break

# 3. 영화 설명 추출 (빈 값 필터링)
# print(meta_data[:10])

movie_descriptions = [m.get("description", "") for m in meta_data]
print(movie_descriptions[:10])
for desc in movie_descriptions:
    print(desc)
    print(type(desc))

# movie_descriptions = [desc for desc in movie_descriptions if desc.strip() != ""]  # 빈 문자열 제거

# # 4. 임베딩 생성 (입력을 명시적 단일 문장으로 처리)
# movie_embeddings = model.encode(
#     movie_descriptions, 
#     convert_to_tensor=True,  # 텐서 변환 활성화
#     show_progress_bar=True   # 진행률 표시
# )