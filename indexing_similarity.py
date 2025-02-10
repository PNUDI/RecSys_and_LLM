import pandas as pd
import ast

# CSV 파일 읽기 (데이터 타입 문제 해결)
df = pd.read_csv("data/amazon/meta_Movies_and_TV.csv", dtype=str, low_memory=False)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 빈 값이 있으면 공백("")으로 대체
df["description"] = df["description"].fillna("")

# TF-IDF 벡터화 (최대 단어 수 증가)
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df["description"])

# 검색 함수
def search_movies(query, top_k=10):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][["title", "description"]]

# 검색 테스트
print(search_movies("sci-fi space adventure"))
