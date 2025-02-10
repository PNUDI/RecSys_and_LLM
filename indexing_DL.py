import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import torch

# print("PyTorch 버전:", torch.__version__)
# print("CUDA 버전:", torch.version.cuda)
# print("CUDA 지원 가능 여부:", torch.cuda.is_available())
# print("GPU 개수:", torch.cuda.device_count())

# if torch.cuda.is_available():
#     print("사용 가능한 GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(torch.cuda.device_count( ))
print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

sbert_model = SentenceTransformer("paraphrase-mpnet-base-v2", device=device)

df = pd.read_csv("data/amazon/meta_Movies_and_TV.csv")  # 정리된 데이터셋 사용

embedding_cache_path = "data/sbert_movie_embeddings.pkl"

if os.path.exists(embedding_cache_path):
    with open(embedding_cache_path, "rb") as f:
        movie_embeddings = pickle.load(f)
    print("Cached embeddings loaded.")
else:
    print("Encoding movie descriptions...")
    movie_embeddings = sbert_model.encode(df["description"].tolist(), show_progress_bar=True)
    
    with open(embedding_cache_path, "wb") as f:
        pickle.dump(movie_embeddings, f)
    print("Embeddings saved.")

def search_movies_sbert(query, top_k=10):
    """
    검색어를 SBERT 벡터로 변환 후, 코사인 유사도를 기반으로 가장 적절한 영화를 추천
    """
    query_embedding = sbert_model.encode([query])[0]  # 검색어 임베딩
    similarities = cosine_similarity([query_embedding], movie_embeddings)[0]  # 유사도 계산
    
    top_indices = similarities.argsort()[-top_k:][::-1]  # 유사도 상위 영화 선택
    return df.iloc[top_indices][["title", "description"]]

print(search_movies_sbert("dream manipulation"))
