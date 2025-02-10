import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
import argparse
from pre_train.sasrec.model import SASRec  # SASRec 모델 불러오기

checkpoint_path = "/home/user/tmp/project/DILAB/A-LLMRec/pre_train/sasrec/Movies_and_TV/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"

checkpoint = torch.load(checkpoint_path, map_location="cpu")

if isinstance(checkpoint, dict):
    state_dict = checkpoint["model_state"]
    args_dict = checkpoint.get("args", {})
elif isinstance(checkpoint, (list, tuple)) and len(checkpoint) == 2:
    args_dict, state_dict = checkpoint  # 리스트 언패킹
else:
    raise ValueError("Checkpoint format is incorrect. Expected a dictionary or a 2-element list.")

args = argparse.Namespace(
    device="cpu",
    hidden_units=args_dict.get("hidden_units", 50),
    num_heads=args_dict.get("num_heads", 1),
    num_blocks=args_dict.get("num_blocks", 2),
    dropout_rate=args_dict.get("dropout_rate", 0.5),
    maxlen=args_dict.get("maxlen", 50),
    num_epochs=args_dict.get("num_epochs", 200),
    batch_size=args_dict.get("batch_size", 128),
    lr=args_dict.get("lr", 0.001),
    l2_emb=args_dict.get("l2_emb", 0.0)
)

user_num = 86679  # 데이터에서 확인한 값
item_num = 86678  # 데이터에서 확인한 값
sasrec_model = SASRec(user_num=user_num, item_num=item_num, args=args)

sasrec_model.load_state_dict(state_dict, strict=False)

sasrec_model.eval()

print(f"SASRec 모델 로드 완료! (User_num: {user_num}, Item_num: {item_num})")
print(f"모델 가중치 크기: {sasrec_model.item_emb.weight.shape}")



print(type(sasrec_model))  # torch.nn.Module인지 확인

# for name, param in sasrec_model.named_parameters():
#     print(f"Layer: {name}, Shape: {param.shape}")

# user_tensor = torch.tensor([123])
# log_seq = torch.randint(1, 10000, (1, 50))  # 임의의 시퀀스 데이터 생성
# cf_scores = sasrec_model.predict(user_tensor, log_seq, log_seq).detach().numpy().flatten()
# print(cf_scores[:10])  # 일부 결과 출력

print(type(sasrec_model))  # torch.nn.Module인지 확인
for name, param in sasrec_model.named_parameters():
    print(f"Layer: {name}, Shape: {param.shape}")  # 모델 가중치 정보 출력

df = pd.read_csv("data/amazon/meta_Movies_and_TV.csv")

sasrec_model = torch.load("/home/user/tmp/project/DILAB/A-LLMRec/pre_train/sasrec/Movies_and_TV/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth", map_location="cpu")

sbert_model = SentenceTransformer("paraphrase-mpnet-base-v2")
with open("data/sbert_movie_embeddings.pkl", "rb") as f:
    movie_embeddings = pickle.load(f)

def hybrid_recommend(user_id, query, top_k=10):
    """
    사용자의 협업 필터링 추천 + 검색 기반 추천을 결합한 최종 추천 함수
    """
    # 1️⃣ **검색 기반 추천 (SBERT)**
    query_embedding = sbert_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], movie_embeddings)[0]
    content_ranking = np.argsort(similarities)[-top_k:][::-1]

    user_tensor = torch.tensor([user_id])
    cf_scores = sasrec_model(user_tensor).detach().numpy().flatten()
    cf_ranking = np.argsort(cf_scores)[-top_k:][::-1]

    hybrid_scores = {}
    for idx in content_ranking:
        hybrid_scores[idx] = 0.7 * similarities[idx] + 0.3 * cf_scores[idx]
    
    for idx in cf_ranking:
        if idx in hybrid_scores:
            hybrid_scores[idx] += 0.7 * cf_scores[idx] + 0.3 * similarities[idx]
        else:
            hybrid_scores[idx] = 0.7 * cf_scores[idx] + 0.3 * similarities[idx]

    # 
    top_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_k]
    return df.iloc[top_indices][["title", "description"]]

# 테스트
print(hybrid_recommend(user_id=123, query="sci-fi space adventure"))
