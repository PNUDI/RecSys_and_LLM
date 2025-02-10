import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import argparse
from pre_train.sasrec.model import SASRec  # SASRec 모델 불러오기

# ✅ 체크포인트 경로
checkpoint_path = "/home/user/tmp/project/DILAB/A-LLMRec/pre_train/sasrec/Movies_and_TV/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"

# ✅ 1️⃣ 체크포인트 불러오기
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# ✅ 2️⃣ checkpoint 타입 확인 후 언패킹
if isinstance(checkpoint, dict):
    state_dict = checkpoint["model_state"]
    args_dict = checkpoint.get("args", {})
elif isinstance(checkpoint, (list, tuple)) and len(checkpoint) == 2:
    args_dict, state_dict = checkpoint  # 리스트 언패킹
else:
    raise ValueError("Checkpoint format is incorrect. Expected a dictionary or a 2-element list.")

# ✅ 3️⃣ args 설정 (훈련 시 설정값 유지)
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

# ✅ 4️⃣ `user_num`, `item_num`을 체크포인트에 맞게 조정
user_num = args_dict.get("user_num", 311143)  # 데이터셋에 따라 유동적으로 변경
item_num = args_dict.get("item_num", 86678)

# ✅ 5️⃣ SASRec 모델 생성
sasrec_model = SASRec(user_num=user_num, item_num=item_num, args=args)

# ✅ 6️⃣ 가중치 로드 (strict=False로 설정하여 불일치 무시)
sasrec_model.load_state_dict(state_dict, strict=False)

# ✅ 7️⃣ 모델 평가 모드 설정
sasrec_model.eval()

print(f"SASRec 모델 로드 완료! (User_num: {user_num}, Item_num: {item_num})")
print(f"모델 가중치 크기: {sasrec_model.item_emb.weight.shape}")

# ✅ 8️⃣ 데이터셋 로드
df = pd.read_csv("data/amazon/meta_Movies_and_TV.csv")

# ✅ 9️⃣ SBERT 모델 & 영화 임베딩 로드
sbert_model = SentenceTransformer("paraphrase-mpnet-base-v2")
with open("data/sbert_movie_embeddings.pkl", "rb") as f:
    movie_embeddings = pickle.load(f)

# ✅ 10️⃣ 추천 시스템 함수
def hybrid_recommend(user_id, query, top_k=10):
    """
    사용자의 협업 필터링 추천 + 검색 기반 추천을 결합한 최종 추천 함수
    """
    # 1️⃣ **검색 기반 추천 (SBERT)**
    query_embedding = sbert_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], movie_embeddings)[0]
    content_ranking = np.argsort(similarities)[-top_k:][::-1]

    # 2️⃣ **협업 필터링 추천 (SASRec)**
    user_tensor = torch.tensor([user_id])
    item_indices = torch.arange(1, item_num + 1).unsqueeze(0)  # ✅ 모든 아이템 대상으로 예측

    with torch.no_grad():  # no_grad()로 그래디언트 저장 방지
        cf_scores = sasrec_model.predict(user_tensor, torch.randint(1, item_num, (1, args.maxlen)), item_indices)
        cf_scores = cf_scores.numpy().flatten()

    cf_ranking = np.argsort(cf_scores)[-top_k:][::-1]

    # 3️⃣ **하이브리드 스코어 계산**
    hybrid_scores = {}
    for idx in content_ranking:
        hybrid_scores[idx] = 0.7 * similarities[idx] + 0.3 * cf_scores[idx]
    
    for idx in cf_ranking:
        if idx in hybrid_scores:
            hybrid_scores[idx] += 0.7 * cf_scores[idx] + 0.3 * similarities[idx]
        else:
            hybrid_scores[idx] = 0.7 * cf_scores[idx] + 0.3 * similarities[idx]

    # 4️⃣ **최종 상위 추천 영화 반환**
    top_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_k]
    return df.iloc[top_indices][["title", "description"]]

# ✅ 11️⃣ 테스트 실행
print(hybrid_recommend(user_id=123, query="sci-fi space adventure"))
