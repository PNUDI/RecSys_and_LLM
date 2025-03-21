# inference.py
import numpy as np
import torch
import torch.nn.functional as F

from recsys_and_llm.ml.models.gSASRec.gsasrec_inference import gsasrec_recommend_top5
from recsys_and_llm.ml.models.Phi4.user_genre_predict import (
    predict_user_preferred_genres,
)
from recsys_and_llm.ml.models.TiSASRec.TiSASRec_inference import tisasrec_recommend_top5
from recsys_and_llm.ml.utils import seq_preprocess


def inference(model_manager, user_id, seq, seq_time, genre_movie_ids):
    # ALLMRec 기반 추천
    seq = seq_preprocess(model_manager.llmrec_args.maxlen, seq)
    seq = np.expand_dims(np.array(seq), axis=0)

    with torch.no_grad():
        log_emb = model_manager.allmrec_model.recsys.model(seq, mode="log_only")
    data = [seq, log_emb]
    allmrec_prediction = model_manager.allmrec_model(data, mode="inference")

    # TiSASRec 기반 추천
    tisasrec_prediction = tisasrec_recommend_top5(
        model_manager.tisasrec_args,
        model_manager.tisasrec_model,
        user_id,
        seq_time,
        model_manager.missing_list,
    )

    # gSASRec 기반 추천
    gsasrec_prediction = gsasrec_recommend_top5(
        model_manager.gsasrec_model,
        user_id,
        seq_time,
        model_manager.gsasrec_args,
        model_manager.missing_list,
    )

    # 선호 장르 아이템 SASRec ranking
    genrerec_prediction = model_manager.allmrec_model.recsys.rank_item_by_genre(
        log_emb, genre_movie_ids
    )

    return {
        "allmrec_prediction": allmrec_prediction,
        "tisasrec_prediction": tisasrec_prediction,
        "gsasrec_prediction": gsasrec_prediction,
        "genrerec_prediction": genrerec_prediction,
    }


def item_content_inference(model_manager, item_id):
    item_contents_emb = torch.tensor(
        model_manager.contentrec_model, device=model_manager.device
    )
    target_emb = item_contents_emb[int(item_id) - 1]
    cosine_sim = F.cosine_similarity(target_emb, item_contents_emb)
    topk = torch.topk(cosine_sim, 21).indices  # 0~
    prediction = []

    for ele in topk:
        ele = ele.item() + 1
        if ele not in model_manager.missing_list and ele != int(item_id):
            prediction.append(str(ele))

    return prediction[:8]


def genre_inference(model_manager, user_genre_counts, k=3):
    candidate_genres = set(user_genre_counts)
    if len(candidate_genres) < 3:
        k = len(candidate_genres)

    sorted_user_genre_counts = sorted(
        user_genre_counts.keys(),
        key=lambda genre: user_genre_counts[genre],
        reverse=True,
    )

    sorted_watched_genres_by_rarity = sorted(
        candidate_genres,
        key=lambda genre: model_manager.global_genre_distribution.get(genre, 1),
        reverse=False,
    )

    user_genre = predict_user_preferred_genres(
        model_manager.genrerec_model,
        candidate_genres,
        sorted_user_genre_counts,
        sorted_watched_genres_by_rarity,
        k=k,
    )

    return user_genre
