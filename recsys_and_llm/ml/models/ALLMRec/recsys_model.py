import contextlib
import glob
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

from recsys_and_llm.ml.models.ALLMRec.pre_train.sasrec.model import SASRec
from recsys_and_llm.ml.utils import *


def load_checkpoint(repo_id, recsys, pre_trained):
    api = HfApi()

    # 저장소 내 파일 리스트 가져오기
    file_list = api.list_repo_files(repo_id)

    # 특정 폴더 내 .pth 파일 필터링
    pth_files = [
        f
        for f in file_list
        if f.startswith(f"recsys/{pre_trained}/{recsys}/") and f.endswith(".pth")
    ]

    assert (
        len(pth_files) == 1
    ), "There should be exactly one model file in the directory."

    # 필요한 파일 다운로드 (메모리 버퍼 사용)
    file_name = pth_files[0]
    pth_file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
    kwargs, checkpoint = torch.load(
        pth_file_path, map_location="cpu", weights_only=False
    )
    logging.info("Loaded checkpoint from Hugging Face: %s" % pth_file_path)
    return kwargs, checkpoint


class RecSys(nn.Module):
    def __init__(self, repo_id, recsys_model, pre_trained_data, device):
        super().__init__()
        kwargs, checkpoint = load_checkpoint(repo_id, recsys_model, pre_trained_data)
        kwargs["args"].device = device
        model = SASRec(**kwargs)
        model.load_state_dict(checkpoint)

        for p in model.parameters():
            p.requires_grad = False

        self.item_num = model.item_num
        self.user_num = model.user_num
        self.model = model.to(device)
        self.hidden_units = kwargs["args"].hidden_units

    def forward():
        print("forward")

    def make_candidate_for_LLM(self, itemnum, log_emb, log_seq, args):
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(log_seq):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(seq)
        rated.add(0)

        item_idx = []
        for t in range(1, itemnum + 1):
            if t in rated:
                continue
            item_idx.append(t)

        # predictions = -model.predict(*[np.array(l) for l in [[-1], [seq], item_idx]])
        predictions = -self.model.predict(
            np.array(item_idx),  # candidate items
            user_ids=np.array([-1]),  # user_id placeholder
            log_seqs=np.array([seq]),  # sequence
            log_emb=log_emb,  # 유저 임베딩
        )
        predictions = predictions[0]  # - for 1st argsort DESC

        # Top-K 아이템 선택 (가장 높은 점수 기준)
        top_k = 20
        top_k_indices = predictions.argsort()[
            :top_k
        ]  # 점수가 높은 Top-K 아이템의 인덱스 선택

        # 실제 아이템 번호로 변환
        top_k_items = [
            item_idx[idx] for idx in top_k_indices
        ]  # 인덱스를 아이템 번호로 매핑

        return top_k_items

    def rank_item_by_genre(self, log_emb, genre_movie_ids):
        # predictions = -model.predict(*[np.array(l) for l in [[-1], [seq], item_idx]])
        predictions = -self.model.predict(
            np.array(genre_movie_ids),  # candidate items
            log_emb=log_emb,  # 유저 임베딩
        )
        predictions = predictions[0]  # - for 1st argsort DESC

        # Top-K 아이템 선택 (가장 높은 점수 기준)
        top_k = 8
        top_k_indices = predictions.argsort()[
            :top_k
        ]  # 점수가 높은 Top-K 아이템의 인덱스 선택

        # 실제 아이템 번호로 변환
        top_k_items = [
            genre_movie_ids[idx] for idx in top_k_indices
        ]  # 인덱스를 아이템 번호로 매핑

        return top_k_items
