import importlib

import numpy as np
import torch
from config import GSASRecExperimentConfig
from dataset_utils import get_num_items
from gsasrec import GSASRec


def load_config(config_file: str) -> GSASRecExperimentConfig:
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def build_model(config: GSASRecExperimentConfig):
    num_items = get_num_items(config.dataset_name)
    model = GSASRec(
        num_items,
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
    )
    return model


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device


def recommend_top5(model, user_seq, user_id, args, text_name_dict):

    user_sequence = user_seq[user_id]

    if len(user_sequence) < 1:
        print(f"User {user_id} has no sequence data.")
        return []

    seq = np.zeros([args.sequence_length], dtype=np.int32)

    idx = args.sequence_length - 1

    for i in reversed(user_sequence):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break
    device = get_device()
    seq = torch.tensor(seq, dtype=torch.long)
    predictions_num, predictions_score = model.get_predictions(seq, 20)  # 반환값 분리
    predictions_num = predictions_num.squeeze(0)

    # predictions = predictions[0]
    # print(predictions)
    # print(predictions[50671])
    # Top-5 아이템 인덱스 추출
    # print(top5_idx)
    i = 0
    top5_titles = []
    for idx in predictions_num:
        if len(top5_titles) == 5:
            break
        if int(idx.item()) + 1 in text_name_dict["title"]:
            top5_titles.append(text_name_dict["title"][int(idx.item()) + 1])

    # top5_titles = [text_name_dict['title'][int(idx.item()) + 1] for idx in top5_idx]

    return top5_titles
