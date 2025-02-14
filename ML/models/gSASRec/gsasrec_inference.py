import numpy as np
import torch

from .gsasrec import *


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device


def build_model(config):
    model = GSASRec(
        num_items = config.itemnum,
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
    )
    return model


def gsasrec_recommend_top5(model, user_sequence, user_id, args, text_name_dict):
    
    if len(user_sequence) < 1:
        print(f"User {user_id} has no sequence data.")
        return []

    seq = np.zeros([args.sequence_length], dtype=np.int32)

    idx = args.sequence_length - 1

    for i in reversed(user_sequence):
        seq[idx] = i[0]
        idx -= 1
        if idx == -1:
            break
    device = get_device()
    model = model.to(args.device)
    seq = torch.tensor(seq, dtype=torch.long)
    seq = seq.to(args.device)
    predictions_num, predictions_score = model.get_predictions(seq, 20)  # 반환값 분리
    predictions_num = predictions_num.squeeze(0)

    # predictions = predictions[0]
    # print(predictions)
    # print(predictions[50671])
    # Top-5 아이템 인덱스 추출
    # print(top5_idx)
    i = 0
    top5_titles = []
    top5_num = []
    for idx in predictions_num:
        if len(top5_titles) == 8:
            break
        if int(idx.item()) + 1 in text_name_dict["title"]:
            top5_titles.append(text_name_dict["title"][int(idx.item()) + 1])
            top5_num.append(int(idx.item()) + 1)
    # top5_titles = [text_name_dict['title'][int(idx.item()) + 1] for idx in top5_idx]

    return top5_titles, top5_num
