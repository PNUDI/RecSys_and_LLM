import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

from .model import TiSASRec


def tisasrec_initialize_model(args, usernum, itemnum):
    model = TiSASRec(usernum, itemnum, itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass  # Ignore initialization errors
    return model


def computeRePos(time_seq, time_span):

    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:  # float as map key?
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    reverse_item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1
        reverse_item_map[i + 1] = item
    # print(item_map[9284])
    # print(reverse_item_map[8981])
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(
            map(lambda x: [item_map[x[0]], time_map[x[1]]], items)
        )

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(
            map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items)
        )
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return (
        User_res,
        len(user_set),
        len(item_set),
        max(time_max),
        item_map,
        reverse_item_map,
    )


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)

    print("Preparing data...")
    f = open("ML/data/%s.txt" % fname, "r")
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    aa = 0
    for line in f:
        try:
            u, i, timestamp = line.rstrip().split("\t")
        except:
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1

    f.close()
    f = open("ML/data/%s.txt" % fname, "r")  # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, timestamp = line.rstrip().split("\t")
        except:
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)

        # print(item_count[1])
        # if user_count[u]<5 or item_count[i]<5: # hard-coded
        # continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])

    f.close()
    time_map = timeSlice(time_set)
    # print(User[1])
    User, usernum, itemnum, timenum, item_map, reverse_item_map = cleanAndsort(
        User, time_map
    )

    print("Preparing done...")
    return [User, usernum, itemnum, timenum, item_map, reverse_item_map]


def tisasrec_recommend_top5(args, model, user_id, user_sequence, missing_list):

    if len(user_sequence) < 1:
        return []

    seq = np.zeros([args.maxlen], dtype=np.int32)
    time_seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1

    for i in reversed(user_sequence):
        seq[idx] = i[0]
        time_seq[idx] = i[1]
        idx -= 1
        if idx == -1:
            break

    item_idx = list(range(1, args.itemnum + 1))

    time_matrix = computeRePos(time_seq, args.time_span)

    predictions = -model.predict(
        *[np.array(l) for l in [[user_id], [seq], [time_matrix], item_idx]]
    )
    predictions = predictions[0]
    for value in seq:
        predictions[value] = float("-inf")
    # print(predictions[50671])
    # Top-5 아이템 인덱스 추출
    # print(len(predictions))

    top5_idx = predictions.argsort(descending=True)[:20]
    # print(top5_idx)
    i = 0

    top5_num = []
    for idx in top5_idx:
        if len(top5_num) == 8:
            break
        if int(idx.item()) in missing_list:
            continue
        top5_num.append(int(idx.item()))

    return top5_num
