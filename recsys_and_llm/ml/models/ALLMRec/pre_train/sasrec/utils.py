import copy
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np
import torch
from pytz import timezone
from torch.utils.data import Dataset

# sampler for batch generation


# 부정 예제 생성 함수 s 집합에 없으며 l~r 사이에 있는 random한 숫자 하나
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(
    user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED
):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    # 모델 학습 과정은 배치 단위로 데이터를 계속 요청한다
    # 큐(result_queue)는 고정된 크기를 가지며 최대치에 도달하면 새 데이터를 담지 못 하고 대기 상태에 들어감
    # 학습 프로세스가 큐에서 데이터를 소비하면 큐에 여유 공간이 생기고 다시 데이터 샘플링을 한다
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


#
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# DataSet for ddp
class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        print("Initializing with num_user:", num_user)

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = idx + 1
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        nxt = self.user_train[user_id][-1]
        length_idx = self.max_len - 1

        # user의 seq set
        ts = set(self.user_train[user_id])
        for i in reversed(self.user_train[user_id][:-1]):
            seq[length_idx] = i
            pos[length_idx] = nxt
            if nxt != 0:
                neg[length_idx] = random_neq(1, self.num_item + 1, ts)
            nxt = i
            length_idx -= 1
            if length_idx == -1:
                break

        return user_id, seq, pos, neg


class SeqDataset_Inference(Dataset):
    def __init__(self, user_train, user_valid, user_test, use_user, num_item, max_len):
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.num_user = len(use_user)
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user
        print("Initializing with num_user:", self.num_user)

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = self.use_user[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        seq[idx] = self.user_valid[user_id][0]
        idx -= 1
        for i in reversed(self.user_train[user_id]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(self.user_train[user_id])
        rated.add(0)
        pos = self.user_test[user_id][0]
        neg = []
        for _ in range(3):
            t = np.random.randint(1, self.num_item + 1)
            while t in rated:
                t = np.random.randint(1, self.num_item + 1)
            neg.append(t)
        neg = np.array(neg)
        return user_id, seq, pos, neg


# train/val/test data generation
def data_partition(fname, path=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1

    # f = open('./pre_train/sasrec/data/%s.txt' % fname, 'r')
    if path == None:
        f = open("./ML/data/amazon/%s.txt" % fname, "r")
    else:
        f = open(path, "r")
    for line in f:
        u, i = line.rstrip().split(" ")
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)  # id가 곧 유저의 개수
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # 3개 이상이면 0~-3까지 train, -2 valid, -1 test
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
