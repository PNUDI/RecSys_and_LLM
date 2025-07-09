import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"

client = MongoClient(MONGO_URI)
db = client["items"]
user = db["user"]
item = db["item"]


def load_data(fname, max_len, path=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    data = {}

    # assume user/item index starting from 1

    # f = open('./pre_train/sasrec/data/%s.txt' % fname, 'r')
    if path == None:
        f = open("./data/amazon/%s.txt" % fname, "r")
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
        seq = [0] * max_len
        length_idx = max_len - 1

        for i in reversed(User[user][:-1]):
            seq[length_idx] = i
            length_idx -= 1
            if length_idx == -1:
                break

        data[user] = seq

    return data, User


def find_missing_numbers(numbers, start, end):
    # 전체 범위 집합
    full_set = set(range(start, end + 1))

    # 리스트에서 실제로 있는 숫자 집합
    number_set = set(numbers)

    # 빠진 숫자 찾기
    missing_numbers = sorted(full_set - number_set)

    return missing_numbers


# Missing item을 포함하는 유저와 아이템 찾기
def find_users_with_missing_items(dataset, missing_items):
    users_with_missing_items = defaultdict(list)

    # 각 유저의 시퀀스를 순회
    for user, sequence in dataset.items():
        # 시퀀스와 missing item의 교집합 찾기
        intersecting_items = set(sequence) & set(missing_items)
        if intersecting_items:
            # 교집합이 있다면 해당 유저와 아이템 저장
            users_with_missing_items[user] = intersecting_items

    return users_with_missing_items


def calculate_seq_distribution(dataset):
    # all_dataset 넣어야 됨
    # Step 1: Calculate sequence lengths
    seq_lengths = [len(items) for items in dataset.values()]

    # Step 2: Count users for each sequence length
    length_counts = defaultdict(int)
    for length in seq_lengths:
        length_counts[length] += 1

    # Step 3: Convert to sorted lists for visualization
    sorted_lengths = sorted(length_counts.items())  # (seq_length, user_count)

    # Print the distribution
    print("Sequence Length Distribution:")
    for length, count in sorted_lengths:
        print(f"Length {length}: {count} users")


def warm_and_cold(dataset):
    item_interactions = []
    for user, items in dataset.items():
        item_interactions.extend(items)

    # Step 2: Count occurrences of each item
    item_counts = Counter(item_interactions)

    # Step 3: Sort items by interaction count
    sorted_items = sorted(
        item_counts.items(), key=lambda x: x[1], reverse=True
    )  # (item, count)

    # Step 4: Calculate thresholds for warm and cold items
    total_items = len(sorted_items)
    warm_threshold = int(total_items * 0.35)  # Top 35%
    cold_threshold = int(total_items * 0.35)  # Bottom 35%

    # Get cold items (bottom 35% of interactions)
    cold_items = [item for item, count in sorted_items[-cold_threshold:]]

    # Get warm items (top 35% of interactions)
    warm_items = [item for item, count in sorted_items[:warm_threshold]]

    return warm_items, cold_items


def find_first_user_with_cold_items(dataset, cold_items):
    for user, items in dataset.items():
        # 사용자의 시퀀스 중 cold_items에 포함된 것이 있는지 확인
        if any(item in cold_items for item in items):
            return user


# missing = find_missing_numbers(
#     list(text_name_dict["title"].keys()),
#     list(text_name_dict["description"].keys()),
#     1,
#     86678,
# )
# dataset, all_dataset = load_data("Movies_and_TV", 50)
# # users_with_missing = find_users_with_missing_items(dataset, missing)

# w, c = warm_and_cold(dataset)

# cold_item_included_users = find_first_user_with_cold_items(dataset, c)
# print(cold_item_included_users)

# 결과 출력
# print(len(list(users_with_missing.keys())))
# print(missing)
# print()
# print(users_with_missing[117])
# print()
# print(dataset[117])

print("📌 Items Collection:")
for doc in item.find().limit(5):  # 첫 5개 문서 출력
    print(doc)
