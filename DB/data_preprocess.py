import gzip
import json
import os
import os.path
import pickle
from collections import defaultdict

from tqdm import tqdm


def parse(path):
    print(f"Looking for file at: {os.path.abspath(path)}")
    g = gzip.open(path, "rb")
    for l in tqdm(g):
        yield json.loads(l)


def preprocess(fname):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    file_path = f"data/{fname}.json.gz"

    # counting interactions for each user and item
    for l in parse(file_path):
        line += 1
        asin = l["asin"]
        rev = l["reviewerID"]
        time = l["unixReviewTime"]
        countU[rev] += 1
        countP[asin] += 1

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    meta_dict = dict()
    item_dict = dict()
    user_dict = dict()  # 추가된 사용자 데이터 저장
    User = dict()
    review_dict = {}
    name_dict = {"title": {}, "description": {}}

    f = open(f"data/meta_{fname}.json", "r")
    json_data = f.readlines()
    f.close()
    data_list = [json.loads(line[:-1]) for line in json_data]
    for l in data_list:
        meta_dict[l["asin"]] = l

    for l in parse(file_path):
        line += 1
        asin = l["asin"]
        rev = l["reviewerID"]
        time = l["unixReviewTime"]

        threshold = 5
        if ("Beauty" in fname) or ("Toys" in fname):
            threshold = 4

        if countU[rev] < threshold or countP[asin] < threshold:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
            # 사용자 데이터 초기화
            user_dict[userid] = {
                "reviewerID": rev,
                "password": "1234",  # 비밀번호 고정
                "items": [],  # 아이템 리스트
            }

        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
            if asin in meta_dict:
                item_dict[itemnum] = meta_dict[asin]
        User[userid].append([time, itemid])

        # 사용자별 아이템 데이터 추가
        user_dict[userid]["items"].append(
            {
                "itemnum": itemid,  # 고유 아이템 번호
                "asin": asin,  # 아이템 ID
                "reviewText": l.get("reviewText", ""),  # 리뷰 텍스트
                "overall": l.get("overall", 0.0),  # 평점
                "summary": l.get("summary", ""),  # 요약
                "unixReviewTime": l.get("unixReviewTime", 0),  # 리뷰 시간
            }
        )

        if itemmap[asin] in review_dict:
            try:
                review_dict[itemmap[asin]]["review"][usermap[rev]] = l["reviewText"]
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]["summary"][usermap[rev]] = l["summary"]
            except:
                a = 0
        else:
            review_dict[itemmap[asin]] = {"review": {}, "summary": {}}
            try:
                review_dict[itemmap[asin]]["review"][usermap[rev]] = l["reviewText"]
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]["summary"][usermap[rev]] = l["summary"]
            except:
                a = 0

        try:
            if len(meta_dict[asin]["description"]) == 0:
                name_dict["description"][itemmap[asin]] = "Empty description"
            else:
                name_dict["description"][itemmap[asin]] = meta_dict[asin][
                    "description"
                ][0]
            name_dict["title"][itemmap[asin]] = meta_dict[asin]["title"]
        except:
            a = 0

    # 사용자별 아이템 시간 정렬
    for userid in user_dict.keys():
        user_dict[userid]["items"].sort(
            key=lambda x: x["unixReviewTime"]
        )  # unixReviewTime 기준 정렬

    # 사용자 데이터 저장
    with open(f"data/{fname}_user_dict.json", "w", encoding="utf-8") as user_file:
        json.dump(user_dict, user_file, ensure_ascii=False, indent=4)

    with open(f"data/{fname}_item_dict.json", "w", encoding="utf-8") as meta_file:
        json.dump(item_dict, meta_file, ensure_ascii=False, indent=4)

    with open(f"data/{fname}_review_dict.json", "w", encoding="utf-8") as review_file:
        json.dump(review_dict, review_file, ensure_ascii=False, indent=4)

    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum)
