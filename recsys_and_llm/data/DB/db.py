import json

from pymongo import MongoClient


# MongoDB 연결
def connect_mongodb(clientname, collname):
    client = MongoClient("mongodb://localhost:27017")
    mydb = client[clientname]  # 데이터베이스 이름
    mycoll = mydb[collname]  # 컬렉션 이름
    return mycoll


# JSON 데이터 삽입
def insert_json_with_item():
    mycoll = connect_mongodb("items", "item")

    # JSON 파일 읽기
    with open("data/Movies_and_TV_item_dict.json", "r", encoding="utf-8") as file:
        json_data = json.load(file)  # JSON 파일 로드

    # 각 키와 값을 MongoDB에 삽입
    for key, value in json_data.items():
        # MongoDB 문서에 "_id"로 키를 추가
        value["_id"] = key
        try:
            mycoll.insert_one(value)  # 문서 삽입
            print(f"Document with _id={key} inserted successfully.")
        except Exception as e:
            print(f"Error inserting document with _id={key}: {e}")


def insert_json_with_review():
    mycoll = connect_mongodb("items", "review")

    # JSON 파일 읽기
    with open("data/Movies_and_TV_review_dict.json", "r", encoding="utf-8") as file:
        json_data = json.load(file)  # JSON 파일 로드

    # 각 키와 값을 MongoDB에 삽입
    for key, value in json_data.items():
        # MongoDB 문서에 "_id"로 키를 추가
        value["_id"] = key
        try:
            mycoll.insert_one(value)  # 문서 삽입
            print(f"Document with _id={key} inserted successfully.")
        except Exception as e:
            print(f"Error inserting document with _id={key}: {e}")


def insert_json_with_user():
    mycoll = connect_mongodb("items", "user")

    # JSON 파일 읽기
    with open("data/Movies_and_TV_user_dict.json", "r", encoding="utf-8") as file:
        json_data = json.load(file)  # JSON 파일 로드

    # 각 키와 값을 MongoDB에 삽입
    for key, value in json_data.items():
        # MongoDB 문서에 "_id"로 키를 추가
        value["_id"] = key
        try:
            mycoll.insert_one(value)  # 문서 삽입
            print(f"Document with _id={key} inserted successfully.")
        except Exception as e:
            print(f"Error inserting document with _id={key}: {e}")


def insert_json_with_imdb():
    mycoll = connect_mongodb("items", "item")

    with open("data/merged_imdb_data.json", "r", encoding="utf-8") as file:
        json_data = json.load(file)

    for key, new_data in json_data.items():
        existing_data = mycoll.find_one({"_id": key})
        if existing_data:
            merged_data = {**new_data, **existing_data}
            mycoll.update_one({"_id": key}, {"$set": merged_data})
        else:
            print(key)
            new_data["_id"] = key
            mycoll.insert_one(new_data)
