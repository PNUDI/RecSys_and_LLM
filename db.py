import pickle

from pymongo import MongoClient

# MongoDB 서버에 연결
client = MongoClient("mongodb://localhost:27017/")
# 데이터베이스와 컬렉션 접근
db = client["items"]

item = db["item"]
user = db["user"]
review = db["review"]


# 각 컬렉션에서 첫 5개 문서 조회
print("📌 Items Collection:")
for doc in item.find().limit(5):  # 첫 5개 문서 출력
    print(doc)

# print("\n📌 Users Collection:")
# for doc in user.find().limit(1):
#     print(doc)

# print("\n📌 Reviews Collection:")
# for doc in review.find().limit(1):
#     print(doc)
