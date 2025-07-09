from pymongo import MongoClient

# MongoDB 서버에 연결
client = MongoClient("mongodb://localhost:27017/")

# 데이터베이스와 컬렉션 접근
db = client["items"]
collection = db["item"]

# 데이터 조회
print(collection.find_one())
