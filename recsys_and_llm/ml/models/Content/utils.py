from pymongo import MongoClient
from tqdm import tqdm


def get_item_data():
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "items"
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    item_collection = db["item"]

    item2info = {}
    errors = []

    for item_data in tqdm(item_collection.find()):
        item_id = str(item_data["_id"])
        try:
            item2info[item_id] = ""
            item2info[item_id] += (
                f'category : {", ".join(item_data["category"])} | '
                if "category" in item_data.keys() and item_data["category"]
                else ""
            )
            item2info[item_id] += (
                f'description : {", ".join(item_data["description"])} | '
                if "description" in item_data.keys() and item_data["description"]
                else ""
            )
            item2info[item_id] += (
                f'title : {item_data["title"]} | '
                if "title" in item_data.keys()
                else ""
            )
            item2info[item_id] += (
                f'cast : {", ".join(item_data["cast"])} | '
                if "cast" in item_data.keys() and item_data["cast"]
                else ""
            )
            item2info[item_id] = item2info[item_id].rstrip()[:-1].rstrip()

        except Exception as e:
            errors.append([item_id, e])

    texts = []
    for k in sorted(item2info.keys()):
        texts.append(item2info[k])

    return texts
