# DB

### How to set DB

### DataSet

```
cd DB
cd data
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Movies_and_TV.json.gz  # download review dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz  # download metadata
gzip -d meta_Movies_and_TV.json.gz
```
### Download and make dir

```
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-8.0.4.tgz
tar -xvzf mongodb-linux-x86_64-ubuntu2204-8.0.4.tgz
mkdir -p ~/mongodb/data/db
mkdir -p ~/mongodb/logs
wget https://downloads.mongodb.com/compass/mongosh-2.3.9-linux-x64.tgz
tar -xvzf mongosh-2.3.9-linux-x64.tgz
```

### Run MongoDB and connect DataBase

```
./mongodb-linux-x86_64-ubuntu2204-8.0.4/bin/mongod --dbpath ~/mongodb/data/db --logpath ~/mongodb/logs/mongod.log --port 27017 --bind_ip_all # namho
./mongodb-linux-x86_64-ubuntu2204-8.0.4/bin/mongod --dbpath ~/mongodb/data/db --logpath ~/mongodb/logs/mongod.log --port 27017 # other user
cd mongosh-2.3.9-linux-x64
cd bln
./mongosh
```

### Preprocess and insert DB

```
cd DB
python main.py
```

## User Document Structure
```
{
  "_id": "1", # UserNum
  "reviewerID": "A3478QRKQDOPQ2", # reviewerID
  "password": "1234", # password default = 1234
  "items": [ # Item sequnce
    {
      "itemnum": 9284, # ItemNum
      "asin": "B00005155P",
      "reviewText": "it sucks, but you can't watch the regular movie, at least if you get t…",
      "overall": 1, # Rating
      "summary": "It isn't really the film, it is a review commentary of the film",
      "unixReviewTime": 1316649600
    },
    {
      "itemnum": ..., # next item
      "asin": "...",
      "reviewText": "...",
      "overall": ...,
      "summary": "...",
      "unixReviewTime": ...
    },
    ...
  ]
}
```
## Item Document Structure
```
{
  "_id": "1", # ItemNum
  "category": [
    "Movies & TV",
    "Art House & International",
    "By Original Language",
    "Spanish"
  ],
  "tech1": "",
  "description": [],
  "fit": "",
  "title": "Peace Child VHS",
  "also_buy": [],
  "tech2": "",
  "brand": "",
  "feature": [],
  "rank": "866,012 in Movies & TV (",
  "also_view": [],
  "main_cat": "Movies & TV",
  "similar_item": "",
  "date": "",
  "price": "",
  "asin": "0001527665",
  "imageURL": [], # original data
  "imageURLHighRes": []
  "cast": [
            "Don Richardson", # imdb data cast
            "Rolf Forsberg"
        ],
  "director": "Rolf Forsberg", #imdb data director
  "poster_url": "aaaa.jpg" # imdb poster_url
}
```
## Review Document Structure
```
{
  "_id": "1", # ItemNum
  "review": {
    "1": "really happy they got evangelised .. spoiler alert==happy ending liked…", # UserNum: Review
    "237431": "The movie was a good synopsis of the main parts of the book. Just shor…"
  },
  "summary": {
    "1": "great", # UserNum: Summary
    "237431": "Synopsis"
  }
}
```
