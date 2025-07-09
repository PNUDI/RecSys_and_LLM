import argparse
import json

from data_preprocess import *
from db import *
from pymongo import MongoClient


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_preprocess", default=True, type=str2bool)
    parser.add_argument("--db_insert", default=True, type=str2bool)
    args = parser.parse_args()

    fname = "Movies_and_TV"
    if args.data_preprocess:
        preprocess(fname)
    if args.db_insert:
        insert_json_with_item()
        insert_json_with_review()
        insert_json_with_user()
        insert_json_with_imdb()
