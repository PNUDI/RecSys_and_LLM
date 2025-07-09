import json
import os
import random
import re
import sys
from argparse import Namespace
from collections import Counter

import torch
from dotenv import load_dotenv
from pymongo import MongoClient

# from recommenders.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

from recsys_and_llm.backend.app.config import ALL_GENRES
from recsys_and_llm.backend.app.inference import (
    genre_inference,
    inference,
    item_content_inference,
)
from recsys_and_llm.ml.models.model_manager import ModelManager
from recsys_and_llm.ml.utils import (
    calculate_genre_distribution,
    find_cold,
    get_missing,
    get_text_name_dict,
)

sys.path.append(sys.path[0] + "/recommenders")
""" 주석처리 """
# from models.base.utils import *
# from util import DataIterator, argmax_top_k, typeassert
# from util.cython.tools import float_type, is_ndarray
""" 주석처리 """
sys.path.remove(sys.path[0] + "/recommenders")


class abstract_arena:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.val_users = args.val_users
        self.val_ratio = args.val_ratio
        self.simulation_name = args.simulation_name
        self.device = torch.device(args.cuda)
        self.n_avatars = args.n_avatars
        self.modeltype = args.modeltype
        self.execution_mode = args.execution_mode
        self.rec_gt = args.rec_gt
        print("============================")

    def excute(self):
        """
        The whole process of the simulation
        """
        self.prepare_dir()
        self.load_data()
        self.load_recommender_and_db()
        self.initialize_all_avatars()
        self.get_block_recommendations()

        if self.val_users:
            self.validate_all_avatars()
        else:
            self.simulate_all_avatars()
            self.save_results()

    def prepare_dir(self):
        # make dir
        def ensureDir(dir_path):
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        self.storage_base_path = (
            f"storage/{self.dataset}/{self.modeltype}/" + self.simulation_name
        )
        ensureDir(self.storage_base_path)
        # ensureDir(self.storage_base_path + "/avatars")
        ensureDir(self.storage_base_path + "/running_logs")
        ensureDir(self.storage_base_path + "/rankings")
        # ensureDir(self.storage_base_path + "/new_train")
        if os.path.exists(self.storage_base_path + "/system_log.txt"):
            os.remove(self.storage_base_path + "/system_log.txt")

    def load_data(self):
        """
        load the data for simulation
        """
        self.movie_detail = pd.read_csv(
            f"datasets/{self.dataset}/simulation/item_detail.csv"
        )

    def load_recommender_and_db(self):
        load_dotenv()
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client[os.getenv("DB_NAME")]
        user_collection = db["user"]
        item_collection = db["item"]

        self.user_collection = user_collection
        self.item_collection = item_collection

        # 모델 사용 데이터 파싱
        cold_items = find_cold(user_collection, 50)
        text_name_dict = get_text_name_dict(item_collection)
        missing_list = get_missing(text_name_dict["title"])
        global_genre_distribution = calculate_genre_distribution(
            item_collection, ALL_GENRES
        )

        data = [cold_items, text_name_dict, missing_list, global_genre_distribution]

        # 모델 로드
        model_manager = ModelManager(data)
        self.model_manager = model_manager

    def prepare_batch_inputs(self, user_data_list, model_manager):
        """
        여러 유저 데이터를 받아 inference 함수에 넣을 수 있는 형식으로 변환.
        """
        inputs = []

        for user_data in user_data_list:
            user_id = user_data["_id"]
            seq = [item["itemnum"] for item in user_data.get("items", [])]
            seq_time = [
                (item["itemnum"], item["unixReviewTime"])
                for item in user_data.get("items", [])
            ]

            # 유저의 시청 장르 추출
            watched_genres = [
                genre
                for item in user_data["items"]
                if "predicted_genre" in item
                for genre in item["predicted_genre"]
            ]
            user_genre_counts = Counter(watched_genres)
            genre = genre_inference(model_manager, user_genre_counts)

            # 해당 장르에 속하는 아이템 ID 목록
            genre_movie_ids = [
                int(movie["_id"])
                for movie in self.item_collection.find(
                    {"predicted_genre": genre}, {"_id": 1}
                )
            ]

            inputs.append(
                {
                    "user_id": user_id,
                    "seq": seq,
                    "seq_time": seq_time,
                    "genre": genre,
                    "genre_movie_ids": genre_movie_ids,
                }
            )

        return inputs

    def get_block_recommendations(self, filename="full_rankings", batch_size=512):
        """
        document the full rankings of the items,
        according to a specific cf model
        """
        # if(os.path.exists(self.storage_base_path + '/{}_{}.npy'.format(filename, self.n_avatars))):
        #     print("loading full rankings from storage")
        #     self.full_rankings = np.load(self.storage_base_path + '/{}_{}.npy'.format(filename, self.n_avatars))
        #     print("finish loading full rankings")
        #     print(type(self.full_rankings))
        # else:
        # dump_dict = merge_user_list([self.data.train_user_list,self.data.valid_user_list])
        """
        일단 주석처리
        """
        # print("nodrop?", self.data.nodrop)
        # # @ Use valid data for simulation.
        # if self.data.nodrop:
        #     dump_dict = merge_user_list(
        #         [self.data.train_nodrop_user_list, self.data.test_user_list]
        #     )
        # else:
        #     dump_dict = merge_user_list(
        #         [self.data.train_user_list, self.data.test_user_list]
        #     )

        """
        일단 주석처리
        """
        # dump_dict = merge_user_list([self.data.train_user_list, self.data.test_user_list])

        user_data_list = list(
            self.user_collection.find({"_id": {"$in": self.simulated_avatars_id}})
        )
        model_inputs = self.prepare_batch_inputs(user_data_list, self.model_manager)
        # 결과 저장 딕셔너리 초기화
        self.block_recommendations = {}

        for input_data in model_inputs:
            res = inference(
                self.model_manager,
                input_data["user_id"],
                input_data["seq"],
                input_data["seq_time"],
                input_data["genre_movie_ids"],
            )
            self.block_recommendations[input_data["user_id"]] = {
                "Top Choice for You": [res["allmrec_prediction"]],  # 1개
                "Recommended for You": res["gsasrec_prediction"],  # 8개
                "Similar movies to what you've watched recently": res[
                    "tisasrec_prediction"
                ],  # 8개
                f"Movies in the {input_data['genre']} genre that you like": res[
                    "genrerec_prediction"
                ],  # 8개
            }

        print("finish block recommendations")
        # breakpoint()

    def initialize_all_avatars(self):
        """
        initialize all avatars
        """
        self.simulated_avatars_id = list(map(str, range(1, self.n_avatars + 1)))

        # all_avatars = sorted(list(self.data.test_user_list.keys()))
        # self.simulated_avatars_id = all_avatars[:self.n_avatars]
        # random.seed(self.args.seed)
        # self.simulated_avatars_id = sorted(random.sample(all_avatars, self.n_avatars))

        print("simulated avatars", self.simulated_avatars_id)

    def validate_all_avatars(self):
        """
        validate the users
        """
        raise NotImplementedError

    def simulate_all_avatars(self):
        """
        excute the simulation for all avatars
        """
        raise NotImplementedError

    def simulate_one_avatar(self):
        """
        excute the simulation for one avatar
        """
        raise NotImplementedError

    def save_results(self):
        """
        save the results of the simulation
        """
        raise NotImplementedError

    def load_additional_info(self):
        """
        load additional information for the simulation
        """
        pass
