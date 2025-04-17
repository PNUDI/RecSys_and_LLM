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
from recommenders.data import Data

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
from models.base.utils import *
from util import DataIterator, argmax_top_k, typeassert
from util.cython.tools import float_type, is_ndarray

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
        self.items_per_page = args.items_per_page
        self.execution_mode = args.execution_mode
        self.rec_gt = args.rec_gt
        self.model_path = (
            "recommenders/weights/"
            + args.dataset
            + "/"
            + args.modeltype
            + "/"
            + args.model_path
        )
        print("============================")
        print(self.model_path)

    def excute(self):
        """
        The whole process of the simulation
        """
        self.load_saved_args(self.model_path)
        self.prepare_dir()
        self.load_data()
        self.load_recommender_and_db()
        self.initialize_all_avatars()
        self.get_block_recommendations()

        self.get_full_rankings()
        self.load_additional_info()
        if self.val_users:
            self.validate_all_avatars()
        else:
            self.simulate_all_avatars()
            self.save_results()

    def load_saved_args(self, model_path):
        """
        load the recommender args, which is saved when training the recommender
        """
        self.saved_args = Namespace()
        # If the path exists, read.
        if os.path.exists(model_path + "/args.txt"):
            with open(model_path + "/args.txt", "r") as f:
                self.saved_args.__dict__ = json.load(f)
        else:
            with open("recommenders/weights/default_args.txt", "r") as f:
                self.saved_args.__dict__ = json.load(f)
        # View current directory.
        self.saved_args.data_path = "datasets/"  # Modify the table of contents.
        self.saved_args.dataset = self.dataset
        self.saved_args.cuda = self.args.cuda
        self.saved_args.modeltype = self.modeltype
        # self.saved_args.nodrop = self.args.nodrop

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
        sys.path.append(sys.path[0] + "/recommenders")
        try:
            exec(
                "from recommenders.models."
                + self.saved_args.modeltype
                + " import "
                + self.saved_args.modeltype
                + "_Data"
            )  # load special dataset
            print(
                "from recommenders.models."
                + self.saved_args.modeltype
                + " import "
                + self.saved_args.modeltype
                + "_Data"
            )
            self.data = eval(self.saved_args.modeltype + "_Data(self.saved_args)")
        except:
            print("no special dataset")
            self.data = Data(self.saved_args)  # load data from the path
            print("finish loading data")
        sys.path.remove(sys.path[0] + "/recommenders")
        # import pickle
        # with open(f'datasets/{self.dataset}/simulation/movie_dict.pkl', 'rb') as f:
        #     self.movie_detail = pickle.load(f)
        self.movie_detail = pd.read_csv(
            f"datasets/{self.dataset}/simulation/movie_detail.csv"
        )

    def load_recommender(self):
        """
        load the recommender for simulation
        """
        sys.path.append(sys.path[0] + "/recommenders")
        self.running_model = self.saved_args.modeltype
        exec(
            "from recommenders.models."
            + self.saved_args.modeltype
            + " import "
            + self.running_model
        )  # import the model first
        self.model = eval(
            self.running_model + "(self.saved_args, self.data)"
        )  # initialize the model with the graph
        self.model.cuda(self.device)
        print("finish generating recommender")
        sys.path.remove(sys.path[0] + "/recommenders")

        # load the checkpoint
        def restore_checkpoint(model, checkpoint_dir, device):
            """
            If a checkpoint exists, restores the PyTorch model from the checkpoint.
            Returns the model and the current epoch.
            """
            cp_files = [
                file_
                for file_ in os.listdir(checkpoint_dir)
                if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
            ]
            if not cp_files:
                print("No saved model parameters found")
            epoch_list = []
            regex = re.compile(r"\d+")
            for cp in cp_files:
                epoch_list.append([int(x) for x in regex.findall(cp)][0])
            loading_epoch = max(epoch_list)

            filename = os.path.join(
                checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(loading_epoch)
            )
            # print("Loading from checkpoint {}?".format(filename))

            checkpoint = torch.load(filename, map_location=str(device))
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> Successfully restored checkpoint (trained for {} epochs)".format(
                    checkpoint["epoch"]
                )
            )

            return model, loading_epoch

        if self.args.modeltype != "Random" and self.args.modeltype != "Pop":
            print("loading checkpoint")
            self.model, self.loading_epoch = restore_checkpoint(
                self.model, self.model_path, self.device
            )  # restore the checkpoint
        # self.model, self.loading_epoch = restore_checkpoint(self.model, self.model_path, self.device) # restore the checkpoint

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

    def get_full_rankings(self, filename="full_rankings", batch_size=512):
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
        print("nodrop?", self.data.nodrop)
        # @ Use valid data for simulation.
        if self.data.nodrop:
            dump_dict = merge_user_list(
                [self.data.train_nodrop_user_list, self.data.test_user_list]
            )
        else:
            dump_dict = merge_user_list(
                [self.data.train_user_list, self.data.test_user_list]
            )
        # dump_dict = merge_user_list([self.data.train_user_list, self.data.test_user_list])
        score_matrix = np.zeros((len(self.simulated_avatars_id), self.data.n_items))
        simulated_avatars_iter = DataIterator(
            self.simulated_avatars_id,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        for batch_id, batch_users in tqdm(enumerate(simulated_avatars_iter)):
            ranking_score = self.model.predict(batch_users, None)  # (B,N)
            if not is_ndarray(ranking_score, float_type):
                ranking_score = np.array(ranking_score, dtype=float_type)
            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.

            for idx, user in enumerate(batch_users):
                dump_items = dump_dict[user]
                # dump_items = [ x for x in dump_items if not x in self.data.test_user_list[user] ]
                ranking_score[idx][dump_items] = -np.inf

                score_matrix[batch_id * batch_size + idx] = ranking_score[idx]

            print("finish recommend one batch", batch_id)
            # break

        print("finish generating score matrix")
        self.full_rankings = np.argsort(-score_matrix, axis=1)
        if self.rec_gt:
            # for user in self.simulated_avatars_id:
            #     for idx, item in enumerate(self.data.train_user_list[user]):
            #         self.full_rankings[user][idx] = item
            gt_dict = pd.read_pickle("scripts/user_ground_truth.pkl")
            for user in self.simulated_avatars_id:
                for idx, item in enumerate(gt_dict[user]):
                    self.full_rankings[user][idx] = item
        np.save(
            self.storage_base_path
            + "/rankings/"
            + "/{}_{}.npy".format(filename, self.n_avatars),
            self.full_rankings,
        )

        print("finish get full rankings")

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
        print("nodrop?", self.data.nodrop)
        # @ Use valid data for simulation.
        if self.data.nodrop:
            dump_dict = merge_user_list(
                [self.data.train_nodrop_user_list, self.data.test_user_list]
            )
        else:
            dump_dict = merge_user_list(
                [self.data.train_user_list, self.data.test_user_list]
            )
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
                "top_pick": res["allmrec_prediction"],  # 1개
                "personalized": res["gsasrec_prediction"],  # 8개
                "recent": res["tisasrec_prediction"],  # 8개
                "genre": res["genrerec_prediction"],  # 8개
            }

        print("finish block recommendations")
        breakpoint()

        # self.full_rankings = np.argsort(-score_matrix, axis=1)
        # if self.rec_gt:
        #     # for user in self.simulated_avatars_id:
        #     #     for idx, item in enumerate(self.data.train_user_list[user]):
        #     #         self.full_rankings[user][idx] = item
        #     gt_dict = pd.read_pickle("scripts/user_ground_truth.pkl")
        #     for user in self.simulated_avatars_id:
        #         for idx, item in enumerate(gt_dict[user]):
        #             self.full_rankings[user][idx] = item
        # np.save(
        #     self.storage_base_path
        #     + "/rankings/"
        #     + "/{}_{}.npy".format(filename, self.n_avatars),
        #     self.full_rankings,
        # )

        print("finish get full rankings")

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

    def page_generator(self):
        """
        generate one page items for one avatar
        """
        raise NotImplementedError

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
