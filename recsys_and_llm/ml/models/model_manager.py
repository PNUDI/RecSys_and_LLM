# model_manager.py
from argparse import Namespace

from recsys_and_llm.ml.models.model_loader import ModelLoader


class ModelManager:
    def __init__(self, data):
        self.device = "cuda:0"
        self.llmrec_args = Namespace(
            multi_gpu=False,
            gpu_num=0,
            llm="opt",
            recsys="sasrec",
            rec_pre_trained_data="Movies_and_TV",
            pretrain_stage1=False,
            pretrain_stage2=False,
            inference=True,
            batch_size1=32,
            batch_size2=2,
            batch_size_infer=2,
            maxlen=50,
            num_epochs=10,
            stage1_lr=0.0001,
            stage2_lr=0.0001,
            device=self.device,
        )

        cold_items, text_name_dict, missing_list, global_genre_distribution = data

        self.llmrec_args.cold_items = cold_items
        self.llmrec_args.text_name_dict = text_name_dict
        self.llmrec_args.missing_items = missing_list
        self.missing_list = missing_list
        self.global_genre_distribution = global_genre_distribution

        # ModelLoader를 통해 모든 모델을 한 번에 로드
        model_loader = ModelLoader(self.llmrec_args)

        self.allmrec_model = model_loader._load_allmrec
        self.tisasrec_model = model_loader._load_tisasrec
        self.gsasrec_model = model_loader._load_gsasrec
        self.contentrec_model = model_loader._load_contentrec
        self.genrerec_model = model_loader._load_genrerec

        self.tisasrec_args = model_loader.tisasrec_args
        self.gsasrec_args = model_loader.gsasrec_args

    def cleanup(self):
        self.allmrec_model.to("cpu")
        self.tisasrec_model.to("cpu")
        self.gsasrec_model.to("cpu")
        self.contentrec_model = None
        # self.genrerec_model.model.to("cpu")
