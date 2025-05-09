# model_loader.py
import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import pipeline

from recsys_and_llm.ml.models.ALLMRec.a_llmrec_model import A_llmrec_model
from recsys_and_llm.ml.models.gSASRec.gsasrec_inference import build_model
from recsys_and_llm.ml.models.TiSASRec.TiSASRec_inference import TiSASRec


class ModelLoader:
    def __init__(self, llmrec_args):
        self.llmrec_args = llmrec_args
        self.tisasrec_args = None
        self.gsasrec_args = None

    @property
    def _load_allmrec(self):  # cuda:0
        """ALLMRec 모델 로드 및 초기화"""
        allmrec_model = A_llmrec_model(self.llmrec_args).to(self.llmrec_args.device)
        phase1_epoch = 10
        phase2_epoch = 10
        allmrec_model.load_model(
            self.llmrec_args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch
        )
        allmrec_model.eval()
        return allmrec_model

    @property
    def _load_tisasrec(self):  # auto
        """TiSASRec 모델 로드"""
        repo_id = "PNUDI/TiSASRec"
        model_file = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", repo_type="model"
        )
        config_file = hf_hub_download(
            repo_id=repo_id, filename="config.json", repo_type="model"
        )
        with open(config_file, "r") as f:
            config_data = json.load(f)
        self.tisasrec_args = argparse.Namespace(**config_data)
        tisasrec_model = TiSASRec(
            self.tisasrec_args.usernum,
            self.tisasrec_args.itemnum,
            self.tisasrec_args.itemnum,
            self.tisasrec_args,
        ).to(self.tisasrec_args.device)
        tisasrec_model.load_state_dict(
            torch.load(model_file, map_location=self.tisasrec_args.device)
        )
        tisasrec_model.eval()
        return tisasrec_model

    @property
    def _load_gsasrec(self):  # cuda:0
        """gSASRec 모델 로드"""
        repo_id = "PNUDI/gSASRec"
        model_file = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", repo_type="model"
        )
        config_file = hf_hub_download(
            repo_id=repo_id, filename="config.json", repo_type="model"
        )
        with open(config_file, "r") as f:
            config_data = json.load(f)
        self.gsasrec_args = argparse.Namespace(**config_data)
        gsasrec_model = build_model(self.gsasrec_args)
        gsasrec_model.load_state_dict(torch.load(model_file, map_location="cpu"))
        gsasrec_model.eval()
        return gsasrec_model

    @property
    def _load_contentrec(self):
        """모델 로드"""
        item_contents_emb = torch.load(
            hf_hub_download(
                repo_id="PNUDI/Item_based",
                filename="item_text_emb_SB.pt",
                repo_type="model",
            ),
            weights_only=False,
        )
        return item_contents_emb

    @property
    def _load_genrerec(self):  # auto
        genrerec_model = pipeline(
            "text-generation",
            model="unsloth/phi-4-unsloth-bnb-4bit",
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )
        return genrerec_model
