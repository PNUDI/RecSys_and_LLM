import io
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast as autocast

from recsys_and_llm.ml.models.ALLMRec.llm4rec import *
from recsys_and_llm.ml.models.ALLMRec.recsys_model import *
from recsys_and_llm.ml.utils import *


class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # fc1: encoder, fc2: decoder
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1


class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        self.cold_items = args.cold_items
        self.missing_items = args.missing_items
        self.text_name_dict = args.text_name_dict
        self.repo_id = "PNUDI/A-LLM"
        self.api = HfApi()

        # pre_trained 모델 불러오기
        self.recsys = RecSys(
            self.repo_id, args.recsys, rec_pre_trained_data, self.device
        )
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768

        # Item Encoder
        self.mlp = two_layer_mlp(self.rec_sys_dim)

        if args.pretrain_stage1 or args.inference:
            self.sbert = SentenceTransformer("nq-distilbert-base-v1")
            # Text Encoder
            self.mlp2 = two_layer_mlp(self.sbert_dim)

        # - Encoder -> Item & Text Embedding을 align 해준다 -> Matching Loss 계산
        # - 기존 Embedding & Encoder-Decoder 거친 Embedding -> Reconstructuion Loss 계산
        self.mse = nn.MSELoss()

        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0

        # User Representation & 긍정 샘플 / 부정 샘플 -> Recommendation Loss(BPR Loss) 계산
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)

            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(
                    self.llm.llm_model.config.hidden_size,
                    self.llm.llm_model.config.hidden_size,
                ),
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(
                    self.llm.llm_model.config.hidden_size,
                    self.llm.llm_model.config.hidden_size,
                ),
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)

    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir += f"{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_"
        if args.pretrain_stage1:
            sbert_buffer = io.BytesIO()
            mlp_buffer = io.BytesIO()
            mlp2_buffer = io.BytesIO()

            torch.save(self.sbert.state_dict(), sbert_buffer)
            torch.save(self.mlp.state_dict(), mlp_buffer)
            torch.save(self.mlp2.state_dict(), mlp2_buffer)

            self.api.upload_file(
                path_or_fileobj=sbert_buffer,
                path_in_repo=out_dir + "sbert.pt",
                repo_id=self.repo_id,
            )
            self.api.upload_file(
                path_or_fileobj=mlp_buffer,
                path_in_repo=out_dir + "mlp.pt",
                repo_id=self.repo_id,
            )
            self.api.upload_file(
                path_or_fileobj=mlp2_buffer,
                path_in_repo=out_dir + "mlp2.pt",
                repo_id=self.repo_id,
            )

        out_dir += f"{args.llm}_{epoch2}_"
        if args.pretrain_stage2:
            log_proj_buffer = io.BytesIO()
            item_proj_buffer = io.BytesIO()

            torch.save(self.log_emb_proj.state_dict(), log_proj_buffer)
            torch.save(self.item_emb_proj.state_dict(), item_proj_buffer)

            self.api.upload_file(
                path_or_fileobj=log_proj_buffer,
                path_in_repo=out_dir + "log_proj.pt",
                repo_id=self.repo_id,
            )
            self.api.upload_file(
                path_or_fileobj=item_proj_buffer,
                path_in_repo=out_dir + "item_proj.pt",
                repo_id=self.repo_id,
            )

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f"{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_"

        # Item Encoder load
        mlp_path = hf_hub_download(repo_id=self.repo_id, filename=out_dir + "mlp.pt")
        mlp_weights = torch.load(mlp_path, map_location=args.device)
        self.mlp.load_state_dict(mlp_weights)
        del mlp_weights
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        # Text Encoder load
        mlp2_path = hf_hub_download(repo_id=self.repo_id, filename=out_dir + "mlp2.pt")
        mlp2_weights = torch.load(mlp2_path, map_location=args.device)
        self.mlp2.load_state_dict(mlp2_weights)
        del mlp2_weights
        for name, param in self.mlp2.named_parameters():
            param.requires_grad = False

        # SBERT model load
        sbert_path = hf_hub_download(
            repo_id=self.repo_id, filename=out_dir + "sbert.pt"
        )
        sbert_weights = torch.load(sbert_path, map_location=args.device)
        self.sbert.load_state_dict(sbert_weights)
        del sbert_weights
        for name, param in self.sbert.named_parameters():
            param.requires_grad = False

        if args.inference:
            out_dir += f"{args.llm}_{phase2_epoch}_"

            log_proj_path = hf_hub_download(
                repo_id=self.repo_id, filename=out_dir + "log_proj.pt"
            )
            log_proj_weights = torch.load(log_proj_path, map_location=args.device)
            self.log_emb_proj.load_state_dict(log_proj_weights)
            del log_proj_weights

            item_proj_path = hf_hub_download(
                repo_id=self.repo_id, filename=out_dir + "item_proj.pt"
            )
            item_proj_weights = torch.load(item_proj_path, map_location=args.device)
            self.item_emb_proj.load_state_dict(item_proj_weights)
            del item_proj_weights

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = "title"
        d = "description"
        t_ = "No Title"
        d_ = "No Description"
        if title_flag and description_flag:
            return [
                f'"{clean_text(self.text_name_dict[t].get(i,t_))}, {self.text_name_dict[d].get(i,d_)}"'
                for i in item
            ]
        elif title_flag and not description_flag:
            return [f'"{clean_text(self.text_name_dict[t].get(i,t_))}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def find_item_text_single(
        self, item, title_flag=True, description_flag=True, mode=None
    ):
        t = "title"
        d = "description"
        t_ = "No Title"
        d_ = "No Description"
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            if mode == "inference":
                return f'"{clean_text(self.text_name_dict[t].get(item,t_))}"'
            else:
                return f'"{clean_text(self.text_name_dict[t].get(item,t_))}"'

        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'

    def get_item_emb(self, item_ids, mode=None):
        # breakpoint()
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(
                torch.LongTensor(item_ids).to(self.device)
            )
            # 2개 layer(encoder, decoder)중, encoder를 거친 값만 사용
            item_embs, _ = self.mlp(item_embs)

            if mode == "cold":
                # Step 1: Cold items의 인덱스 추출
                cold_indices = [
                    idx for idx, item in enumerate(item_ids) if item in self.cold_items
                ]

                # cold item 있는지 확인
                if cold_indices:
                    # Step 2: Cold items 텍스트 추출
                    cold_item_ids = [item_ids[idx] for idx in cold_indices]
                    cold_texts = self.find_item_text(cold_item_ids)  # 한번에 처리

                    # Step 3: Cold items 텍스트 임베딩 계산 (배치 처리)
                    cold_tokens = self.sbert.tokenize(cold_texts)
                    cold_text_embeddings = self.sbert(
                        {
                            "input_ids": cold_tokens["input_ids"].to(self.device),
                            "attention_mask": cold_tokens["attention_mask"].to(
                                self.device
                            ),
                        }
                    )[
                        "sentence_embedding"
                    ]  # SBERT hidden dim 768

                    cold_text_matching_text, _ = self.mlp2(cold_text_embeddings)

                    for idx, emb in zip(cold_indices, cold_text_matching_text):
                        item_embs[idx] = emb

        return item_embs

    def forward(self, data, optimizer=None, batch_iter=None, mode="phase1"):
        if mode == "phase1":
            self.pre_train_phase1(data, optimizer, batch_iter)
        elif mode == "phase2":
            self.pre_train_phase2(data, optimizer, batch_iter)
        elif mode == "generate":
            result = self.generate(data)
        elif mode == "inference":
            result = self.inference(data)
            return result

    def pre_train_phase1(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data  # u: user 나타냄
        indices = [self.maxlen * (i + 1) - 1 for i in range(u.shape[0])]

        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(
                u, seq, pos, neg, mode="item"
            )  # item이라 지정 -> 1600, 50

        # 각 행렬 내 행들의 마지막 원소 모음 -> 시퀀스 마지막 아이템 관련 행렬들
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]

        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0
        original_loss = 0

        while start_inx < len(log_emb_):
            # 한 번에 처리할 수 있는 크기 start_inx ~ end_inx
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]

            # 아이템 텍스트 찾기 위해 만듬
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]

            start_inx = end_inx
            end_inx += 60
            iterss += 1

            # 마지막 아이템들에 관한 title, description을 이어서 반환
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)

            pos_token = self.sbert.tokenize(pos_text)
            # SBert hidden dim 768
            pos_text_embedding = self.sbert(
                {
                    "input_ids": pos_token["input_ids"].to(self.device),
                    "attention_mask": pos_token["attention_mask"].to(self.device),
                }
            )["sentence_embedding"]
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding = self.sbert(
                {
                    "input_ids": neg_token["input_ids"].to(self.device),
                    "attention_mask": neg_token["attention_mask"].to(self.device),
                }
            )["sentence_embedding"]

            # 아이템: 인코딩, 인코딩-디코딩 결과
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)

            # 텍스트: 인코딩, 인코딩-디코딩 결과
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)

            # Recommendation Loss
            # log_emb -> user representation(batch 크기 * 각 시퀀스 마지막 아이템 임베딩 크기), element-wise & 각 행 평균
            pos_logits, neg_logits = (log_emb * pos_proj).mean(axis=1), (
                log_emb * neg_proj
            ).mean(axis=1)
            pos_labels, neg_labels = torch.ones(
                pos_logits.shape, device=pos_logits.device
            ), torch.zeros(neg_logits.shape, device=pos_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)

            # Matching Loss & Reconstruction Loss
            matching_loss = self.mse(
                pos_text_matching, pos_text_matching_text
            ) + self.mse(neg_text_matching, neg_text_matching_text)
            reconstruction_loss = self.mse(pos_proj, pos_emb) + self.mse(
                neg_proj, neg_emb
            )
            text_reconstruction_loss = self.mse(
                pos_text_proj, pos_text_embedding.data
            ) + self.mse(neg_text_proj, neg_text_embedding.data)

            total_loss = (
                loss
                + matching_loss
                + 0.5 * reconstruction_loss
                + 0.2 * text_reconstruction_loss
            )
            total_loss.backward()
            optimizer.step()

            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()

        print(
            "loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(
                epoch,
                total_epoch,
                step,
                total_step,
                mean_loss / iterss,
                bpr_loss / iterss,
                gt_loss / iterss,
                rc_loss / iterss,
                text_rc_loss / iterss,
            )
        )

    def make_interact_text(self, interact_ids, interact_max_num):
        interact_ids = [item for item in interact_ids if item not in self.missing_items]
        interact_item_titles_ = self.find_item_text(
            interact_ids, title_flag=True, description_flag=False
        )
        interact_text = []
        if interact_max_num == "all":
            for title in interact_item_titles_:
                interact_text.append(title + "[HistoryEmb]")
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + "[HistoryEmb]")
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ",".join(interact_text)
        return interact_text, interact_ids

    def make_candidate_text(
        self, interact_ids, candidate_num, target_item_id, target_item_title
    ):
        neg_item_id = []
        while len(neg_item_id) < 50:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)

        # 정답 id
        candidate_ids = [target_item_id]
        candidate_text = [target_item_title + "[CandidateEmb]"]

        # candidate_ids에 neg_item_id(랜덤 생성)를 넣는다,,,,, -> historical & user-representation을 결합한 벡터와 유사한 아이템을 넣을 순 없나
        for neg_candidate in neg_item_id[: candidate_num - 1]:
            if neg_candidate not in self.missing_items:
                candidate_text.append(
                    self.find_item_text_single(
                        neg_candidate, title_flag=True, description_flag=False
                    )
                    + "[CandidateEmb]"
                )
                candidate_ids.append(neg_candidate)
            else:
                continue

        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]

        return ",".join(candidate_text), candidate_ids

    def make_candidate_text_inference(self, log_emb, interact_ids, candidate_num):
        candidate_ids = self.recsys.make_candidate_for_LLM(
            self.item_num, log_emb, interact_ids, self.args
        )
        candidate_text = []

        # candidate_ids에 neg_item_id(랜덤 생성)를 넣는다,,,,, -> historical & user-representation을 결합한 벡터와 유사한 아이템을 넣을 순 없나
        candidate_ids = [
            item for item in candidate_ids if item not in self.missing_items
        ]
        candidate_mapper = {}
        for candidate in candidate_ids:
            title = self.find_item_text_single(
                candidate, title_flag=True, description_flag=False, mode="inference"
            )
            candidate_text.append(title + "[CandidateEmb]")
            candidate_mapper[title] = candidate

        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]

        return ",".join(candidate_text), candidate_ids, candidate_mapper

    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter

        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0

        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        self.llm.eval()

        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode="log_only")

        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(
                target_item_id, title_flag=True, description_flag=False
            )

            interact_text, interact_ids = self.make_interact_text(
                seq[i][seq[i] > 0], 10
            )
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(
                seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title
            )

            # 왜 user representation 없는지
            input_text = ""
            input_text += " is a user representation."

            if self.args.rec_pre_trained_data == "Movies_and_TV":
                input_text += "This user has watched "
            elif self.args.rec_pre_trained_data == "Video_Games":
                input_text += "This user has played "
            elif (
                self.args.rec_pre_trained_data == "Luxury_Beauty"
                or self.args.rec_pre_trained_data == "Toys_and_Games"
            ):
                input_text += "This user has bought "

            input_text += interact_text

            if self.args.rec_pre_trained_data == "Movies_and_TV":
                input_text += " in the previous. Recommend one next movie for this user to watch next from the following movie title set, "
            elif self.args.rec_pre_trained_data == "Video_Games":
                input_text += " in the previous. Recommend one next game for this user to play next from the following game title set, "
            elif (
                self.args.rec_pre_trained_data == "Luxury_Beauty"
                or self.args.rec_pre_trained_data == "Toys_and_Games"
            ):
                input_text += " in the previous. Recommend one next item for this user to buy next from the following item title set, "

            input_text += candidate_text
            input_text += ". The recommendation is "

            text_input.append(input_text)
            text_output.append(target_item_title)

            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))

        samples = {
            "text_input": text_input,
            "text_output": text_output,
            "interact": interact_embs,
            "candidate": candidate_embs,
        }
        log_emb = self.log_emb_proj(log_emb)
        loss_rm = self.llm(log_emb, samples)
        # llm 내부의 파라미터는 학습하지 않고, llm과 연결된 외부 레이어들의 파라미터만 학습한다.
        # -> Pytorch의 AutoGrad 기능으로 인해 자동으로 연결된 외부 레이어들의 파라미터를 학습한다.
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        print(
            "A-LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(
                epoch, total_epoch, step, total_step, mean_loss
            )
        )

    def generate(self, data):
        u, seq, pos, neg, rank = data

        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode="log_only")
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(
                    target_item_id, title_flag=True, description_flag=False
                )

                interact_text, interact_ids = self.make_interact_text(
                    seq[i][seq[i] > 0], 10
                )
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(
                    seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title
                )

                input_text = ""
                input_text += " is a user representation."
                if self.args.rec_pre_trained_data == "Movies_and_TV":
                    input_text += "This user has watched "
                elif self.args.rec_pre_trained_data == "Video_Games":
                    input_text += "This user has played "
                elif (
                    self.args.rec_pre_trained_data == "Luxury_Beauty"
                    or self.args.rec_pre_trained_data == "Toys_and_Games"
                ):
                    input_text += "This user has bought "

                input_text += interact_text

                if self.args.rec_pre_trained_data == "Movies_and_TV":
                    input_text += " in the previous. Recommend one next movie for this user to watch next from the following movie title set, "
                elif self.args.rec_pre_trained_data == "Video_Games":
                    input_text += " in the previous. Recommend one next game for this user to play next from the following game title set, "
                elif (
                    self.args.rec_pre_trained_data == "Luxury_Beauty"
                    or self.args.rec_pre_trained_data == "Toys_and_Games"
                ):
                    input_text += " in the previous. Recommend one next item for this user to buy next from the following item title set, "

                input_text += candidate_text
                input_text += ". The recommendation is "
                # mapping
                answer.append(target_item_title)
                text_input.append(input_text)

                interact_embs.append(
                    self.item_emb_proj(self.get_item_emb(interact_ids, mode="cold"))
                )
                candidate_embs.append(
                    self.item_emb_proj(self.get_item_emb(candidate_ids, mode="cold"))
                )

        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)

        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input, padding="longest", return_tensors="pt"
            ).to(self.device)

            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(
                    llm_tokens.input_ids
                )

                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(
                    llm_tokens, inputs_embeds, interact_embs, candidate_embs
                )

                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=1,
                    # max_length=512,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                    max_new_tokens=512,
                )

            outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]

        for i in range(len(text_input)):
            f = open(
                f"./ML/models/ALLMRec/evaluation/recommendation_sbert_output_35.txt",
                "a",
            )
            f.write(text_input[i])
            f.write("\n\n")

            f.write("Answer: " + answer[i])
            f.write("\n\n")

            f.write("LLM: " + str(output_text[i]))
            f.write("\n\n")
            f.close()

        return output_text

    def inference(self, data):
        seq, log_emb = data

        text_input = []
        interact_embs = []
        candidate_embs = []
        with torch.no_grad():
            interact_text, interact_ids = self.make_interact_text(seq[seq > 0], 10)

            candidate_num = 20
            candidate_text, candidate_ids, candidate_mapper = (
                self.make_candidate_text_inference(log_emb, seq[seq > 0], candidate_num)
            )

            # 영화 추천
            input_text = ""
            input_text += " is a user representation. This user has watched "
            input_text += interact_text
            input_text += " in the previous. Recommend one next movie for this user to watch next from the following movie title set, "
            input_text += candidate_text
            input_text += ". The recommendation is "

            text_input.append(input_text)

            interact_embs.append(
                self.item_emb_proj(self.get_item_emb(interact_ids, mode="cold"))
            )
            candidate_embs.append(
                self.item_emb_proj(self.get_item_emb(candidate_ids, mode="cold"))
            )

        # user representation projection
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)

        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input, padding="longest", return_tensors="pt"
            ).to(self.device)

            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(
                    llm_tokens.input_ids
                )
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(
                    llm_tokens, inputs_embeds, interact_embs, candidate_embs
                )
                attention_mask = llm_tokens.attention_mask
                # user representation + text input embedding
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                # breakpoint()
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=1,
                    max_length=512,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]

        print(text_input[0])

        output_key = str(output_text[0])
        print("LLM: " + output_key)

        return candidate_mapper.get(output_key, None)
