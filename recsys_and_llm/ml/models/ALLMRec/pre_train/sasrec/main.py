import argparse
import io
import os
import time

import torch
from data_preprocess import *
from huggingface_hub import HfApi
from model import SASRec
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=50, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.5, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--inference_only", default=False, action="store_true")
parser.add_argument("--state_dict_path", default=None, type=str)

args = parser.parse_args()

if __name__ == "__main__":

    # global dataset
    preprocess(args.dataset)
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print("user num:", usernum, "item num:", itemnum)
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    # dataloader
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
    )
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(
                args.state_dict_path, map_location=torch.device(args.device)
            )
            kwargs["args"].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except:
            print("failed loading state_dicts, pls check file path: ", end="")
            print(args.state_dict_path)
            print(
                "pdb enabled for your quick check, pls type exit() if you do not need it"
            )
            import pdb

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(t_test)

    # CE loss: 다중클래스분류에서 사용, 여러 레이블 중 "정답 레이블"을 얼마나 높은 확률로 설정했는지 체크(나머지 클래스 예측 확률 고려 X)
    # BCE loss: 이진클래스분류에서 사용, 두 가지 클래스 예측 확률 고려
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only:
            break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(
                pos_logits.shape, device=args.device
            ), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(
                pos != 0
            )  # indices[0]: 0이 아닌 값의 행 인덱스, indices[1]: 0이 아닌 값의 열 인덱스
            # pos -> 1, neg -> 0 이진 분류이므로 BCE
            # pos, neg samples -> BCE Loss 사용
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # 모델의 임베딩 파라미터에 대해 L2-norm 값을 계산하여 손실에 추가
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0:
                print(
                    "loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())
                )  # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating", end="")
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(
                model, dataset, args
            )  # 학습 중간 성능 확인용, 하이퍼파라미터 조정용
            print("\n")
            print(
                "epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)"
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            )

            print(str(t_valid) + " " + str(t_test) + "\n")
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:

            folder = args.dataset
            fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
            fname = fname.format(
                args.num_epochs,
                args.lr,
                args.num_blocks,
                args.num_heads,
                args.hidden_units,
                args.maxlen,
            )

            repo_id = "PNUDI/A-LLM"
            out_dir = f"recsys/{folder}/sasrec/"
            api = HfApi()
            recsys_buffer = io.BytesIO()

            torch.save([model.kwargs, model.state_dict()], recsys_buffer)
            api.upload_file(
                path_or_fileobj=recsys_buffer,
                path_in_repo=out_dir + fname,
                repo_id=repo_id,
            )

    sampler.close()
    print("Done")
