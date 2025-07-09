import argparse
import os
import pickle
import time

import numpy as np
import torch
from model import TiSASRec
from tqdm import tqdm
from utils import *


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_args(args):
    folder = f"{args.dataset}_{args.train_dir}"
    ensure_dir(folder)
    with open(os.path.join(folder, "args.txt"), "w") as f:
        f.write("\n".join([f"{k},{v}" for k, v in sorted(vars(args).items())]))


def load_relation_matrix(dataset, maxlen, time_span):
    try:
        return pickle.load(
            open(f"../data/relation_matrix_{dataset}_{maxlen}_{time_span}.pickle", "rb")
        )
    except FileNotFoundError:
        return None


def save_relation_matrix(relation_matrix, dataset, maxlen, time_span):
    with open(
        f"../data/relation_matrix_{dataset}_{maxlen}_{time_span}.pickle", "wb"
    ) as f:
        pickle.dump(relation_matrix, f)


def initialize_model(args, usernum, itemnum):
    model = TiSASRec(usernum, itemnum, itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass  # Ignore initialization errors
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=50, type=int)
    parser.add_argument("--hidden_units", default=64, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.00005, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--inference_only", default=False, type=str2bool)
    parser.add_argument("--state_dict_path", default=None, type=str)
    parser.add_argument("--time_span", default=256, type=int)

    args = parser.parse_args()
    save_args(args)

    dataset = data_partition(args.dataset)
    (
        User,
        user_train,
        user_valid,
        user_test,
        usernum,
        itemnum,
        timenum,
        item_map,
        reverse_item_map,
        all_item,
    ) = dataset
    num_batch = len(user_train) // args.batch_size
    # print(len(user_train))
    # print(num_batch)

    relation_matrix = load_relation_matrix(args.dataset, args.maxlen, args.time_span)
    if relation_matrix is None:
        relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
        save_relation_matrix(relation_matrix, args.dataset, args.maxlen, args.time_span)

    model = initialize_model(args, usernum, itemnum)
    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    with open(f"data/Movies_and_TV_text_name_dict.json.gz", "rb") as ft:
        text_name_dict = pickle.load(ft)

    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            # epoch_start_idx = int(args.state_dict_path.split('epoch_')[1].split('.')[0]) + 1
        except Exception as e:
            print(f"Failed to load state_dict: {e}")
            epoch_start_idx = 1
    else:
        epoch_start_idx = 1

    if args.inference_only:
        model.eval()

        while True:
            # print(type(User))
            # print(User)
            user_id = int(input("User ID: ").strip())
            if user_id == 0:
                break
            tok5_title = recommend_top5(model, user_id)
            print(f"User ID = {user_id} title: {tok5_title}")
    else:
        sampler = WarpSampler(
            user_train,
            usernum,
            itemnum,
            relation_matrix,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            n_workers=3,
        )
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        T = 0.0
        t0 = time.time()
        best_val = 0
        best_test = 0
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            for step in range(num_batch):
                u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch()
                u, seq, pos, neg = map(np.array, (u, seq, pos, neg))
                time_seq, time_matrix = map(np.array, (time_seq, time_matrix))

                pos_logits, neg_logits = model(u, seq, time_matrix, pos, neg)
                pos_labels = torch.ones_like(pos_logits, device=args.device)
                neg_labels = torch.zeros_like(neg_logits, device=args.device)

                optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                for param in model.parameters():
                    loss += args.l2_emb * torch.norm(param)

                loss.backward()
                optimizer.step()
                if step % 200 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

            if epoch % 50 == 0:
                model.eval()
                t1 = time.time() - t0
                T += t1
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                print(
                    f"Epoch {epoch}, Time: {T:.2f}s, Valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), "
                    f"Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})"
                )
                t0 = time.time()
                f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
                f.flush()

                if t_test[1] > best_test or t_valid[1] > best_val:
                    torch.save(
                        model.state_dict(),
                        f"{args.dataset}_{args.train_dir}/TiSASRec_model_epoch_{args.num_epochs}.pt",
                    )
                    best_test = t_test[1]
                    best_val = t_valid[1]
                model.train()

        torch.save(
            model.state_dict(),
            f"{args.dataset}_{args.train_dir}/TiSASRec_model_epoch_{args.num_epochs}.pt",
        )
        print("Model saved.")
        sampler.close()

    print("Training complete.")
