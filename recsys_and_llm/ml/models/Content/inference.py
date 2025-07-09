import argparse
import os

import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

from recsys_and_llm.ml.models.Content.utils import get_item_data


def main(args):
    # GET DATA FROM DB & LOAD MODEL
    texts = get_item_data()
    "all-mpnet-base-v2"
    bert_model = SentenceTransformer(args.bert_path, device=args.device)

    # INFERENCE
    text_sb = bert_model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True
    )

    # SAVE EMBEDDING FILE
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    torch.save(text_sb, f"{args.saved_path}/{args.file_name}.pt")

    # UPLOAD TO HFFUB
    api = HfApi()
    api.upload_file(
        path_or_fileobj=f"{args.saved_path}/{args.file_name}.pt",
        path_in_repo=f"{args.file_name}.pt",
        repo_id=args.repo_id,
        commit_message=f"Upload item text info embedding file ({args.bert_path})",
        repo_type="model",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--bert_path", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--saved_path", type=str, default="./saved_models")
    parser.add_argument("--file_name", type=str, default="item_text_emb_SB")
    parser.add_argument("--repo_id", type=str, default="PNUDI/Item_based")

    main(parser.parse_args())
