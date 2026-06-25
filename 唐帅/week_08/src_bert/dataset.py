import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

random.seed(42)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_single(tokenizer, text, max_length):
    encode = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encode["input_ids"].squeeze(0),
        "attention_mask": encode["attention_mask"].squeeze(0),
        "token_type_ids": encode["token_type_ids"].squeeze(0),
    }


# PairDataset
class PairDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        self.rows = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        encode_a = encode_single(self.tokenizer, row["sentence1"], self.max_length)
        encode_b = encode_single(self.tokenizer, row["sentence2"], self.max_length)

        return {
            "input_ids_a": encode_a["input_ids"],
            "attention_mask_a": encode_a["attention_mask"],
            "token_type_ids_a": encode_a["token_type_ids"],
            "input_ids_b": encode_b["input_ids"],
            "attention_mask_b": encode_b["attention_mask"],
            "token_type_ids_b": encode_b["token_type_ids"],
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


# TripletDataset
class TripletDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = self._build_triplets(load_jsonl(data_path))

    def _build_triplets(self, rows):
        # 为每个句子建立"其负样本列表"索引
        neg_by_sent = defaultdict()
        all_sents = set()
        for r in rows:
            all_sents.add(r["sentence1"])
            all_sents.add(r["sentence2"])
            if r["label"] == 0:
                neg_by_sent[r["sentence1"]].append(r["sentence2"])
                neg_by_sent[r["sentence2"]].append(r["sentence1"])

        global_pool = list(all_sents)

        triplets = []
        for r in rows:
            if r["label"] != 1:
                continue
            anchor = r["sentence1"]
            positive = r["sentence2"]

            negs = neg_by_sent.get(anchor, [])
            if negs:
                negative = random.choice(negs)
            else:
                # anchor 没有自身的负样本 → 从全局随机选
                negative = anchor
                while negative in (anchor, positive):
                    negative = random.choice(global_pool)

            triplets.append((anchor, positive, negative))

        print(f"  TripletDataset: 构建 {len(triplets):,} 个三元组")
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        enc_a = encode_single(self.tokenizer, anchor, self.max_length)
        enc_p = encode_single(self.tokenizer, positive, self.max_length)
        enc_n = encode_single(self.tokenizer, negative, self.max_length)
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_p": enc_p["input_ids"],
            "attention_mask_p": enc_p["attention_mask"],
            "token_type_ids_p": enc_p["token_type_ids"],
            "input_ids_n": enc_n["input_ids"],
            "attention_mask_n": enc_n["attention_mask"],
            "token_type_ids_n": enc_n["token_type_ids"],
        }


# CrossEncoderDataset
class CrossEncoderDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.rows = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        encode = self.tokenizer(
            row["sentence1"],
            row["sentence2"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encode["input_ids"].squeeze(0),
            "attention_mask": encode["attention_mask"].squeeze(0),
            "token_type_ids": encode["token_type_ids"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


# DataLoader工厂函数
def build_pair_loaders(data_dir, tokenizer, max_length=64, batch_size=32):
    data_dir = Path(data_dir)
    train_ds = PairDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds = PairDataset(data_dir / "test.jsonl", tokenizer, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch")
    return train_loader, val_loader, test_loader


def build_triplet_loader(data_dir, tokenizer, max_length=64, batch_size=32):
    data_dir = Path(data_dir)
    train_ds = TripletDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  triplet train : {len(train_ds):>7,} 三元组, {len(train_loader):>5} batch")
    print(f"  val (pair)    : {len(val_ds):>7,} 对,     {len(val_loader):>5} batch")
    return train_loader, val_loader


def build_crossencoder_loaders(data_dir, tokenizer, max_length=128, batch_size=32):
    data_dir = Path(data_dir)
    train_ds = CrossEncoderDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = CrossEncoderDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds = CrossEncoderDataset(data_dir / "test.jsonl", tokenizer, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch")
    return train_loader, val_loader, test_loader
