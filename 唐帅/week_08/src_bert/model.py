import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel


# BiEncoder
class BiEncoder(nn.Module):
    """
    表示型文本匹配：Siamese Bi-Encoder

    结构：
      shared BertModel → 池化 → Dropout → L2 归一化 → 句向量

    匹配方式：
      sim = cosine_similarity(encode(s1), encode(s2))
      sim ∈ [-1, 1]，越接近 1 越相似

    支持两种 Loss：
      CosineEmbeddingLoss — 直接用相似度与标签计算损失
      TripletLoss         — 拉近 (anchor, positive)，推远 (anchor, negative)
    """

    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=12):
        super().__init__()

        config = BertConfig.from_pretrained(bert_path)
        config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.pool = pool
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        logit = self.pool(output.last_hidden_state, attention_mask)
        logit = self.dropout(logit)
        return F.normalize(logit, p=2, dim=-1)

    def forward(self, batch_a, batch_b):
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b

    def pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values


# CrossEncoder
class CrossEncoder(nn.Module):
    def __init__(self, bert_path, dropout=0.1, num_hidden_layers=12):
        super().__init__()

        config = BertConfig.from_pretrained(bert_path)
        config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = output.last_hidden_state[:, 0, :]
        cls_vec = self.dropout(cls_vec)
        return self.classifier(cls_vec)


def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    """构建 BiEncoder 并打印参数量。"""
    model = BiEncoder(
        bert_path, pool=pool, dropout=dropout, num_hidden_layers=num_hidden_layers
    )
    _print_param_info(
        model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})"
    )
    return model


def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    """构建 CrossEncoder 并打印参数量。"""
    model = CrossEncoder(
        bert_path, dropout=dropout, num_hidden_layers=num_hidden_layers
    )
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")
    return model


def _print_param_info(model, name):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: {name}")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")
