"""
任务：新闻标题 -> 5个主题类别（科技/体育/财经/娱乐/政治）
模型：Embedding -> RNN -> MaxPool -> BN -> Dropout -> Linear(5) -> CrossEntropyLoss
优化：Adam（lr=1e-3） 损失：CrossEntropyLoss

"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 超参数
SEED = 42
N_SAMPLES = 5000
MAXLEN = 30
EMBED_DIM = 128
HIDDEN_DIM = 128
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 30
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# 1.类别定义
CATEGORIES = {0: "科技", 1: "体育", 2: "财经", 3: "娱乐", 4: "政治"}

# 各类别的关键词和模板
CATEGORY_DATA = {
    0: {  # 科技
        "keywords": [
            "AI",
            "人工智能",
            "算法",
            "芯片",
            "5G",
            "量子",
            "机器人",
            "自动驾驶",
            "云计算",
            "大数据",
        ],
        "templates": [
            "{}技术迎来重大突破",
            "{}公司发布新款{}产品",
            "科学家在{}领域取得进展",
            "{}应用改变生活方式",
            "{}成为行业新趋势",
        ],
    },
    1: {  # 体育
        "keywords": [
            "进球",
            "夺冠",
            "比赛",
            "决赛",
            "奥运",
            "世界杯",
            "篮球",
            "足球",
            "NBA",
            "冠军",
        ],
        "templates": [
            "{}队在{}中绝杀对手",
            "{}选手夺得{}冠军",
            "{}进入{}决赛",
            "{}赛季精彩瞬间回顾",
            "{}遗憾止步{}强",
        ],
    },
    2: {  # 财经
        "keywords": [
            "股市",
            "基金",
            "涨停",
            "跌停",
            "A股",
            "美股",
            "投资",
            "理财",
            "经济",
            "GDP",
        ],
        "templates": [
            "{}指数{}个点",
            "{}公司发布{}财报",
            "{}行业迎来{}机遇",
            "央行调整{}政策",
            "{}成为投资新热点",
        ],
    },
    3: {  # 娱乐
        "keywords": [
            "明星",
            "电影",
            "演唱会",
            "综艺",
            "绯闻",
            "首映",
            "票房",
            "电视剧",
            "新歌",
            "颁奖",
        ],
        "templates": [
            "{}主演{}即将上映",
            "{}出席{}活动引关注",
            "{}发布{}新歌",
            "{}获得{}大奖",
            "{}恋情曝光登上热搜",
        ],
    },
    4: {  # 政治
        "keywords": [
            "会议",
            "政策",
            "领导",
            "外交",
            "法案",
            "选举",
            "谈判",
            "声明",
            "协议",
            "改革",
        ],
        "templates": [
            "{}举行{}重要会议",
            "双方达成{}协议",
            "{}宣布新的{}政策",
            "{}发表{}重要讲话",
            "{}通过{}法案",
        ],
    },
}


def generate_text(category_id):
    """为指定类别生成文本"""
    data = CATEGORY_DATA[category_id]

    # 随机选择模板或关键词组合
    if random.random() < 0.6:
        # 使用模板
        template = random.choice(data["templates"])
        # 替换模板中的占位符
        if "{}" in template:
            placeholder_count = template.count("{}")
            replacements = []
            for _ in range(placeholder_count):
                replacements.append(random.choice(data["keywords"]))
            return template.format(*replacements)
        return template
    else:
        # 关键词组合
        kw_count = random.randint(2, 4)
        keywords = random.sample(data["keywords"], min(kw_count, len(data["keywords"])))
        return "".join(keywords) + random.choice(["突破", "创新", "新动态", "最新消息"])


def build_dataset(n_per_class=1000):
    """构建平衡数据集"""
    data = []
    for category_id in range(len(CATEGORIES)):
        for _ in range(n_per_class):
            text = generate_text(category_id)
            data.append((text, category_id))

    random.shuffle(data)
    print(f"数据集构建完成：{len(data)} 条文本，5个类别各约 {n_per_class} 条")

    # 打印示例
    print("\n生成示例：")
    for i, (text, label) in enumerate(data[:10]):
        print(f"  [{CATEGORIES[label]}] {text[:50]}...")

    return data


# 2.词表构建与编码
def build_vocab(data, min_freq=2):
    """构建词表（字符级别）"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    freq = {}

    for text, _ in data:
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1

    for ch, f in freq.items():
        if f >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)

    print(f"词表大小：{len(vocab)}(包含PAD和UNK)")
    return vocab


def encode(text, vocab, maxlen=MAXLEN):
    """将文本编码为索引序列"""
    ids = [vocab.get(ch, 1) for ch in text]  # 1 UNK
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))  # 0 PAD
    return ids


# 3.Dataset / DataLoader
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = [encode(text, vocab) for text, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# 4.多分类模型定义
class MultiClassRNN(nn.Module):
    """
    多分类RNN模型
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear(5) → CrossEntropyLoss
    """

    def __init__(
        self,
        vocab_size,
        num_classes=5,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 输出5个类别

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)

        # RNN输出
        rnn_out, _ = self.rnn(embedded)

        # MaxPooling
        pooled = rnn_out.max(dim=1)[0]

        # BatchNorm + Dropout
        pooled = self.bn(pooled)
        pooled = self.dropout(pooled)

        # 分类
        logits = self.fc(pooled)

        return logits


# 5.训练与评估
def evaluate(model, loader):
    """评估模型准确率"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train():
    print("=" * 60)
    print("中文新闻标题多分类任务（5个类别）")
    print("=" * 60)

    # 生成数据集
    print("\n1. 生成数据集...")
    data = build_dataset(n_per_class=N_SAMPLES // len(CATEGORIES))

    # 构建词表
    print("\n2. 构建词表...")
    vocab = build_vocab(data)

    # 划分数据集
    print("\n3. 划分训练/验证集...")
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    print(f"  训练集：{len(train_data)} 条")
    print(f"  验证集：{len(val_data)} 条")

    # 创建DataLoader
    train_loader = DataLoader(
        TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 初始化模型
    print("\n4. 初始化模型...")
    model = MultiClassRNN(vocab_size=len(vocab), num_classes=len(CATEGORIES))
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}")

    # 训练
    print("\n5. 开始训练...")
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_rnn_model.pth")

        print(
            f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.6f}"
        )

    print(f"\n最佳验证准确率：{best_acc:.4f}")

    # 最终评估
    print("\n6. 最终评估...")
    model.load_state_dict(torch.load("best_rnn_model.pth"))
    final_acc = evaluate(model, val_loader)
    print(f"最终验证准确率：{final_acc:.4f}")

    # 推理示例
    print("\n7. 推理示例：")
    model.eval()

    test_examples = [
        "苹果发布新款iPhone搭载AI芯片",
        "湖人队击败勇士晋级季后赛",
        "A股大涨突破3000点关口",
        "周杰伦新歌MV播放量破千万",
        "中美达成重要贸易协议",
        "这家餐厅味道很不错",
    ]

    with torch.no_grad():
        for text in test_examples:
            ids = torch.tensor([encode(text, vocab)], dtype=torch.long)
            logits = model(ids)
            prob = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred_class].item()

            print(f"\n  文本：{text}")
            print(f"  预测：{CATEGORIES[pred_class]} (置信度: {confidence:.2%})")

            # 显示所有类别的概率
            print(f"  详细：", end=" ")
            for i, cat in CATEGORIES.items():
                print(f"{cat}:{prob[0][i]:.2%}", end="  ")
            print()


if __name__ == "__main__":
    train()
