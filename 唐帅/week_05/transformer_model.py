import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import math

torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1.模型超参数（优化后）
class Config:
    vocab_size = 200  # 降低词表大小
    d_model = 64  # 降低模型维度
    n_head = 4  # 减少注意力头
    d_ff = 256  # 减少FFN维度
    n_layers = 2  # 减少层数
    max_seq_len = 32  # 缩短序列长度
    dropout = 0.1

    # 训练参数
    batch_size = 8  # 减小batch size
    learning_rate = 5e-4
    epochs = 50  # 增加训练轮数


config = Config()


# 2.实现因果注意力Mask
def create_causal_mask(seq_len):
    """创建因果注意力掩码，单向因果生成"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask == 1


class CausalMultiHeadAttention(nn.Module):
    """带因果掩码的多头注意力"""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 线性变换
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 分割多头
        Q = Q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用因果掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = torch.matmul(attn_weights, V)

        # 合并多头
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # 输出投影
        output = self.w_o(context)

        return output


# 3.实现单向Transformer层
class CausalTransformerLayer(nn.Module):
    """单向Transformer层"""

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()

        self.attention = CausalMultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力 + 残差
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


# 4.构建语言模型
class CausalLanguageModel(nn.Module):
    """单向语言模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token嵌入
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )

        # 位置嵌入
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer层
        self.layers = nn.ModuleList(
            [
                CausalTransformerLayer(
                    config.d_model, config.n_head, config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )

        # 输出层
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重共享
        self.token_embedding.weight = self.lm_head.weight

        # 因果掩码
        self.register_buffer("causal_mask", create_causal_mask(config.max_seq_len))

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # 嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 获取因果掩码
        mask = self.causal_mask[:seq_len, :seq_len]

        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, mask)

        # 输出
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # 计算损失（忽略PAD token）
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 忽略PAD token (id=0) 的损失
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return logits, loss

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50):
        """文本生成"""
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if input_ids.shape[1] > self.config.max_seq_len:
                    input_ids = input_ids[:, -self.config.max_seq_len :]

                logits, _ = self(input_ids)
                next_token_logits = logits[0, -1, :] / temperature

                # 关键改进：禁止生成PAD(0)和UNK(1)
                next_token_logits[0] = -float("Inf")  # PAD
                next_token_logits[1] = -float("Inf")  # UNK

                if top_k is not None:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return input_ids


# 5.准备数据集
class TextDataset(Dataset):
    """文本数据集"""

    def __init__(self, texts, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 将所有文本转为token ids
        self.data = []
        for text in texts:
            tokens = tokenizer(text)
            if len(tokens) >= 2:  # 至少要有2个token
                self.data.append(tokens)

        # 数据增强：创建滑动窗口
        print("进行数据增强...")
        augmented_data = []
        for tokens in self.data:
            # 从每个文本创建多个子序列
            window_size = max_seq_len
            for i in range(0, max(1, len(tokens) - 1), max(1, window_size // 4)):
                window = tokens[i : i + window_size]
                if len(window) >= 2:
                    augmented_data.append(window)

        self.data.extend(augmented_data)
        print(f"数据集大小（增强后）: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx].copy()

        # 填充或截断
        if len(tokens) < self.max_seq_len:
            # 只在右侧填充PAD
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[: self.max_seq_len]

        return torch.tensor(tokens, dtype=torch.long)


class SimpleTokenizer:
    """简单的字符级tokenizer"""

    def __init__(self, texts, vocab_size):
        # 统计所有字符
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        # 选择最常见的字符
        most_common = char_counts.most_common(vocab_size - 4)

        # 构建词表
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for char, _ in most_common:
            self.vocab[char] = len(self.vocab)

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"实际词表大小: {len(self.vocab)}")

    def encode(self, text):
        """文本转token ids"""
        return [self.vocab.get(char, 1) for char in text]

    def decode(self, tokens):
        """token ids转文本，过滤特殊token"""
        result = []
        for tok in tokens:
            if tok not in [0, 1, 2, 3]:  # 过滤PAD, UNK, BOS, EOS
                result.append(self.reverse_vocab.get(tok, "<UNK>"))
        return "".join(result)

    def __call__(self, text):
        return self.encode(text)


# 6.训练模型
def train_model(model, train_loader, config):
    """训练模型"""
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    model.train()
    print("开始训练...")
    print("=" * 50)

    best_loss = float("inf")

    for epoch in range(config.epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # 前向传播
            logits, loss = model(batch, labels=batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step()

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")

        # 每5个epoch生成示例文本
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}"
            )
            generate_sample(model, "床前", tokenizer)
            generate_sample(model, "春眠", tokenizer)
            print("-" * 50)
        elif (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

    print("训练完成！")
    return model


def generate_sample(model, prompt, tokenizer, max_new_tokens=20):
    """生成示例文本"""
    model.eval()

    # 编码输入
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

    # 生成
    output_ids = model.generate(input_ids, max_new_tokens, temperature=0.8, top_k=30)

    # 解码输出
    generated_text = tokenizer.decode(output_ids[0].cpu().numpy())

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


# 7.准备数据并运行（扩展训练数据）
# 扩展训练数据
training_texts = [
    # 原古诗
    "床前明月光，疑是地上霜。",
    "举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。",
    "夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。",
    "欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。",
    "孤舟蓑笠翁，独钓寒江雪。",
    "离离原上草，一岁一枯荣。",
    "野火烧不尽，春风吹又生。",
    "好雨知时节，当春乃发生。",
    "随风潜入夜，润物细无声。",
    # 增加古诗
    "日照香炉生紫烟，遥看瀑布挂前川。",
    "飞流直下三千尺，疑是银河落九天。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。",
    "窗含西岭千秋雪，门泊东吴万里船。",
    "月落乌啼霜满天，江枫渔火对愁眠。",
    "姑苏城外寒山寺，夜半钟声到客船。",
    "葡萄美酒夜光杯，欲饮琵琶马上催。",
    "醉卧沙场君莫笑，古来征战几人回？",
    # 增加现代短句
    "今天天气真好",
    "我喜欢学习人工智能",
    "深度学习很有趣",
    "神经网络很强大",
    "机器学习需要大量数据",
    "PyTorch是一个好框架",
    "Transformer改变了NLP领域",
]

# 数据重复增强
training_texts = training_texts * 5  # 简单重复增强

print(f"训练数据量: {len(training_texts)} 条")

# 创建tokenizer
tokenizer = SimpleTokenizer(training_texts, config.vocab_size)

# 创建数据集和数据加载器
dataset = TextDataset(training_texts, tokenizer, config.max_seq_len)
train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

print(f"批次数: {len(train_loader)}")

# 创建模型
model = CausalLanguageModel(config).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练模型
model = train_model(model, train_loader, config)

# 保存最终模型
torch.save(model.state_dict(), "language_model_final.pth")
print("模型已保存")


# 8.文本生成演示
def simple_generation(model, tokenizer):
    """极简版文本生成器 - 自动生成，无多余参数"""
    print("\n" + "=" * 50)
    print("文本生成器已启动！输入 'quit' 退出")
    print("=" * 50)

    model.eval()

    # 获取句号的token id（用于早停）
    period_token = tokenizer.vocab.get("。", None)

    while True:
        prompt = input("\n请输入提示词: ")
        if prompt.lower() == "quit":
            break

        if len(prompt) == 0:
            continue

        # 生成文本
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

        with torch.no_grad():
            generated = input_ids
            for _ in range(25):  # 最多生成25个token
                if generated.shape[1] > config.max_seq_len:
                    generated = generated[:, -config.max_seq_len :]

                logits, _ = model(generated)
                next_token_logits = logits[0, -1, :] / 0.8  # temperature=0.8

                # 禁止PAD和UNK
                next_token_logits[0] = -float("Inf")
                next_token_logits[1] = -float("Inf")

                # Top-k
                top_k = 30
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = -float("Inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 如果生成了句号且已经有足够内容，就停止
                if (
                    period_token
                    and next_token.item() == period_token
                    and generated.shape[1] > len(prompt) + 2
                ):
                    # 添加句号后停止
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    break

                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        generated_text = tokenizer.decode(generated[0].cpu().numpy())

        # 后处理：清理多余句号
        generated_text = generated_text.rstrip("。")
        # 确保古诗有结尾句号
        if not generated_text.endswith("。") and len(generated_text) > len(prompt):
            generated_text = generated_text + "。"

        print(f"\n{generated_text}")
        print("-" * 50)


# 加载最佳模型进行生成
def load_and_generate():
    """加载训练好的模型并生成"""
    model = CausalLanguageModel(config).to(device)

    try:
        model.load_state_dict(torch.load("best_model.pth"))
        print("加载最佳模型成功！")
    except:
        try:
            model.load_state_dict(torch.load("language_model_final.pth"))
            print("加载最终模型成功！")
        except:
            print("未找到训练好的模型，请先训练")
            return

    simple_generation(model, tokenizer)


# 运行
if __name__ == "__main__":
    import os

    if os.path.exists("best_model.pth") or os.path.exists("language_model_final.pth"):
        load_and_generate()
    else:
        print("开始训练新模型...")
        # 训练代码已经在上面执行
        load_and_generate()
