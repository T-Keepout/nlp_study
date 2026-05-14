import torch
import torch.nn as nn
import torch.nn.functional as fc
import math


class MultiHeadAttention(nn.Module):
    """实现多头注意力机制"""

    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super.__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        # 同时生成w和b
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """将输入分割成多个头"""

        batch_size, seq_len = x.shape

        # 重塑：[batch,seq_len,n_head,d_k]
        x = x.view(batch_size, seq_len, self.n_head, self.d_k)

        # 转置：[batch,n_head,seq_len,d_k]
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        """合并多个头"""

        batch_size, seq_len = x.shape

        # 转置回来：[batch,seq_len,n_head,d_k]
        x = x.transpose(1, 2)

        # 重塑：[batch,seq_len,d_model]
        x = x.contiguous().view(batch_size, seq_len, self.d_model)

        return x

    def forward(self, query, key, value):
        # 1. 线性变换并分割多头
        # [batch, seq_len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. 分割成多个头
        # [batch, n_head, seq_len, d_k]
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3.计算注意力
        # [batch, n_head, seq_len, d_k] @ [batch, n_head, d_k, seq_len]
        # = [batch, n_head, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Softmax得到注意力权重
        attention_weights = fc.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和value
        # [batch, n_head, seq_len, seq_len] @ [batch, n_head, seq_len, d_k]
        # = [batch, n_head, seq_len, d_k]
        context = torch.matmul(attention_weights, V)

        # 4. 合并多头
        # [batch, seq_len, d_model]
        context = self.combine_heads(context)

        # 5. 最终线性变换
        output = self.w_o(context)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()

        # 两层线性变换，中间用ReLU激活
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            [batch_size, seq_len, d_model]
        """
        # 第一层：升维 + ReLU
        x = fc.relu(self.linear1(x))
        x = self.dropout(x)

        # 第二层：降维回原始维度
        x = self.linear2(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_ff=2048, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            n_head: 注意力头数
            d_ff: FFN隐藏层维度
            dropout: dropout比率
        """
        super().__init__()

        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, n_head, dropout)

        # 前馈网络层
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 两个LayerNorm层（残差连接后使用）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入 [batch_size, seq_len, d_model]
        返回:
            output: 输出 [batch_size, seq_len, d_model]
        """
        # 1. 多头注意力 + 残差连接 + LayerNorm
        # 保存残差
        residual = x

        # 自注意力
        attn_output, _ = self.attention(x, x, x)

        # Dropout
        attn_output = self.dropout(attn_output)

        # 残差连接 + LayerNorm
        x = self.norm1(residual + attn_output)

        # 2. 前馈网络 + 残差连接 + LayerNorm
        # 保存残差
        residual = x

        # 前馈网络
        ffn_output = self.feed_forward(x)

        # Dropout
        ffn_output = self.dropout(ffn_output)

        # 残差连接 + LayerNorm
        x = self.norm2(residual + ffn_output)

        return x
