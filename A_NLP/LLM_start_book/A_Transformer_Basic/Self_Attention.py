'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1.设置随机种子
torch.manual_seed(42)

# 2.定义自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # 嵌入维度
        self.embed_size = embed_size
        # 多头向量
        self.heads = heads
        # 每个头的维度
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        # 定义线性变换用于生成查询、键、值(K,Q,V)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N=query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 分头计算Q,K,V矩阵
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = query.view(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算Q与K的点积，除以缩放因子sqrt(d_k)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 计算注意力权重
        attention = torch.softmax(energy, dim=-1)  # (N, heads, query_len, key_len)

        # 注意力加权权重乘以V
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # (N, query_len, heads, head_dim) -> (N, query_len, embed_size)

# 设置参数
embed_size = 128 # 嵌入维度
heads = 8       # 注意力头数
seq_length = 10 # 序列长度
batch_size = 2 # 批次大小

# 创建随机输入
values = torch.rand((batch_size, seq_length, embed_size))
keys = torch.rand((batch_size, seq_length, embed_size))
queries = torch.rand((batch_size, seq_length, embed_size))

# 初始化自注意力层
self_attention_layer = SelfAttention(embed_size, heads)

# 前向传播
output = self_attention_layer(values, keys, queries, mask=None)

print("输出的形状：", output.shape)  # 应该是 (batch_size, seq_length, embed_size
print("自注意力机制的输出：\n", output) # 输出的形状应该是 (batch_size, seq_length, embed_size)

# 进一步展示注意力权重计算
class SelfAttentionWithWeights(SelfAttention):
    def forward(self, values, keys, query, mask):
        N=query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = query.view(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)  # (N, heads, query_len, key_len)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # (N, query_len, heads, head_dim) -> (N, query_len, embed_size)
        return self.fc_out(out), attention
    
# 使用带有权重输出的自注意力层
self_attention_with_weights = SelfAttentionWithWeights(embed_size, heads)
output, attention_weights = self_attention_with_weights(values, keys, queries, mask=None)

print("注意力权重的形状：", attention_weights.shape)  # 应该是 (batch_size, heads, query_len, key_len)
print("注意力权重：\n", attention_weights) # 输出注意力权重