'''

'''

import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter
import random

# 1.设置随机种子
random.seed(42)
torch.manual_seed(42)

# 2.分词器，将文本转化为词汇索引
class Tokenizer:
    def __init__(self, texts, min_freq=1, max_vocab_size=10000):
        self.texts = texts
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.build_vocab()
    
    # 清洗并分割文本
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    # 构建词汇表
    def build_vocab(self):
        word_counter = Counter()
        for text in self.texts:
            tokens = self.tokenize(text)
            word_counter.update(tokens)
        vocab={"<pad>":0, "<unk>":1}
        for word, freq in word_counter.most_common(self.max_vocab_size - 2):
            if freq >= self.min_freq:
                vocab[word] = len(vocab)
        return vocab
    
    # 将文本转化为词汇索引序列
    def text_to_sequence(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
    
# 定义示例文本数据
texts = ["The quick brown fox jumps over the lazy dog",
         "PyTorch is widely used for deep learning tasks",
         "Natural laguage processing enables complex interactions",
         "This example demonstrates text embedding in PyTorch", ]
    
# 实例化分词器
tokenizer = Tokenizer(texts)

# 定义示例文本
text_sequence=tokenizer.text_to_sequence("The quick brown fox")
print("分词后的索引序列：", text_sequence)

# 嵌入层定义，将词汇索引映射到嵌入向量
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
    
    def forward(self, x):
        return self.embedding(x)
    
# 超参数设置
VOCAB_SIZE = len(tokenizer.vocab)
EMB_DIM = 8 # 嵌入维度
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建嵌入层实例
embedding_layer = TextEmbedding(VOCAB_SIZE, EMB_DIM).to(DEVICE)

# 将文本索引序列转为张量并获取嵌入向量
text_tensor = torch.tensor([text_sequence], dtype=torch.long).to(DEVICE) # 增加批次维度
embedded_output = embedding_layer(text_tensor)
print("嵌入向量形状：", embedded_output.shape) # [1, seq_len, emb_dim]
print("嵌入向量：", embedded_output)

# 嵌入层的训练示例， 使用随机生成的目标嵌入向量计算损失
optimizer = optim.Adam(embedding_layer.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 假设目标是另一个随机嵌入向量
target_embedding = torch.rand(embedded_output.shape).to(DEVICE)
loss = criterion(embedded_output, target_embedding)

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("训练后的损失：", loss.item())

# 显示嵌入层权重的部分
print("嵌入层权重矩阵：\n", embedding_layer.embedding.weight.data[:5]) # 显示前5个词的嵌入向量