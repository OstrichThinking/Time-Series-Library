import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

# 多注意力模块
class MultiAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super(MultiAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch_size, n_modalities, seq_len, d_model]
        batch_size, n_modalities, seq_len, d_model = x.size()
        
        # Linear transformations
        q = self.q_linear(x)  # [batch_size, n_modalities, seq_len, d_model]
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Split into heads
        q = q.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        k = k.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        v = v.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        # [batch_size, n_modalities, n_heads, seq_len, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [batch_size, n_modalities, n_heads, seq_len, d_k]
        
        # Concatenate heads
        context = context.transpose(2, 3).contiguous().view(batch_size, n_modalities, seq_len, d_model)
        output = self.out_linear(context)
        return output

# CMA 模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = 30  # 观察窗口长度，例如 30
        self.pred_len = 10  # 预测窗口长度，例如 10
        self.d_model = 512  # 模型维度，例如 64
        self.n_heads = 8  # 注意力头数，例如 8
        self.dropout = 0.1  # 丢弃率，例如 0.1
        
        # 输入嵌入：将 7 个特征映射到 d_model
        self.input_embedding = nn.Linear(7, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=1000)
        
        # 多模态表示：4 个动态指标，每模态拼接 3 个静态指标
        self.n_modalities = 4
        self.modality_embedding = nn.Linear(4, self.d_model)  # 每模态 [seq_len, 4] -> [seq_len, d_model]
        
        # 多注意力机制
        self.multi_attention = MultiAttention(self.d_model, self.n_heads, self.dropout)
        
        # CNN 模块
        self.conv1 = nn.Conv2d(self.n_modalities, (self.n_modalities + 7) // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d((self.n_modalities + 7) // 2, (self.n_modalities + 7) // 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.elu = nn.ELU()
        
        # 全连接层
        fc_input_dim = (self.n_modalities + 7) // 4 * (self.seq_len // 2) * (self.d_model // 2)
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.pred_len * 1)  # 输出 [pred_len, 1]
        
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x, x_mark=None, dec_inp=None, y_mark=None):
        # x: [batch_size, seq_len, 7]
        # x_mark: [batch_size, seq_len, d_model] 时间戳信息（可选）
        
        # 输入嵌入
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoding(x)
        
        # 多模态表示
        batch_size = x.size(0)
        dynamic = x[:, :, :4]  # [batch_size, seq_len, 4] 动态指标
        static = x[:, 0, 4:7].unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, 3] 静态指标
        modalities = []
        for i in range(self.n_modalities):
            modality = torch.cat([dynamic[:, :, i:i+1], static], dim=-1)  # [batch_size, seq_len, 4]
            modality = self.modality_embedding(modality)  # [batch_size, seq_len, d_model]
            modalities.append(modality)
        modalities = torch.stack(modalities, dim=1)  # [batch_size, 4, seq_len, d_model]
        
        # 多注意力机制
        attn_out = self.multi_attention(modalities)  # [batch_size, 4, seq_len, d_model]
        attn_out = self.layer_norm(attn_out)
        
        # CNN 处理
        cnn_out = attn_out.transpose(1, 2)  # [batch_size, seq_len, 4, d_model]
        cnn_out = cnn_out.transpose(1, 3)  # [batch_size, d_model, 4, seq_len]
        cnn_out = cnn_out.transpose(1, 2)  # [batch_size, 4, d_model, seq_len]
        
        cnn_out = self.conv1(cnn_out)  # [batch_size, (4+7)//2, 4, seq_len]
        cnn_out = self.elu(cnn_out)
        cnn_out = self.conv2(cnn_out)  # [batch_size, (4+7)//4, 4, seq_len]
        cnn_out = self.elu(cnn_out)
        cnn_out = self.pool(cnn_out)  # [batch_size, (4+7)//4, 2, seq_len//2]
        
        # 展平并输入全连接层
        cnn_out = cnn_out.view(batch_size, -1)  # [batch_size, fc_input_dim]
        fc_out = self.elu(self.fc1(cnn_out))
        fc_out = self.elu(self.fc2(fc_out))
        output = self.fc3(fc_out)  # [batch_size, pred_len * 1]
        output = output.view(batch_size, self.pred_len, 1)  # [batch_size, pred_len, 1]
        
        return output