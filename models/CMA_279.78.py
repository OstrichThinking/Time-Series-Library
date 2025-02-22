import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding

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

# 多头注意力模块（使用余弦相似度）
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
        Q = self.q_linear(x)  # [batch_size, n_modalities, seq_len, d_model]
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Split into heads
        Q = Q.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        K = K.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        V = V.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        # [batch_size, n_modalities, n_heads, seq_len, d_k]
        
        # 使用余弦相似度计算注意力得分
        Q_norm = F.normalize(Q, p=2, dim=-1)  # L2归一化
        K_norm = F.normalize(K, p=2, dim=-1)
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))  # 余弦相似度
        attn = F.softmax(scores, dim=-1)  # Softmax归一化
        attn = self.dropout(attn)
        
        # 加权求和值
        context = torch.matmul(attn, V)  # [batch_size, n_modalities, n_heads, seq_len, d_k]
        
        # Concatenate heads
        context = context.transpose(2, 3).contiguous().view(batch_size, n_modalities, seq_len, d_model)
        output = self.out_linear(context)
        return output

# CMA 模型
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len  # 观察窗口长度，例如 30
        self.pred_len = args.pred_len  # 预测窗口长度，例如 10
        self.d_model = args.d_model  # 模型维度，例如 64
        self.n_heads = args.n_heads  # 注意力头数，例如 8
        self.dropout = args.dropout  # 丢弃率，例如 0.1
        
        # 输入特征数：7 (sex, age, bmi, 无创舒张压, 无创平均动脉压, 体温, 心率)
        self.input_dim = 7
        # 模态数：4 (每个动态特征 + 3 个静态特征)
        self.n_modalities = 4
        # 每个模态的输入维度：1 (动态) + 3 (静态) = 4
        self.modality_input_dim = 4
        
        # 输入嵌入：将每个模态的 4 个特征映射到 d_model
        self.modality_embedding = nn.Linear(self.modality_input_dim, self.d_model)
        
        # 时间特征嵌入
        self.temporal_embedding = TemporalEmbedding(d_model=args.d_model, embed_type=args.embed,
                                                    freq=args.freq) if args.embed != 'timeF' else TimeFeatureEmbedding(
            d_model=args.d_model, embed_type=args.embed, freq=args.freq)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=1000)
        
        # 多头注意力机制（为每个模态独立实例化）
        self.attentions = nn.ModuleList([MultiAttention(self.d_model, self.n_heads, self.dropout) 
                                        for _ in range(self.n_modalities)])
        
        # CNN 模块：输入通道数为 n_modalities * d_model（拼接后）
        self.conv1 = nn.Conv1d(self.n_modalities * self.d_model, (self.n_modalities + self.input_dim) // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d((self.n_modalities + self.input_dim) // 2, (self.n_modalities + self.input_dim) // 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.elu = nn.ELU()
        
        # 全连接层
        fc_input_dim = (self.n_modalities + self.input_dim) // 4 * (self.seq_len // 2)
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.pred_len * 1)  # 输出 [pred_len, 1]
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # 层归一化
        self.output_layer_norm = nn.LayerNorm(self.pred_len * 1)  # 添加层归一化

    def forward(self, x, x_mark=None, y=None, y_mark=None):
        # x: [batch_size, seq_len, 7]
        # x_mark: [batch_size, seq_len, d_model] 时间戳信息（可选）
        batch_size = x.size(0)
        
        # 分离静态和动态特征
        static = x[:, :, :3]  # [batch_size, seq_len, 3] (sex, age, bmi)
        dynamic = x[:, :, 3:]  # [batch_size, seq_len, 4] (无创舒张压, 无创平均动脉压, 体温, 心率)
        
        # 构造 4 个模态
        modalities = []
        for i in range(self.n_modalities):
            # 每个模态：1 个动态特征 + 3 个静态特征
            modality = torch.cat([dynamic[:, :, i:i+1], static], dim=-1)  # [batch_size, seq_len, 4]
            modality = self.modality_embedding(modality)  # [batch_size, seq_len, d_model]
            
            if x_mark is None:
                modality = self.pos_encoding(modality)  # 直接添加位置编码
            else:
                modality = self.pos_encoding(modality) + self.temporal_embedding(x_mark)    # 添加位置编码+时间编码
            
            modalities.append(modality)
        
        # 对每个模态应用多头注意力机制
        attn_outputs = []
        for i, attn in enumerate(self.attentions):
            # 输入需要增加一个维度以适配 MultiAttention 的输入格式
            modality = modalities[i].unsqueeze(1)  # [batch_size, 1, seq_len, d_model]
            attn_out = attn(modality)  # [batch_size, 1, seq_len, d_model]
            # attn_out = self.layer_norm(attn_out)  # 归一化
            attn_outputs.append(attn_out.squeeze(1))  # [batch_size, seq_len, d_model]
        
        # Concat 4 个模态的多头注意力结果
        combined = torch.cat(attn_outputs, dim=-1)  # [batch_size, seq_len, n_modalities * d_model]
        
        # CNN 处理（调整为 1D 卷积）
        cnn_out = combined.transpose(1, 2)  # [batch_size, n_modalities * d_model, seq_len]
        cnn_out = self.conv1(cnn_out)  # [batch_size, (n_modalities + 7) // 2, seq_len]
        cnn_out = self.elu(cnn_out)
        cnn_out = self.conv2(cnn_out)  # [batch_size, (n_modalities + 7) // 4, seq_len]
        cnn_out = self.elu(cnn_out)
        cnn_out = self.pool(cnn_out)  # [batch_size, (n_modalities + 7) // 4, seq_len // 2]
        
        # 展平并输入全连接层
        cnn_out = cnn_out.view(batch_size, -1)  # [batch_size, fc_input_dim]
        fc_out = self.elu(self.fc1(cnn_out))
        fc_out = self.elu(self.fc2(fc_out))
        output = self.fc3(fc_out)  # [batch_size, pred_len * 1]
        # output = output.view(batch_size, self.pred_len, 1)  # [batch_size, pred_len, 1]
        output = self.output_layer_norm(output.squeeze(-1))  # 归一化输出，去掉最后一个维度
        output = output.view(batch_size, self.pred_len, 1)  # [batch_size, pred_len, 1]
        
        return output