import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=1000):  # 与实验一致
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class MultiAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=8, dropout=0.1):
        super(MultiAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, n_modalities, seq_len, d_model = x.size()
        Q = self.q_linear(x)
        KV = self.kv_linear(x)
        K = KV
        V = KV
        Q = Q.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        K = K.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        V = V.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(2, 3).contiguous().view(batch_size, n_modalities, seq_len, d_model)
        return self.out_linear(context)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len if args.pred_len else self.seq_len
        self.d_model = args.d_model if args.d_model else 64
        self.n_heads = args.n_heads if args.n_heads else 8
        self.dropout = args.dropout if args.dropout else 0.1
        self.n_modalities = 4
        self.modality_input_dim = 4

        self.modality_embedding = nn.Linear(self.modality_input_dim, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)
        self.attention = MultiAttention(self.d_model, self.n_heads, self.dropout)
        n_indicators = 7
        n_pe = 7
        self.conv1 = nn.Conv1d(self.n_modalities * self.d_model, (n_indicators + n_pe) // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d((n_indicators + n_pe) // 2, (n_indicators + n_pe) // 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.elu = nn.ELU()
        fc_input_dim = (n_indicators + n_pe) // 4 * (self.seq_len // 2)
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, self.pred_len)

    def forward(self, x, x_mark=None, y=None, y_mark=None):
        batch_size = x.size(0)
        static = x[:, :, :3]
        dynamic = x[:, :, 3:]
        modalities = []
        for i in range(self.n_modalities):
            modality = torch.cat([dynamic[:, :, i:i+1], static], dim=-1)
            modality = self.modality_embedding(modality)
            modality = self.pos_encoding(modality)
            modalities.append(modality)
        
        combined = torch.stack(modalities, dim=1)
        attn_output = self.attention(combined)
        cnn_out = attn_output.view(batch_size, self.n_modalities * self.d_model, self.seq_len)
        cnn_out = self.elu(self.conv1(cnn_out))
        cnn_out = self.elu(self.conv2(cnn_out))
        cnn_out = self.pool(cnn_out)
        cnn_out = cnn_out.view(batch_size, -1)
        fc_out = self.elu(self.fc1(cnn_out))
        fc_out = self.elu(self.fc2(fc_out))
        fc_out = self.layer_norm(fc_out)
        output = self.fc3(fc_out)
        return output.unsqueeze(-1)