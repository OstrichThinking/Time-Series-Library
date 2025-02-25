import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # 1. 个体特征提取模块 (使用 1D CNN)
        self.encoders = nn.ModuleList([
            nn.Conv1d(1, configs.d_model, kernel_size=3, padding=1)  # 输入通道为 1
            for _ in range(self.n_vars)
        ])

        # 2. 自适应关系选择模块
        # 全局注意力机制 (共享全局上下文)
        self.global_attention = nn.MultiheadAttention(configs.d_model, configs.n_heads, dropout=configs.dropout)

        # MLP 预测 Ki
        self.mlp_k = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, 1)
        )
        # 计算 Si 的线性层
        self.correlation_layer = nn.Linear(configs.d_model * 2, self.n_vars)

        # 3. 关系融合模块 (局部 Transformer)
        self.local_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(configs.d_model, configs.n_heads, dim_feedforward=configs.d_model*2, dropout=configs.dropout),
            num_layers=1
        )
        # 预测头 (输出 pred_len 步)
        self.predictor = nn.Linear(configs.d_model, configs.pred_len)

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.en_embedding(x_enc)
        x_dec = self.ex_embedding(x_dec)

        x_enc = self.encoder(x_enc, x_mark_enc)
        x_dec = self.decoder(x_dec, x_mark_dec)
        