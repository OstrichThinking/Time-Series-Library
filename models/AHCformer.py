import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
    

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
        self.hidden_dim = configs.d_model

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
        """
        前向传播
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, n_channel)
        返回:
            output (torch.Tensor): 预测结果，形状为 (batch_size, pred_len, 1)
        """
        batch_size, seq_len, n_channel = x_enc.size()

        # 1. 个体特征提取
        h = torch.zeros(batch_size, n_channel, self.hidden_dim, seq_len).to(x_enc.device)
        for i in range(n_channel):
            # 对每个通道独立应用 CNN
            xi = x_enc[:, :, i].unsqueeze(1)  # (batch, 1, seq_len)
            hi = self.encoders[i](xi)  # (batch, hidden_dim, seq_len)
            h[:, i, :, :] = hi.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)

        # 取最后一时间步的特征
        h_last = h[..., -1]  # (batch, n_channel, hidden_dim)

        # 2. 自适应关系选择
        # 共享全局上下文 (batch_first=True)
        C, _ = self.global_attention(h_last, h_last, h_last)  # (batch, n_channel, hidden_dim)

        # 计算 Ki 和 Si
        outputs = []
        for i in range(n_channel):
            # 拼接 hi 和 C[i]
            hi = h_last[:, i, :]  # (batch, hidden_dim)
            ci = C[:, i, :]  # (batch, hidden_dim)
            input_mlp = torch.cat([hi, ci], dim=-1)  # (batch, 2*hidden_dim)

            # MLP 预测 Ki
            ki_raw = self.mlp_k(input_mlp)  # (batch, 1)
            ki_norm = torch.sigmoid(ki_raw) * (self.n_channel - 1)
            ki = torch.floor(ki_norm).long()  # (batch,)

            # 计算相关性得分 Si
            si = torch.sigmoid(self.correlation_layer(input_mlp))  # (batch, n_channel)

            # 选择 Top-Ki 相关变量
            _, topk_indices = torch.topk(si, ki.max().item(), dim=-1)  # (batch, ki)

            # 3. 关系融合 (局部 Transformer)
            related_h = h_last[:, topk_indices, :]  # (batch, ki, hidden_dim)
            local_input = torch.cat([h_last[:, i:i+1, :], related_h], dim=1)  # (batch, ki+1, hidden_dim)
            zi = self.local_transformer(local_input)[:, 0, :]  # (batch, hidden_dim)

            # 预测 (输出 pred_len 步)
            yi = self.predictor(zi)  # (batch, pred_len)
            outputs.append(yi)

        # 整合输出
        output = torch.stack(outputs, dim=1)  # (batch, n_channel, pred_len)
        # 假设只预测一个全局值，平均池化通道维度
        output = output.mean(dim=1, keepdim=False).unsqueeze(-1)  # (batch, pred_len, 1)
        return output
        