import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted_GRU, PositionalEmbedding
import numpy as np

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.channel_gate = ChannelGate(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):

        # x:en
        # cross: ex
        B, L, D = cross.shape

        # en patching attention
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # 计算通道注意力 [B, n_vars-1, d_model]
        channel_gates = self.channel_gate(cross)

        # 使用通道注意力作为门控权重，对每个通道进行加权 [B, n_vars-1, d_model]
        gated_ex = cross * channel_gates
        
        # 计算内生变量与加权全局注意力的交叉注意力 en_embed: [B, patch_num + 1, d_model]
        x = x + self.dropout(self.cross_attention(
            x, gated_ex, gated_ex,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)
    
class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    # TODO: 明确glb_token的作用或更加有效的利用glb_token
    def forward(self, x):
        # do patching 将每个变量进行patching
        n_vars = x.shape[1]

        # x: [B, 1, seq_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # x: [B，1, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # x: [B, patch_num, patch_len]
        x = self.value_embedding(x) + self.position_embedding(x)
        # x: [B, patch_num, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # x: [B, patch_num + 1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars
    
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class ChannelGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model // 2, d_model, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
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
        self.hidden_dim = configs.d_model

        # TODO: 讨论 patch_embedding 和 variable-length embedding 的优劣
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        # TODO: 讨论 使用 LSTM, CNN, Linear 的优劣
        self.ex_embedding = DataEmbedding_inverted_GRU(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)


    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, D] [64, 450, 23]
        # x_mark_enc: [B, L, D] [64, 450, 1]
        
        if self.use_norm:
            # Normalization from Non-stationa
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        
        # 内生变量 'Solar8000/ART_MBP_window_sample' 做细粒度的patch_embedding en_embed: [B, patch_num + 1, d_model] eg.[64, 29, 256]
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))

        # 外生变量 使用 variable-length embedding [B, n_vars-1, d_model]12
        # 单变量时序预测时候，内生变量和外生变量为同一个
        if self.n_vars > 1:
            ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
        else:
            ex_embed = self.ex_embedding(x_enc[:, :, -1].unsqueeze(-1), x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)

        # 预测头 dec_out: [B, n_vars, pred_len]
        dec_out = self.head(enc_out)
        dec_out = dec_out.unsqueeze(-1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
        