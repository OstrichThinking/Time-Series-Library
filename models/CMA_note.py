import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding

# TODO 位置编码，卷积npe

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=1000):  # TODO 论文说的d=7是什么意思
        super(PositionalEncoding, self).__init__()
        # 初始化位置编码张量，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)  # [1000, 64]
        # 生成位置索引 [0, 1, ..., max_len-1]，增加维度为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [1000] -> [1000, 1]
        # 计算频率项，用于正弦和余弦函数，长度为 d_model//2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # [32]
        # 偶数列填充正弦值
        pe[:, 0::2] = torch.sin(position * div_term)  # [1000, 32]
        # 奇数列填充余弦值，确保维度匹配
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # [1000, 32]
        # 增加批次维度，变为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)  # [1, 1000, 64]
        # 注册为缓冲区，不参与梯度计算
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入 x: [batch_size, seq_len, d_model]，如 [64, 30, 64]
        seq_len = x.size(1)  # 获取序列长度，例如 30
        # 将位置编码加到输入上，pe 截取前 seq_len 个位置，[1, 30, 64] 广播到 [64, 30, 64]
        return x + self.pe[:, :seq_len, :].to(x.device)  # 输出: [64, 30, 64]

class MultiAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=8, dropout=0.1):
        super(MultiAttention, self).__init__()
        self.d_model = d_model  # 特征维度，默认 64
        self.n_heads = n_heads  # 注意力头数，默认 8
        self.d_k = d_model // n_heads  # 每个头的维度，64 // 8 = 8
        # 定义丢弃层
        self.dropout = nn.Dropout(dropout)
        # 查询的线性变换层
        self.q_linear = nn.Linear(d_model, d_model)  # 输入 64 -> 输出 64
        # 键和值的共享线性变换层
        self.kv_linear = nn.Linear(d_model, d_model)  # 输入 64 -> 输出 64
        # 输出投影层
        self.out_linear = nn.Linear(d_model, d_model)  # 输入 64 -> 输出 64

    def forward(self, x):
        # 输入 x: [batch_size, n_modalities, seq_len, d_model]，如 [64, 4, 30, 64]
        batch_size, n_modalities, seq_len, d_model = x.size()  # 64, 4, 30, 64
        # 计算查询 Q
        Q = self.q_linear(x)  # [64, 4, 30, 64]
        # 计算键和值（共享变换）
        KV = self.kv_linear(x)  # [64, 4, 30, 64]
        K = KV  # K 和 V 相同
        V = KV  # [64, 4, 30, 64]
        # 将 Q 分成多头，形状变换并转置
        Q = Q.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)  # [64, 4, 30, 8, 8] -> [64, 4, 8, 30, 8]
        K = K.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)  # [64, 4, 30, 8, 8] -> [64, 4, 8, 30, 8]
        V = V.view(batch_size, n_modalities, seq_len, self.n_heads, self.d_k).transpose(2, 3)  # [64, 4, 30, 8, 8] -> [64, 4, 8, 30, 8]
        # 对 Q 和 K 进行 L2 归一化，用于余弦相似度计算
        Q_norm = F.normalize(Q, p=2, dim=-1)  # [64, 4, 8, 30, 8]
        K_norm = F.normalize(K, p=2, dim=-1)  # [64, 4, 8, 30, 8]
        # 计算注意力得分（余弦相似度）
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))  # [64, 4, 8, 30, 8] × [64, 4, 8, 8, 30] = [64, 4, 8, 30, 30]
        # 应用 Softmax 归一化注意力得分
        attn = F.softmax(scores, dim=-1)  # [64, 4, 8, 30, 30]
        # 应用 dropout
        attn = self.dropout(attn)  # [64, 4, 8, 30, 30]
        # 加权求和得到上下文
        context = torch.matmul(attn, V)  # [64, 4, 8, 30, 30] × [64, 4, 8, 30, 8] = [64, 4, 8, 30, 8]
        # 转置并重塑回原始形状
        context = context.transpose(2, 3).contiguous().view(batch_size, n_modalities, seq_len, d_model)  # [64, 4, 30, 8, 8] -> [64, 4, 30, 64]
        # 投影输出
        return self.out_linear(context)  # [64, 4, 30, 64]

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len  # 观察窗口长度，例如 30
        self.pred_len = args.pred_len if args.pred_len else self.seq_len  # 预测长度，默认等于 seq_len，例如 30
        self.d_model = args.d_model if args.d_model else 64  # 特征维度，默认 64
        self.n_heads = args.n_heads if args.n_heads else 8  # 注意力头数，默认 8
        self.dropout = args.dropout if args.dropout else 0.1  # 丢弃率，默认 0.1
        self.n_modalities = 4  # 模态数，固定为 4
        self.modality_input_dim = 4  # 每个模态输入维度（1动态+3静态）

        # 模态嵌入层，将每个模态的 4 个特征映射到 d_model
        self.modality_embedding = nn.Linear(self.modality_input_dim, self.d_model)  # 4 -> 64
        # 位置编码模块
        self.pos_encoding = PositionalEncoding(self.d_model)
        # 时间特征嵌入
        self.temporal_embedding = TemporalEmbedding(d_model=args.d_model, embed_type=args.embed,
                                                    freq=args.freq) if args.embed != 'timeF' else TimeFeatureEmbedding(
            d_model=args.d_model, embed_type=args.embed, freq=args.freq)
        # 多头注意力模块
        self.attention = MultiAttention(self.d_model, self.n_heads, self.dropout)
        n_indicators = 7  # 指标总数（3静态+4动态）
        n_pe = 7  # TODO 论文说的所有序列的位置编码大小是什么意思
        # 第一个卷积层，输入通道为 n_modalities * d_model，输出通道为 (n_indicators + n_pe) // 2
        self.conv1 = nn.Conv1d(self.n_modalities * self.d_model, (n_indicators + n_pe) // 2, kernel_size=3, padding=1)  # 256 -> 7
        # 第二个卷积层
        self.conv2 = nn.Conv1d((n_indicators + n_pe) // 2, (n_indicators + n_pe) // 4, kernel_size=3, padding=1)  # 7 -> 3
        # 最大池化层，减半序列长度
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # ELU 激活函数
        self.elu = nn.ELU()
        # 计算全连接层输入维度
        fc_input_dim = (n_indicators + n_pe) // 4 * (self.seq_len // 2)  # 3 * (30 // 2) = 45
        # 全连接层
        self.fc1 = nn.Linear(fc_input_dim, 256)  # 45 -> 256
        self.fc2 = nn.Linear(256, 128)  # 256 -> 128
        # 层归一化，应用于 fc2 输出
        self.layer_norm = nn.LayerNorm(128)
        # 最后一层全连接，输出预测序列
        self.fc3 = nn.Linear(128, self.pred_len)  # 128 -> 30

    def forward(self, x, x_mark=None, y=None, y_mark=None):
        # 输入 x: [batch_size, seq_len, 7]，如 [64, 30, 7]
        batch_size = x.size(0)  # 64
        # 分离静态特征（性别、年龄、BMI）
        static = x[:, :, :3]  # [64, 30, 3]
        # 分离动态特征（SDP, MAP, BT, HR）
        dynamic = x[:, :, 3:]  # [64, 30, 4]
        modalities = []
        # 循环构建 4 个模态
        for i in range(self.n_modalities):  # i = 0, 1, 2, 3
            # 将第 i 个动态特征与静态特征拼接
            modality = torch.cat([dynamic[:, :, i:i+1], static], dim=-1)  # [64, 30, 1] + [64, 30, 3] = [64, 30, 4]
            # 嵌入到 d_model 维度
            modality = self.modality_embedding(modality)  # [64, 30, 64]
            if x_mark is None:
                modality = self.pos_encoding(modality)  # [64, 30, 64] 直接添加位置编码
            else:
                modality = self.pos_encoding(modality) + self.temporal_embedding(x_mark)    # 添加位置编码+时间编码
            modalities.append(modality)
        
        # 堆叠 4 个模态
        combined = torch.stack(modalities, dim=1)  # [64, 4, 30, 64]
        # 通过多头注意力机制
        attn_output = self.attention(combined)  # [64, 4, 30, 64]
        # 重塑为卷积输入格式
        cnn_out = attn_output.view(batch_size, self.n_modalities * self.d_model, self.seq_len)  # [64, 256, 30]
        # 第一个卷积层
        cnn_out = self.elu(self.conv1(cnn_out))  # [64, 7, 30]（padding=1 保持长度）
        # 第二个卷积层
        cnn_out = self.elu(self.conv2(cnn_out))  # [64, 3, 30]
        # 最大池化，序列长度减半
        cnn_out = self.pool(cnn_out)  # [64, 3, 15]
        # 展平为全连接层输入
        cnn_out = cnn_out.view(batch_size, -1)  # [64, 45]（3 * 15）
        # 全连接层处理
        fc_out = self.elu(self.fc1(cnn_out))  # [64, 256]
        fc_out = self.elu(self.fc2(fc_out))  # [64, 128]
        # 层归一化
        fc_out = self.layer_norm(fc_out)  # [64, 128]
        # 输出预测序列
        output = self.fc3(fc_out)  # [64, 30]
        # 增加最后一个维度，符合输出要求
        return output.unsqueeze(-1)  # [64, 30, 1]