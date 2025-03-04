import torch
import torch.nn as nn
import torch.nn.functional as F

# 组合模型，集成回归模型和分类头
class CombinedModel(nn.Module):
    def __init__(self, regression_model, classification_head):
        super(CombinedModel, self).__init__()
        self.regression_model = regression_model
        self.classification_head = classification_head

    def forward(self, x, x_mark, y, y_mark):
        # 回归模型输出 [batch_size, pred_len, feature_dim]
        regression_out = self.regression_model(x, x_mark, y, y_mark)    # [batch_size, pred_len, feature_dim]
        # 分类头输出 [batch_size, cls_out_dim]
        classification_out = self.classification_head(regression_out)
        return regression_out, classification_out

class InMinClassificationHead(nn.Module):
    def __init__(self, input_dim, d_model, cls_out_dim=1, agg_method='attention', 
                 n_heads=8, dropout=0.1):
        '''
        低血压预测窗口内全局分类
        input_dim: 输入特征维度, 即时间序列特征维度 c_out
        d_model: 目标特征维度
        cls_out_dim: 分类数量
        '''
        super().__init__()
        self.agg_method = agg_method
        self.d_model = d_model
        self.input_dim = input_dim  # c_out

        # 输入特征映射层：将 input_dim 映射到 d_model/2
        self.feature_map = nn.Linear(input_dim, d_model//2)

        # 初始化特征聚合层
        if agg_method == 'attention':
            self.attn_layers = nn.ModuleList([
                nn.MultiheadAttention(d_model//2, n_heads)
                for _ in range(3)  # 3层注意力
            ])
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(d_model//2)
                for _ in range(3)
            ])
        elif agg_method == 'cnn':
            self.conv_block = nn.Sequential(
                nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model//2, kernel_size=3, padding=1)
            )
        elif agg_method == 'lstm':
            self.rnn = nn.LSTM(d_model//2, d_model//2, batch_first=True)
        elif agg_method == 'gru':
            self.rnn = nn.GRU(d_model//2, d_model//2, batch_first=True)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(d_model//2, d_model//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//4, cls_out_dim)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] -> [64, 10, 1]
        batch_size = x.size(0)

        # 特征映射: [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model/2]
        x = self.feature_map(x)  # [64, 10, 512/2]

        if self.agg_method == 'attention':
            # x = x.transpose(0, 1)  # [seq_len, batch_size, d_model/2] -> [10, 64, 512/2]
            # attn_out, _ = self.attn(x, x, x)
            # pooled = attn_out.transpose(0, 1).mean(dim=1)  # [batch_size, d_model/2] -> [64, 512/2]
            x = x.transpose(0, 1)
            for attn, norm in zip(self.attn_layers, self.norm_layers):
                attn_out, _ = attn(x, x, x)
                x = norm(x + attn_out)
            pooled = x.transpose(0, 1).mean(dim=1)
        elif self.agg_method == 'cnn':
            x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len] -> [64, 512/2, 10]
            conv_out = self.conv_block(x)
            pooled = conv_out.mean(dim=-1)  # [batch_size, d_model/2] -> [64, 512/2]
        elif self.agg_method in ['lstm', 'gru']:
            rnn_out, _ = self.rnn(x)  # [batch_size, seq_len, d_model/2] -> [64, 10, 512/2]
            pooled = rnn_out.mean(dim=1)  # [batch_size, d_model/2] -> [64, 512/2]
        
        logits = self.classifier(pooled)  # [batch_size, cls_out_dim] -> [64, 1]
        logits = torch.sigmoid(logits)
        return logits.squeeze(-1)
    
class PerMinClassificationHead(nn.Module):
    """
    低血压预测窗口内逐时间步分类头。
    将每个时间步的预测 MAP 值转换为低血压的预测概率。
    """

    def __init__(self, dropout_prob=0.1):
        """
        初始化 PerMinClassificationHead。
        
        参数:
            dropout_prob (float): dropout 概率，默认为 0.1。
        """
        super(PerMinClassificationHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """
        前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, 1]。
        
        返回:
            torch.Tensor: 输出张量，形状为 [batch_size, seq_len, 1]，表示每个时间步的低血压预测概率。
        """
        return torch.sigmoid(self.net(x))