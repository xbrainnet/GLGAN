import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.config import Config

class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class ContrastiveLearningHead(nn.Module):
    """对比学习头"""
    
    def __init__(self, input_dim):
        super(ContrastiveLearningHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, Config.PROJECTION_DIM1),
            nn.ReLU(),
            nn.Linear(Config.PROJECTION_DIM1, Config.PROJECTION_DIM2)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)

class GLGAT(nn.Module):
    """全局-局部图注意力网络"""
    
    def __init__(self):
        super(GLGAT, self).__init__()
        
        # 图注意力层
        self.gat1 = GraphAttentionLayer(
            Config.N_FEAT, Config.N_HID, 
            Config.DROPOUT, Config.ALPHA
        )
        self.gat2 = GraphAttentionLayer(
            Config.N_HID, Config.N_CLASS,
            Config.DROPOUT, Config.ALPHA, concat=False
        )
        
        # 全局注意力机制
        self.global_attention = nn.MultiheadAttention(
            embed_dim=Config.EMBED_DIM,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True
        )
        
        # 对比学习头
        self.contrastive_head = ContrastiveLearningHead(Config.EMBED_DIM)
        
        # 分类器
        classifier_layers = []
        dims = Config.CLASSIFIER_DIMS
        for i in range(len(dims) - 1):
            classifier_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if i < len(dims) - 2 else nn.Identity(),
                nn.Dropout(Config.DROPOUT) if i < len(dims) - 2 else nn.Identity()
            ])
        self.classifier = nn.Sequential(*classifier_layers)
        
        self.dropout = nn.Dropout(Config.DROPOUT)
        
    def forward(self, x, adj, return_attention=False):
        """
        前向传播
        
        Args:
            x: 节点特征 [batch_size, n_nodes, n_features]
            adj: 邻接矩阵 [batch_size, n_nodes, n_nodes]
            return_attention: 是否返回注意力权重
        """
        batch_size = x.size(0)
        outputs = []
        attention_weights = []
        
        for i in range(batch_size):
            # 局部GAT处理
            h = self.dropout(x[i])
            h = self.gat1(h, adj[i])
            h = self.dropout(h)
            h = self.gat2(h, adj[i])
            outputs.append(h)
        
        # 堆叠输出
        gat_output = torch.stack(outputs, dim=0)  # [batch_size, n_nodes, n_class]
        
        # 展平用于全局注意力
        flattened = gat_output.view(batch_size, -1).unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 全局注意力
        global_output, attn_weights = self.global_attention(
            flattened, flattened, flattened
        )
        global_output = global_output.squeeze(1)  # [batch_size, embed_dim]
        
        # 对比学习特征
        contrastive_features = self.contrastive_head(global_output)
        
        # 分类
        logits = self.classifier(global_output)
        
        if return_attention:
            return logits, contrastive_features, attn_weights
        else:
            return logits, contrastive_features

def pagerank_centrality(adj_matrix, alpha=0.85, max_iter=100, tol=1e-6):
    """计算PageRank中心性"""
    n = adj_matrix.shape[0]
    
    # 归一化邻接矩阵
    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 避免除零
    transition_matrix = adj_matrix / row_sums[:, np.newaxis]
    
    # 初始化PageRank向量
    pr = np.ones(n) / n
    
    for _ in range(max_iter):
        pr_new = (1 - alpha) / n + alpha * np.dot(transition_matrix.T, pr)
        
        if np.linalg.norm(pr_new - pr) < tol:
            break
        pr = pr_new
    
    return pr
