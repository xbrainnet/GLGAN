import numpy as np
import torch
from torch import nn
from torch.nn import Parameter, Module, init
import torch.nn.functional as F
import networkx as nx
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        # 定义多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # 定义查询、键、值的线性变换
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x = x.permute(2, 0, 1) #(sequence_length, batch_size, embed_dim)
        # 对输入进行线性变换得到查询、键、值
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        # 计算多头注意力
        attn_output, attn_output_weights = self.multihead_attn(Q, K, V)

        return attn_output, attn_output_weights
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=90):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class LocalGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(LocalGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.self_attention = SelfAttention(8100, 8100)
        self.Tanh = nn.Tanh()
        

        self.W = Parameter(torch.empty(in_features, out_features))
        init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        batch_size = adj.size(0)
        edge_att = adj.view(batch_size, 8100)
        edge_att = self.self_attention(edge_att)
        edge_att = self.Tanh(edge_att)
        edge_att = edge_att.view(batch_size, 90, -1)

        Wh = torch.matmul(h, self.W)
        e = self.prepare_attentional_mechanism_input(Wh, batch_size)
        zero_vec = -9e15 * torch.ones_like(e)
        node_att = torch.where(adj > 0, e, zero_vec)
        node_att = F.softmax(node_att, dim=2)
        
        local_att = torch.add(node_att, edge_att)
        local_att = F.dropout(local_att, self.dropout)
        local_feature = torch.matmul(local_att, Wh)
        return local_feature

    def prepare_attentional_mechanism_input(self, Wh, batch_size):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        Wh2 = Wh2.view(batch_size, 1, -1)
        e = Wh1 + Wh2
        return self.leakyrelu(e)

class GlobalGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GlobalGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.self_attention = SelfAttention(8100, 8100)
        self.Tanh = nn.Tanh()
        self.attention = Attention(8100)

        self.W = Parameter(torch.empty(in_features, out_features))
        init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, h, adj):
        adj = adj.cpu()
        batch_size = adj.size(0)

        node_att = h.view(batch_size, 8100)
        node_att = self.self_attention(node_att)
        node_att = self.Tanh(node_att)
        node_att = node_att.view(batch_size, 90, -1)

        Wh = torch.matmul(h, self.W)
        p_list = []
        for i in range(batch_size):
            matrix = adj.numpy()
            matrix = matrix[i]
            matrix = np.reshape(matrix, (90, 90))
            matrix = nx.from_numpy_array(matrix)
            graph = nx.DiGraph(matrix)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            pagerank = nx.pagerank(graph, alpha=0.85)
            values = pagerank.values()
            a = list(values)
            pagerank = np.array(a)
            pagerank = pagerank.reshape(90, 1)
            p_list.append(pagerank)
        p_list = np.array(p_list)
        pagerank = torch.Tensor(p_list)

        zero_vec = -9e15 * torch.ones_like(pagerank)
        adj = adj.clone().detach().requires_grad_(True)
        edge_att = torch.where(adj > 0, pagerank, zero_vec)
        edge_att = F.softmax(edge_att, dim=2)
        edge_att = edge_att.to(DEVICE)
        node_att = node_att.to(DEVICE)
        
        global_att = torch.add(node_att, edge_att)
        global_att = F.dropout(global_att, self.dropout)
        global_feature = torch.matmul(global_att, Wh)
        return global_feature

class GLGAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GLGAT, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.localGAL = LocalGraphAttentionLayer(nfeat, nhid, dropout=self.dropout, alpha=self.alpha)
        self.globalGAL = GlobalGraphAttentionLayer(nfeat, nhid, dropout=self.dropout, alpha=self.alpha)
        self.attention = Attention(8100)

        self.classify1 = nn.Linear(8100, 1024)
        self.classify2 = nn.Linear(1024, 128)
        self.classify3 = nn.Linear(128, 2)
        self.logs = nn.LogSoftmax(dim=1)

        self.fc1 = torch.nn.Linear(8100, 2048)
        self.fc2 = torch.nn.Linear(2048, 256)

    def forward(self, x, adj):
        loss_contrast = 0.0
        batch_size = adj.size(0)
        local_feature = F.relu(self.localGAL(x, adj))
        local_feature = self.localGAL(local_feature, adj).view(batch_size, -1)
        emb1 = local_feature.view(-1, 90)
        global_feature = F.relu(self.globalGAL(x, adj))
        global_feature = self.globalGAL(global_feature, adj).view(batch_size, -1)
        emb2 = global_feature.view(-1, 90)

        all_feature = torch.stack([local_feature, global_feature], dim=1) 
        all_feature, att = self.attention(all_feature)
        out = self.classify1(all_feature)
        out = self.classify2(out)
        out = self.classify3(out)
        out = self.logs(out)
        
        emb1 = self.transform(emb1)
        emb2 = self.transform(emb2)
        emb1 = emb1.view(1, 8100)
        emb2 = emb2.view(1, 8100)
        loss_contrast = self.contrast_loss1(emb1, emb2)
        loss_contrast = torch.abs(loss_contrast)
        return out, loss_contrast

    def transform(self, x):
        x = x.view(-1, 90, 90)
        x = x[:].sum(dim=0)
        return x