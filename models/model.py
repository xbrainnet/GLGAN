import torch
from torch import nn
import torch.nn.functional as F

from config.config import Config


class LocalGraphAttentionLayer(nn.Module):
    """Single local graph-attention layer combining GAT and self-attention."""

    def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float, use_self_attention: bool):
        super().__init__()
        self.dropout = dropout
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.attention = nn.Parameter(torch.empty(2 * out_features, 1))
        self.self_attention = nn.MultiheadAttention(out_features, num_heads=1, dropout=dropout, batch_first=True)
        self.use_self_attention = use_self_attention
        self.leaky_relu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attention, gain=1.414)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(features, self.weight)
        scores = self._attention_scores(projected)
        mask = adjacency > 0
        scores = torch.where(mask, scores, torch.full_like(scores, -9e15))
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        gat_features = torch.matmul(attention, projected)
        if not self.use_self_attention:
            return F.elu(gat_features)
        attended, _ = self.self_attention(gat_features, gat_features, gat_features)
        return F.elu(attended)

    def _attention_scores(self, projected: torch.Tensor) -> torch.Tensor:
        left = torch.matmul(projected, self.attention[: projected.shape[-1], :])
        right = torch.matmul(projected, self.attention[projected.shape[-1] :, :])
        return self.leaky_relu(left + right.transpose(1, 2))


class GlobalGraphAttentionLayer(nn.Module):
    """Single global layer combining PageRank node weights and self-attention."""

    def __init__(self, in_features: int, out_features: int, dropout: float, use_self_attention: bool):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)
        self.self_attention = nn.MultiheadAttention(out_features, num_heads=1, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.use_self_attention = use_self_attention

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor, alpha: float, iterations: int, use_pagerank: bool
    ) -> torch.Tensor:
        if use_pagerank:
            pagerank = pagerank_centrality(adjacency, alpha=alpha, iterations=iterations).unsqueeze(-1)
        else:
            pagerank = torch.ones((*features.shape[:2], 1), device=features.device, dtype=features.dtype)
        projected = self.projection(features)
        weighted = projected * pagerank
        if not self.use_self_attention:
            return F.elu(self.dropout(weighted))
        attended, _ = self.self_attention(weighted, weighted, weighted)
        return F.elu(self.dropout(attended))


class GLGAN(nn.Module):
    """Global-Local Graph Attention Network used for subject classification."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.local_layer = LocalGraphAttentionLayer(
            config.n_regions, config.hidden_dim, config.dropout, config.alpha, config.use_self_attention
        )
        self.global_layer = GlobalGraphAttentionLayer(
            config.n_regions, config.hidden_dim, config.dropout, config.use_self_attention
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.flattened_feature_dim, config.classifier_hidden_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_1, config.classifier_hidden_2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_2, config.n_classes),
        )

    def forward(self, fmri: torch.Tensor, dti: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branch_features = []
        if self.config.use_local_branch:
            branch_features.append(self.local_layer(fmri, dti))
        if self.config.use_global_branch:
            branch_features.append(
                self.global_layer(
                    fmri,
                    dti,
                    alpha=self.config.pagerank_alpha,
                    iterations=self.config.pagerank_iters,
                    use_pagerank=self.config.use_pagerank,
                )
            )
        if not branch_features:
            raise ValueError("At least one model branch must be enabled.")
        features = torch.cat(branch_features, dim=-1).flatten(start_dim=1)
        logits = self.classifier(features)
        return logits, features


def pagerank_centrality(adjacency: torch.Tensor, alpha: float = 0.85, iterations: int = 50) -> torch.Tensor:
    adjacency = torch.clamp(adjacency, min=0)
    row_sums = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    transition = adjacency / row_sums
    n_nodes = adjacency.shape[-1]
    rank = torch.full(adjacency.shape[:-1], 1.0 / n_nodes, device=adjacency.device, dtype=adjacency.dtype)
    teleport = (1.0 - alpha) / n_nodes
    for _ in range(iterations):
        rank = teleport + alpha * torch.matmul(transition.transpose(1, 2), rank.unsqueeze(-1)).squeeze(-1)
    return rank / rank.sum(dim=-1, keepdim=True).clamp_min(1e-8)
