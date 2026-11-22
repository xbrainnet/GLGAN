import math

import torch
from torch import nn
import torch.nn.functional as F

from config.config import Config


class LocalGraphAttentionLayer(nn.Module):
    """Aggregate structural neighbors using functional GAT and DTI attention."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        alpha: float,
        use_self_attention: bool,
        *,
        n_regions: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.value_proj = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.Parameter(torch.empty(2 * out_features, 1))
        self.use_self_attention = use_self_attention
        if use_self_attention:
            dti_features = in_features if n_regions is None else n_regions
            self.dti_q_proj = nn.Linear(dti_features, out_features)
            self.dti_k_proj = nn.Linear(dti_features, out_features)
        self.leaky_relu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attention, gain=1.414)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        values = self.value_proj(features)
        mask = adjacency > 0
        beta = masked_softmax(self._attention_scores(values), mask, dim=-1)
        weights = F.dropout(beta, self.dropout, training=self.training)
        if self.use_self_attention:
            dti = adjacency.to(dtype=features.dtype)
            queries = self.dti_q_proj(dti)
            keys = self.dti_k_proj(dti)
            scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])
            structural_attention = F.softmax(scores, dim=-1)
            weights = (weights + structural_attention).masked_fill(~mask, 0.0)
        return F.elu(torch.matmul(weights, values))

    def _attention_scores(self, projected: torch.Tensor) -> torch.Tensor:
        left = torch.matmul(projected, self.attention[: projected.shape[-1], :])
        right = torch.matmul(projected, self.attention[projected.shape[-1] :, :])
        return self.leaky_relu(left + right.transpose(1, 2))


class GlobalGraphAttentionLayer(nn.Module):
    """Aggregate structural neighbors using PageRank and functional attention."""

    def __init__(self, in_features: int, out_features: int, dropout: float, use_self_attention: bool):
        super().__init__()
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.q_proj = nn.Linear(in_features, out_features)
            self.k_proj = nn.Linear(in_features, out_features)
        self.value_proj = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        alpha: float,
        iterations: int,
        use_pagerank: bool,
        *,
        pagerank_weighted: bool = False,
    ) -> torch.Tensor:
        mask = adjacency > 0
        if use_pagerank:
            hub_scores = pagerank_centrality(adjacency, alpha=alpha, iterations=iterations, weighted=pagerank_weighted)
        else:
            hub_scores = features.new_zeros(features.shape[:2])
        weights = masked_softmax(hub_scores.unsqueeze(1).expand_as(adjacency), mask, dim=-1)
        if self.use_self_attention:
            queries = self.q_proj(features)
            keys = self.k_proj(features)
            scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])
            functional_attention = F.softmax(scores, dim=-1)
            weights = (weights + functional_attention).masked_fill(~mask, 0.0)

        aggregated = torch.matmul(weights, self.value_proj(features))
        return F.elu(self.dropout(aggregated))


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize over valid entries and return zeros for fully masked rows."""
    masked_scores = scores.masked_fill(~mask, -torch.inf)
    # Avoid softmax over an entire row of negative infinities.
    masked_scores = masked_scores.masked_fill(~mask.any(dim=dim, keepdim=True), 0.0)
    return F.softmax(masked_scores, dim=dim).masked_fill(~mask, 0.0)


class BidirectionalAdjacencyRefiner(nn.Module):
    """Refine existing structural edges with a single functional attention update."""

    def __init__(self, n_regions: int, hidden_dim: int, epsilon: float = 0.1):
        super().__init__()
        self.query = nn.Linear(n_regions, hidden_dim)
        self.key = nn.Linear(n_regions, hidden_dim)
        self.epsilon = epsilon

    def forward(self, fmri: torch.Tensor, dti: torch.Tensor) -> torch.Tensor:
        queries, keys = self.query(fmri), self.key(fmri)
        scores = queries @ keys.transpose(1, 2) / math.sqrt(queries.shape[-1])
        mask = dti > 0
        attention = masked_softmax(scores, mask)
        refined = dti.masked_fill(~mask, 0.0) + self.epsilon * attention
        row_sum = refined.sum(dim=-1, keepdim=True)
        return refined / row_sum.masked_fill(row_sum == 0, 1.0)


def class_conditional_view_loss(
    global_view: torch.Tensor, local_view: torch.Tensor, labels: torch.Tensor, tau: float = 0.4
) -> torch.Tensor:
    """Average the negative log probability of same-class partners in both directions."""
    global_view = F.normalize(global_view.flatten(start_dim=1), dim=1)
    local_view = F.normalize(local_view.flatten(start_dim=1), dim=1)
    similarity = (global_view @ local_view.T) / tau
    positives = labels[:, None] == labels[None, :]
    positive_logits = similarity.masked_fill(~positives, -torch.inf)
    global_to_local = similarity.logsumexp(dim=1) - positive_logits.logsumexp(dim=1)
    local_to_global = similarity.logsumexp(dim=0) - positive_logits.logsumexp(dim=0)
    return (global_to_local.mean() + local_to_global.mean()) / 2


class GLGAN(nn.Module):
    """Global-Local Graph Attention Network used for subject classification."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tau = config.tau
        self.refiner = (
            BidirectionalAdjacencyRefiner(config.n_regions, config.hidden_dim)
            if config.coupling_mode == "bidirectional"
            else None
        )
        self.local_layer = LocalGraphAttentionLayer(
            config.n_regions, config.hidden_dim, config.dropout, config.alpha, config.use_self_attention
        )
        self.global_layer = GlobalGraphAttentionLayer(
            config.n_regions, config.hidden_dim, config.dropout, config.use_self_attention
        )
        if config.use_local_branch and config.use_global_branch:
            self.projector = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.roi_attention = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, 1, bias=False),
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

    def forward(
        self, fmri: torch.Tensor, dti: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and a scalar contrastive loss, zero for a single branch."""
        embedding, contrastive_loss = self.fused_embedding(fmri, dti, labels=labels)
        return self.classifier(embedding.flatten(start_dim=1)), contrastive_loss

    def fused_embedding(
        self, fmri: torch.Tensor, dti: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.refiner is not None:
            dti = self.refiner(fmri, dti)
        z_local = self.local_layer(fmri, dti) if self.config.use_local_branch else None
        z_global = None
        if self.config.use_global_branch:
            z_global = self.global_layer(
                fmri,
                dti,
                alpha=self.config.pagerank_alpha,
                iterations=self.config.pagerank_iters,
                use_pagerank=self.config.use_pagerank,
                pagerank_weighted=self.config.pagerank_weighted,
            )

        if z_global is not None and z_local is not None:
            z_final, contrastive_loss = self._fuse_branches(z_global, z_local, labels)
        else:
            z_final = z_global if z_local is None else z_local
            assert z_final is not None
            contrastive_loss = z_final.new_zeros(())

        return z_final, contrastive_loss

    def _fuse_branches(
        self, z_global: torch.Tensor, z_local: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_projection = F.normalize(self.projector(z_global), dim=-1)
        local_projection = F.normalize(self.projector(z_local), dim=-1)
        similarity = torch.matmul(global_projection, local_projection.transpose(1, 2)) / self.tau
        p_global = F.softmax(similarity, dim=-1).diagonal(dim1=1, dim2=2)
        p_local = F.softmax(similarity.transpose(1, 2), dim=-1).diagonal(dim1=1, dim2=2)
        if self.config.contrastive_mode == "roi_infonce":
            positive_prob = ((p_global + p_local) / 2).mean(dim=-1)
            contrastive_loss = -torch.log(positive_prob.clamp_min(1e-8)).mean()
        elif self.training:
            if labels is None or labels.shape != (z_global.shape[0],):
                raise ValueError("Class-conditional training requires one label per subject.")
            contrastive_loss = class_conditional_view_loss(global_projection, local_projection, labels, self.tau)
        else:
            contrastive_loss = z_global.new_zeros(())

        delta_global = self.roi_attention(z_global).squeeze(-1)
        delta_local = self.roi_attention(z_local).squeeze(-1)
        attention = F.softmax(torch.stack((delta_global, delta_local), dim=-1), dim=-1)
        gamma_global = attention[..., 0] + p_global
        gamma_local = attention[..., 1] + p_local
        z_final = gamma_global.unsqueeze(-1) * z_global + gamma_local.unsqueeze(-1) * z_local
        return z_final, contrastive_loss


def pagerank_centrality(
    adjacency: torch.Tensor,
    alpha: float = 0.85,
    iterations: int = 50,
    *,
    weighted: bool = False,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Compute PageRank with uniform redistribution from nodes without outgoing edges."""
    n_nodes = adjacency.shape[-1]
    dtype = adjacency.dtype if adjacency.is_floating_point() else torch.get_default_dtype()
    weights = adjacency.clamp_min(0) if weighted else (adjacency > 0)
    weights = weights.to(dtype=dtype)
    degree = weights.sum(dim=-1, keepdim=True)
    dangling = degree == 0
    transition = weights / degree.masked_fill(dangling, 1.0)

    rank = weights.new_full(adjacency.shape[:-1], 1.0 / n_nodes)
    teleport = (1.0 - alpha) / n_nodes
    for _ in range(iterations):
        incoming = torch.matmul(rank.unsqueeze(-2), transition).squeeze(-2)
        dangling_mass = (rank * dangling.squeeze(-1)).sum(dim=-1, keepdim=True)
        next_rank = teleport + alpha * (incoming + dangling_mass / n_nodes)
        converged = (next_rank - rank).abs().sum(dim=-1) <= tolerance
        rank = next_rank
        if converged.all():
            break
    return rank
