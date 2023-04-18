import math

import torch
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self, n_ents, n_rels, norm=1, hidden_size=256, margin=1.0):
        super().__init__()
        self.n_ents = n_ents
        self.n_rels = n_rels
        self.norm = norm
        self.hidden_size = hidden_size
        self.margin = margin
        self.uniform_range = 6 / math.sqrt(self.hidden_size)
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

        self.ent_emb = nn.Embedding(
            num_embeddings=self.n_ents + 1,
            embedding_dim=self.hidden_size,
            padding_idx=self.n_ents
        )
        self.ent_emb.weight.data.uniform_(-self.uniform_range, self.uniform_range)
        self.rel_emb = nn.Embedding(
            num_embeddings=self.n_rels + 1,
            embedding_dim=self.hidden_size,
            padding_idx=self.n_rels
        )
        self.rel_emb.weight.data.uniform_(-self.uniform_range, self.uniform_range)
        self.rel_emb.weight.data[:-1, :].div_(self.rel_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))

    def forward(self, pos, neg):
        self.ent_emb.weight.data[:-1, :].div_(self.ent_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        self.rel_emb.weight.data[:-1, :].div_(self.rel_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        pos_dist = self._distance(pos)
        neg_dist = self._distance(neg)
        return self.loss_fn(pos_dist, neg_dist, -1 * torch.ones_like(pos_dist).to(pos_dist.device)), pos_dist, neg_dist

    def predict(self, batch):
        return self.ent_emb(batch)

    def _distance(self, triplets):
        return (self.ent_emb(triplets[:, 0]) + self.rel_emb(triplets[:, 1]) - self.ent_emb(triplets[:, 2])).norm(p=self.norm, dim=1)
    