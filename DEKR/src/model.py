import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from data_loader import DataLoader


class DEKR(torch.nn.Module):
    def __init__(self, num_user, num_item, num_ent, num_rel, kg, args, device):
        super(DEKR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size

        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)

        self._gen_adj()

        self.usr = torch.nn.Embedding(num_user, args.dim)
        self.itm = torch.nn.Embedding(num_item, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)

        self.embed_size = args.dim
        self.embed_size_add = args.dim
        self.desc_embed_size = args.desc_dim
        self.dropout = 0.1

        data_loader = DataLoader(args.dataset)
        self.entity_desc_embed_vector = data_loader.get_entity_desc_embed_vector()
        self.desc_embedding = torch.nn.Embedding(num_embeddings=(self.num_user + self.num_item), embedding_dim=self.desc_embed_size)
        self.desc_embedding.weight.data.copy_(torch.from_numpy(self.entity_desc_embed_vector))
        self.desc_embedding.weight.requires_grad = False

        self.desc_reduce = nn.Sequential(
            nn.Linear(self.desc_embed_size, self.embed_size_add),
            nn.ReLU()
        )

        self.MLP_layer123 = nn.Sequential(
            nn.Linear(self.embed_size_add * 2, self.embed_size_add * 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(self.embed_size_add * 2, self.embed_size_add),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(self.embed_size_add, self.embed_size_add // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.NeuMF = nn.Sequential(
            nn.Linear((self.embed_size_add + self.embed_size_add // 2), 1),
            nn.Sigmoid()
        )

    def _gen_adj(self):

        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def forward(self, user_index, item_index):

        batch_size = user_index.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        user_index = user_index.view((-1, 1))
        item_index = item_index.view((-1, 1))

# ---------------------------------Encoding---------------------------------
        # Graph structure information
        user_graph_embeddings = self.ent(user_index).squeeze(dim=1)
        item_entities, item_relations = self._get_neighbors(item_index)
        item_graph_embeddings = self._aggregate(user_graph_embeddings, item_entities, item_relations)
        user_entities, user_relations = self._get_neighbors(user_index)
        user_graph_embeddings = self._aggregate(item_graph_embeddings, user_entities, user_relations)

        # Entity description information
        user_desc_embeddings = self.desc_embedding(user_index).squeeze(dim=1)
        item_desc_embeddings = self.desc_embedding(item_index).squeeze(dim=1)
        user_desc_embeddings = self.desc_reduce(user_desc_embeddings)
        item_desc_embeddings = self.desc_reduce(item_desc_embeddings)
# ---------------------------------Predicting---------------------------------
        # prediction based on the graph
        scores_graph = (user_graph_embeddings * item_graph_embeddings).sum(dim=1)
        scores_graph = torch.sigmoid(scores_graph)

        # prediction based on the description
        user_item_desc_l = torch.mul(user_desc_embeddings, item_desc_embeddings)
        user_item_desc_cat = torch.cat((user_desc_embeddings, item_desc_embeddings), dim=1)
        user_item_desc_nl = self.MLP_layer123(user_item_desc_cat)
        user_item_desc = torch.cat((user_item_desc_l, user_item_desc_nl), dim=1)
        scores_desc = self.NeuMF(user_item_desc)

        return scores_graph.squeeze(), scores_desc.squeeze()

# ----------------------------------------------------------------------------
    # Information Propagation over the Knowledge Graph.
    def _get_neighbors(self, v):

        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    # Aggregating the information of neighboring nodes to the core entity.
    def _aggregate(self, side_embeddings, entities, relations):

        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    side_embeddings=side_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.dim))


class Aggregator(torch.nn.Module):

    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.aggregator = aggregator
        if aggregator == 'sum':
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, side_embeddings, act):

        batch_size = side_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._agg_neighbor_vectors(neighbor_vectors, neighbor_relations, side_embeddings)
        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
        else:
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))

    def _agg_neighbor_vectors(self, neighbor_vectors, neighbor_relations, side_embeddings):

        side_embeddings = side_embeddings.view((self.batch_size, 1, 1, self.dim))
        side_relation_scores = (side_embeddings * neighbor_relations).sum(dim=-1)
        side_relation_scores_normalized = F.softmax(side_relation_scores, dim=-1)
        side_relation_scores_normalized = side_relation_scores_normalized.unsqueeze(dim=-1)
        neighbors_aggregated = (side_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated
