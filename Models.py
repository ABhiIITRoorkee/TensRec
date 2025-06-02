import torch
import torch.nn as nn
import torch.nn.functional as F




class TensRec(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, layer_num, dropout_list, rank):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.rank = rank  # CP Decomposition rank for tensor factorization
        
        self.n_layers = layer_num
        self.dropout_list = nn.ModuleList()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize W_u and W_i with the correct dimensions
        self.W_u = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W_i = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        self._init_weight_()

        for i in range(self.n_layers):
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        # Define attention layers for users and items
        self.attn_user = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

        self.attn_item = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.W_u)
        nn.init.xavier_uniform_(self.W_i)

    def forward(self, adj_tensor_u1, adj_tensor_u2, adj_tensor_i1, adj_tensor_i2):
        adj_tensor_u1 = torch.sparse.sum(adj_tensor_u1, dim=2).float()
        adj_tensor_i1 = torch.sparse.sum(adj_tensor_i1, dim=2).float()

        # User embeddings
        hu = self.user_embedding.weight.float()
        embedding_u = [hu]

        for i in range(self.n_layers):
            hu_next = torch.sparse.mm(adj_tensor_u1, embedding_u[-1] @ self.W_u)
            hu_next = F.relu(hu_next)
            hu_next = self.dropout_list[i](hu_next)
            embedding_u.append(hu_next)

        # Attention-based aggregation for user embeddings
        embedding_u_stack = torch.stack(embedding_u, dim=1)  # shape: [n_users, n_layers+1, emb_dim]
        attention_scores_u = torch.softmax(self.attn_user(embedding_u_stack), dim=1)  # [n_users, n_layers+1, 1]
        u_emb = torch.sum(attention_scores_u * embedding_u_stack, dim=1)  # weighted sum

        # Item embeddings
        hi = self.item_embedding.weight.float()
        embedding_i = [hi]

        for i in range(self.n_layers):
            hi_next = torch.sparse.mm(adj_tensor_i1, embedding_i[-1] @ self.W_i)
            hi_next = F.relu(hi_next)
            hi_next = self.dropout_list[i](hi_next)
            embedding_i.append(hi_next)

        # Attention-based aggregation for item embeddings
        embedding_i_stack = torch.stack(embedding_i, dim=1)  # shape: [n_items, n_layers+1, emb_dim]
        attention_scores_i = torch.softmax(self.attn_item(embedding_i_stack), dim=1)  # [n_items, n_layers+1, 1]
        i_emb = torch.sum(attention_scores_i * embedding_i_stack, dim=1)

        return u_emb, i_emb

   
