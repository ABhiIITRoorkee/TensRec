import torch
import torch.nn as nn
import torch.nn.functional as F


class HyRec(nn.Module):
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

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.W_u)
        nn.init.xavier_uniform_(self.W_i)

    def forward(self, adj_tensor_u1, adj_tensor_u2, adj_tensor_i1, adj_tensor_i2):
        # Reduce the sparse tensors to 2D by summing or selecting along the 3rd dimension
        adj_tensor_u1 = torch.sparse.sum(adj_tensor_u1, dim=2).float()  # Convert to float
        adj_tensor_i1 = torch.sparse.sum(adj_tensor_i1, dim=2).float()

        # User embeddings
        hu = self.user_embedding.weight.float()  # Ensure this is float32
        embedding_u = [hu]

        for i in range(self.n_layers):
            # Sparse matrix multiplication for user adjacency tensor
            hu_next = torch.sparse.mm(adj_tensor_u1, embedding_u[-1] @ self.W_u)
            hu_next = F.relu(hu_next)
            hu_next = self.dropout_list[i](hu_next)
            embedding_u.append(hu_next)

        # Stack the embeddings across layers and compute the mean
        u_emb = torch.stack(embedding_u, dim=1).mean(dim=1)

        # Item embeddings
        hi = self.item_embedding.weight.float()  # Ensure this is float32
        embedding_i = [hi]

        for i in range(self.n_layers):
            # Sparse matrix multiplication for item adjacency tensor
            hi_next = torch.sparse.mm(adj_tensor_i1, embedding_i[-1] @ self.W_i)
            hi_next = F.relu(hi_next)
            hi_next = self.dropout_list[i](hi_next)
            embedding_i.append(hi_next)

        # Stack the embeddings across layers and compute the mean
        i_emb = torch.stack(embedding_i, dim=1).mean(dim=1)

        return u_emb, i_emb




    # def forward(self, adj_tensor_u1, adj_tensor_u2, adj_tensor_i1, adj_tensor_i2):
    #         # Sum or reduce the sparse tensors along the 3rd dimension to get 2D adjacency matrices
    #         adj_tensor_u1 = torch.sparse.sum(adj_tensor_u1, dim=2).float()
    #         adj_tensor_i1 = torch.sparse.sum(adj_tensor_i1, dim=2).float()

    #         # User embeddings
    #         hu = self.user_embedding.weight.float()
    #         embedding_u = [hu]

    #         for i in range(self.n_layers):
    #             hu_next = torch.sparse.mm(adj_tensor_u1, embedding_u[-1] @ self.W_u)
    #             hu_next = F.relu(hu_next)
    #             hu_next = self.dropout_list[i](hu_next)
    #             embedding_u.append(hu_next)

    #         u_emb = torch.stack(embedding_u, dim=1).mean(dim=1)

    #         # Item embeddings
    #         hi = self.item_embedding.weight.float()
    #         embedding_i = [hi]

    #         for i in range(self.n_layers):
    #             hi_next = torch.sparse.mm(adj_tensor_i1, embedding_i[-1] @ self.W_i)
    #             hi_next = F.relu(hi_next)
    #             hi_next = self.dropout_list[i](hi_next)
    #             embedding_i.append(hi_next)

    #         i_emb = torch.stack(embedding_i, dim=1).mean(dim=1)

    #         return u_emb, i_emb




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import tensorly as tl
# from tensorly.decomposition import parafac

# class HyRec(nn.Module):
#     def __init__(self, n_users, n_items, embedding_dim, layer_num, dropout_list, rank, weight_decay=1e-4):
#         super().__init__()
#         self.n_users = n_users
#         self.n_items = n_items
#         self.embedding_dim = embedding_dim
#         self.rank = rank  # CP Decomposition rank for tensor factorization
#         self.n_layers = layer_num
#         self.weight_decay = weight_decay

#         self.dropout_list = nn.ModuleList()

#         # User and item embeddings
#         self.user_embedding = nn.Embedding(n_users, embedding_dim)
#         self.item_embedding = nn.Embedding(n_items, embedding_dim)

#         # Weight matrices for propagation
#         self.W_u = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
#         self.W_i = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

#         # Gating mechanism for better control between first- and second-order embeddings
#         self.gate = nn.Parameter(torch.rand(1))

#         # Initialize weights
#         self._init_weight_()

#         for i in range(self.n_layers):
#             self.dropout_list.append(nn.Dropout(dropout_list[i]))

#     def _init_weight_(self):
#         nn.init.xavier_uniform_(self.user_embedding.weight)
#         nn.init.xavier_uniform_(self.item_embedding.weight)
#         nn.init.xavier_uniform_(self.W_u)
#         nn.init.xavier_uniform_(self.W_i)

#     def forward(self, adj_tensor_u1, adj_tensor_u2, adj_tensor_i1, adj_tensor_i2):
#         # Reduce sparse tensors along the 3rd dimension
#         adj_tensor_u1 = torch.sparse.sum(adj_tensor_u1, dim=2).float()
#         adj_tensor_i1 = torch.sparse.sum(adj_tensor_i1, dim=2).float()

#         # User embeddings initialization
#         hu = self.user_embedding.weight.float()
#         embedding_u = [hu]

#         # Initialize attention module for users
#         attention_module_u = AttentionModule(self.embedding_dim)

#         for i in range(self.n_layers):
#             # First-order propagation
#             hu_first_order = torch.sparse.mm(adj_tensor_u1, embedding_u[-1] @ self.W_u)

#             # Second-order propagation
#             hu_second_order = torch.sparse.mm(adj_tensor_u1, hu_first_order)

#             # Apply attention mechanism and gating
#             hu_combined = attention_module_u(hu_first_order, hu_second_order)
#             hu_next = self.gate * hu_combined + (1 - self.gate) * embedding_u[-1]  # Weighted skip connection

#             hu_next = F.relu(hu_next)
#             hu_next = self.dropout_list[i](hu_next)
#             embedding_u.append(hu_next)

#         # Final user embedding by averaging across layers
#         u_emb = torch.stack(embedding_u, dim=1).mean(dim=1)

#         # Item embeddings initialization
#         hi = self.item_embedding.weight.float()
#         embedding_i = [hi]

#         # Initialize attention module for items
#         attention_module_i = AttentionModule(self.embedding_dim)

#         for i in range(self.n_layers):
#             # First-order propagation
#             hi_first_order = torch.sparse.mm(adj_tensor_i1, embedding_i[-1] @ self.W_i)

#             # Second-order propagation
#             hi_second_order = torch.sparse.mm(adj_tensor_i1, hi_first_order)

#             # Apply attention mechanism and gating
#             hi_combined = attention_module_i(hi_first_order, hi_second_order)
#             hi_next = self.gate * hi_combined + (1 - self.gate) * embedding_i[-1]  # Weighted skip connection

#             hi_next = F.relu(hi_next)
#             hi_next = self.dropout_list[i](hi_next)
#             embedding_i.append(hi_next)

#         # Final item embedding by averaging across layers
#         i_emb = torch.stack(embedding_i, dim=1).mean(dim=1)

#         return u_emb, i_emb

# class AttentionModule(nn.Module):
#     def __init__(self, embedding_dim):
#         super(AttentionModule, self).__init__()
#         # Define attention weights for first and second-order embeddings (size 2)
#         self.attention_weight = nn.Parameter(torch.Tensor(2))  
#         nn.init.xavier_uniform_(self.attention_weight)

#     def forward(self, first_order_emb, second_order_emb):
#         # Stack the first and second-order embeddings: (batch_size, 2, embedding_dim)
#         stacked_embeddings = torch.stack([first_order_emb, second_order_emb], dim=1)  

#         # Compute attention scores for each (first and second-order): (batch_size, 2)
#         attention_scores = F.softmax(self.attention_weight, dim=0)

#         # Reshape attention scores for broadcasting: (2, 1) to be applied over embedding_dim
#         attention_scores = attention_scores.view(2, 1)

#         # Apply attention scores to the stacked embeddings: (batch_size, embedding_dim)
#         weighted_embedding = (attention_scores * stacked_embeddings).sum(dim=1)

#         return weighted_embedding














































