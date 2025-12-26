import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    简单的图卷积层
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: [B, N, in_features]
            adj: [B, N, N]
        Returns:
            out: [B, N, out_features]
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias


class TopologyGNN(nn.Module):
    """
    拓扑预测GNN模块
    结合GCN和GRU进行迭代细化
    """
    
    def __init__(self, 
                 embed_dim=256,
                 num_layers=6,
                 dropout=0.1):
        """
        Args:
            embed_dim: 特征维度
            num_layers: GNN层数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        self.gcn = GraphConvolution(embed_dim, embed_dim)
        
        self.gru = nn.GRUCell(embed_dim, embed_dim)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, node_features, init_adj=None):
        """
        前向传播
        
        Args:
            node_features: [B, N, D], 节点特征（来自扩散模型）
            init_adj: [B, N, N], 初始邻接矩阵（可选）
        
        Returns:
            adj_matrix: [B, N, N], 预测的邻接矩阵
            adj_logits: [B, N, N], 邻接矩阵的logits
        """
        B, N, D = node_features.shape
        device = node_features.device
        
        if init_adj is None:
            adj = torch.zeros(B, N, N, device=device)
        else:
            adj = init_adj
        
        h = torch.zeros(B * N, D, device=device)
        
        adj_logits_list = []
        
        for layer in range(self.num_layers):
            node_features_flat = node_features.view(B * N, D)
            
            h = self.gru(node_features_flat, h)
            
            h_reshaped = h.view(B, N, D)
            
            gcn_out = self.gcn(h_reshaped, adj)
            gcn_out = self.layer_norm(gcn_out)
            
            adj_logits = self.predict_edges(gcn_out)
            
            adj = torch.sigmoid(adj_logits)
            
            adj_logits_list.append(adj_logits)
        
        return adj, adj_logits_list
    
    def predict_edges(self, node_features):
        """
        预测边（邻接矩阵）
        
        Args:
            node_features: [B, N, D]
        
        Returns:
            adj_logits: [B, N, N]
        """
        B, N, D = node_features.shape
        
        node_i = node_features.unsqueeze(2).expand(B, N, N, D)
        node_j = node_features.unsqueeze(1).expand(B, N, N, D)
        
        edge_features = torch.cat([node_i, node_j], dim=-1)
        
        adj_logits = self.edge_predictor(edge_features).squeeze(-1)
        
        return adj_logits
