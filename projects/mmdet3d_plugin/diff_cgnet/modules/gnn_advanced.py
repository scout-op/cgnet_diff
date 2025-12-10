import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.runner.base_module import BaseModule, Sequential


class LclcGNNLayer(nn.Module):
    """
    CGNet原版的图卷积层
    支持有向图，区分前向和后向边
    """
    
    def __init__(self, in_features, out_features, edge_weight=0.5):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            edge_weight: 边的权重（0-1之间）
        """
        super(LclcGNNLayer, self).__init__()
        self.edge_weight = edge_weight
        
        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))
        
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
    
    def forward(self, input, adj):
        """
        前向传播
        
        Args:
            input: [B, N, D], 节点特征
            adj: [B, N, N], 邻接矩阵（有向）
        
        Returns:
            output: [B, N, D], 聚合后的特征
        """
        support_loop = torch.matmul(input, self.weight)
        output = support_loop
        
        if self.edge_weight != 0:
            support_forward = torch.matmul(input, self.weight_forward)
            output_forward = torch.matmul(adj, support_forward)
            output += self.edge_weight * output_forward
            
            support_backward = torch.matmul(input, self.weight_backward)
            output_backward = torch.matmul(adj.permute(0, 2, 1), support_backward)
            output += self.edge_weight * output_backward
        
        return output


class CustomGRU(nn.Module):
    """
    CGNet原版的GRU实现
    """
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: [B, N, D]
            hidden: [B, N, D]
        
        Returns:
            new_h: [B, N, D]
        """
        if hidden is None:
            hidden = torch.zeros_like(x)
        
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, -1)
        h_r, h_i, h_n = gate_h.chunk(3, -1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy


class AdvancedTopologyGNN(BaseModule):
    """
    CGNet原版的GNN实现（完整版）
    结合MLP、LclcGNN和GRU进行拓扑预测
    """
    
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_fcs=2,
                 ffn_drop=0.1,
                 init_cfg=None,
                 edge_weight=0.8,
                 num_layers=6,
                 **kwargs):
        """
        Args:
            embed_dims: 嵌入维度
            feedforward_channels: 前馈网络维度
            num_fcs: MLP层数
            ffn_drop: Dropout概率
            edge_weight: 边的权重
            num_layers: 迭代层数
        """
        super(AdvancedTopologyGNN, self).__init__(init_cfg)
        
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.num_layers = num_layers
        self.activate = nn.ReLU(inplace=True)
        
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop)
                )
            )
            in_channels = feedforward_channels
        
        layers.append(
            Sequential(
                Linear(feedforward_channels, embed_dims),
                self.activate,
                nn.Dropout(ffn_drop)
            )
        )
        
        self.layers = Sequential(*layers)
        self.edge_weight = edge_weight
        
        self.lclc_gnn_layer = LclcGNNLayer(
            embed_dims, embed_dims, edge_weight=edge_weight
        )
        
        self.gru = CustomGRU(embed_dims, embed_dims)
        
        self.downsample = nn.Linear(embed_dims, embed_dims)
        
        self.gnn_dropout1 = nn.Dropout(ffn_drop)
        self.gnn_dropout2 = nn.Dropout(ffn_drop)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_drop),
            nn.Linear(embed_dims, 1)
        )
    
    def forward(self, lc_query, init_adj=None):
        """
        前向传播（CGNet原版逻辑）
        
        Args:
            lc_query: [B, N, D], 车道查询特征
            init_adj: [B, N, N], 初始邻接矩阵
        
        Returns:
            adj_matrix: [B, N, N], 预测的邻接矩阵
            adj_logits_list: List, 每层的logits
        """
        B, N, D = lc_query.shape
        device = lc_query.device
        
        if init_adj is None:
            lclc_adj = torch.zeros(B, N, N, device=device)
        else:
            lclc_adj = init_adj
        
        hidden = None
        adj_logits_list = []
        
        for layer in range(self.num_layers):
            out = self.layers(lc_query)
            out = out.permute(1, 0, 2)
            
            out = self.lclc_gnn_layer(out, lclc_adj)
            
            out = self.activate(out)
            out = self.gnn_dropout1(out)
            out = self.downsample(out)
            out = self.gnn_dropout2(out)
            
            out = out.permute(1, 0, 2)
            
            lc_query = lc_query + out
            
            hidden = self.gru(lc_query, hidden)
            
            adj_logits = self.predict_edges(hidden)
            lclc_adj = torch.sigmoid(adj_logits)
            
            adj_logits_list.append(adj_logits)
        
        return lclc_adj, adj_logits_list
    
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
