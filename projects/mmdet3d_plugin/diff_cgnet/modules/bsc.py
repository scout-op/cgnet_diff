import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BezierSpaceConnection(nn.Module):
    """
    Bézier Space Connection模块
    在贝塞尔空间中对连接的线段施加连续性约束
    """
    
    def __init__(self, 
                 embed_dim=256,
                 num_ctrl_points=4,
                 num_combined_points=8):
        """
        Args:
            embed_dim: 特征维度
            num_ctrl_points: 单条线的控制点数
            num_combined_points: 连接后的控制点数
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_ctrl_points = num_ctrl_points
        self.num_combined_points = num_combined_points
        
        self.bezier_matrix = self.compute_bezier_projection_matrix(
            num_ctrl_points * 2, num_combined_points
        )
        
        self.bezier_decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_combined_points * 2)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def compute_bezier_projection_matrix(self, input_dim, output_dim):
        """
        计算贝塞尔投影矩阵（伪逆）
        
        将2n个点投影到m个控制点
        """
        from math import factorial
        
        def comb(n, k):
            return factorial(n) // (factorial(k) * factorial(n - k))
        
        n_points = input_dim // 2
        n_control = output_dim
        
        A = np.zeros((n_points, n_control))
        t = np.linspace(0, 1, n_points)
        
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = comb(n_control - 1, j) * \
                          np.power(1 - t[i], n_control - 1 - j) * \
                          np.power(t[i], j)
        
        A_pinv = np.linalg.pinv(A)
        
        matrix = torch.from_numpy(A_pinv).float()
        
        return nn.Parameter(matrix, requires_grad=False)
    
    def forward(self, lane_embeddings, lane_ctrl_points, connectivity):
        """
        前向传播
        
        Args:
            lane_embeddings: [B, N, D], 车道特征
            lane_ctrl_points: [B, N, 4, 2], 控制点
            connectivity: [B, N, N], 连接关系
        
        Returns:
            bsc_loss: scalar, 贝塞尔空间连续性损失
            enhanced_embeddings: [B, N, D], 增强后的特征
        """
        B, N, D = lane_embeddings.shape
        device = lane_embeddings.device
        
        connected_pairs = (connectivity > 0.5).nonzero(as_tuple=False)
        
        if len(connected_pairs) == 0:
            return torch.tensor(0.0, device=device), lane_embeddings
        
        total_loss = 0
        enhanced_emb = lane_embeddings.clone()
        
        for b, i, j in connected_pairs:
            emb_i = lane_embeddings[b, i]
            emb_j = lane_embeddings[b, j]
            
            concat_emb = torch.cat([emb_i, emb_j], dim=-1)
            
            fused_emb = self.feature_fusion(concat_emb)
            
            enhanced_emb[b, i] = enhanced_emb[b, i] + 0.1 * fused_emb
            enhanced_emb[b, j] = enhanced_emb[b, j] + 0.1 * fused_emb
            
            ctrl_i = lane_ctrl_points[b, i].flatten()
            ctrl_j = lane_ctrl_points[b, j].flatten()
            concat_ctrl = torch.cat([ctrl_i, ctrl_j], dim=-1)
            
            bezier_proj = torch.matmul(
                self.bezier_matrix.to(device),
                concat_ctrl.unsqueeze(-1)
            ).squeeze(-1)
            
            pred_combined_ctrl = self.bezier_decoder(concat_emb)
            
            continuity_loss = F.l1_loss(pred_combined_ctrl, bezier_proj)
            
            total_loss += continuity_loss
        
        avg_loss = total_loss / max(len(connected_pairs), 1)
        
        return avg_loss, enhanced_emb
