import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from .utils import bezier_interpolate, normalize_coords


class BezierDeformableAttention(nn.Module):
    """
    基于贝塞尔插值的Deformable Attention
    解决控制点稀疏导致的特征采样不充分问题
    """
    
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 num_sample_points=10):
        """
        Args:
            embed_dim: 特征维度
            num_heads: 注意力头数
            num_levels: 特征层级数
            num_points: 每个头的采样点数
            num_sample_points: 贝塞尔插值的密集点数
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_sample_points = num_sample_points
        
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, 
                query_embed,
                ctrl_points, 
                bev_features,
                spatial_shapes,
                pc_range):
        """
        前向传播
        
        Args:
            query_embed: torch.Tensor, shape (B, N, D), 查询嵌入
            ctrl_points: torch.Tensor, shape (B, N, 4, 2), 控制点（可能有噪声）
            bev_features: torch.Tensor, shape (B, C, H, W), BEV特征
            spatial_shapes: torch.Tensor, BEV特征的空间尺寸
            pc_range: list, 坐标范围
        
        Returns:
            output: torch.Tensor, shape (B, N, D), 输出特征
        """
        B, N = ctrl_points.shape[:2]
        device = ctrl_points.device
        
        reference_points = self.generate_reference_points(
            ctrl_points, pc_range
        )
        
        bev_features_flat = bev_features.flatten(2).permute(0, 2, 1)
        
        query = self.query_proj(query_embed)
        
        output = self.deform_attn(
            query=query.flatten(0, 1).unsqueeze(0),
            key=None,
            value=bev_features_flat,
            reference_points=reference_points.flatten(0, 1).unsqueeze(0),
            spatial_shapes=spatial_shapes,
            level_start_index=torch.tensor([0], device=device)
        )
        
        output = output.squeeze(0).view(B, N, -1)
        output = self.output_proj(output)
        
        return output
    
    def generate_reference_points(self, ctrl_points, pc_range):
        """
        从贝塞尔控制点生成密集的参考点
        
        Args:
            ctrl_points: torch.Tensor, shape (B, N, 4, 2)
            pc_range: list
        
        Returns:
            reference_points: torch.Tensor, shape (B, N, K, 2), 归一化到[0,1]
        """
        dense_points = bezier_interpolate(
            ctrl_points, 
            num_points=self.num_sample_points
        )
        
        reference_points = normalize_coords(dense_points, pc_range)
        
        reference_points = torch.clamp(reference_points, 0.01, 0.99)
        
        if torch.isnan(reference_points).any() or torch.isinf(reference_points).any():
            print("Warning: Invalid reference points detected, using fallback")
            reference_points = torch.clamp(reference_points, 0.01, 0.99)
            reference_points = torch.nan_to_num(reference_points, nan=0.5, posinf=0.99, neginf=0.01)
        
        return reference_points
