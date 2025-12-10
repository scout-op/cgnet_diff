import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    线性注意力机制（高效版本）
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, N, D]
            key_value: [B, C, H, W]
        
        Returns:
            output: [B, N, D]
        """
        B, N, D = query.shape
        _, C, H, W = key_value.shape
        
        kv_flat = key_value.flatten(2).permute(0, 2, 1)
        
        Q = self.query_proj(query)
        K = self.key_proj(kv_flat)
        V = self.value_proj(kv_flat)
        
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        KV = torch.einsum('bnc,bnd->bcd', K, V)
        Z = torch.einsum('bnc,bc->bn', K, torch.ones(B, C, device=K.device))
        
        output = torch.einsum('bnc,bcd->bnd', Q, KV) / (torch.einsum('bnc,bn->bn', Q, Z).unsqueeze(-1) + 1e-6)
        
        output = self.out_proj(output)
        
        return output


class JunctionAwareQuery(nn.Module):
    """
    Junction Aware Query Enhancement模块
    增强模型对路口/分叉点的感知能力
    """
    
    def __init__(self, 
                 embed_dim=256,
                 dilate_radius=9,
                 use_linear_attn=True):
        """
        Args:
            embed_dim: 特征维度
            dilate_radius: 路口热图膨胀半径
            use_linear_attn: 是否使用线性注意力
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.dilate_radius = dilate_radius
        
        self.junction_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.junction_projector = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, 1)
        )
        
        if use_linear_attn:
            self.attention = LinearAttention(embed_dim)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim, num_heads=8, batch_first=True
            )
        
        self.use_linear_attn = use_linear_attn
    
    def forward(self, query_embed, bev_features, gt_junctions=None):
        """
        前向传播
        
        Args:
            query_embed: [B, N, D], 查询嵌入
            bev_features: [B, C, H, W], BEV特征
            gt_junctions: [B, H, W], GT路口热图（训练时）
        
        Returns:
            enhanced_query: [B, N, D], 增强后的查询
            junction_heatmap: [B, 1, H, W], 预测的路口热图
            junction_loss: scalar, 路口损失（训练时）
        """
        B, N, D = query_embed.shape
        
        junction_feat = self.junction_decoder(bev_features)
        
        junction_heatmap = self.junction_projector(junction_feat)
        
        if self.use_linear_attn:
            enhanced_query = self.attention(query_embed, junction_feat)
        else:
            junction_feat_flat = junction_feat.flatten(2).permute(0, 2, 1)
            enhanced_query, _ = self.attention(
                query_embed, junction_feat_flat, junction_feat_flat
            )
        
        junction_loss = None
        if gt_junctions is not None:
            gt_junctions_dilated = self.dilate_junctions(
                gt_junctions, self.dilate_radius
            )
            
            junction_loss = F.binary_cross_entropy_with_logits(
                junction_heatmap,
                gt_junctions_dilated.unsqueeze(1).float(),
                reduction='mean'
            )
        
        return enhanced_query, junction_heatmap, junction_loss
    
    def dilate_junctions(self, junctions, radius):
        """
        膨胀路口点（用于训练）
        
        Args:
            junctions: [B, H, W], 路口mask
            radius: int, 膨胀半径
        
        Returns:
            dilated: [B, H, W]
        """
        B, H, W = junctions.shape
        device = junctions.device
        
        kernel_size = 2 * radius + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        
        junctions_float = junctions.float().unsqueeze(1)
        
        dilated = F.conv2d(
            junctions_float,
            kernel,
            padding=radius
        )
        
        dilated = (dilated > 0).float().squeeze(1)
        
        return dilated
