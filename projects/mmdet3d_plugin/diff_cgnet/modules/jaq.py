import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    CGNet 风格的线性注意力（用于 KPALayer）
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, query, key, value, q_mask=None, kv_mask=None):
        """
        Args:
            query: [B, L, H, D]
            key: [B, S, H, D]
            value: [B, S, H, D]
        
        Returns:
            output: [B, L, H, D]
        """
        # Numerical stability note:
        # The sequence length S (= H*W/4) can be large (e.g., 5k). Under AMP/fp16,
        # the einsum accumulation in KV = K^T @ V may overflow and yield inf/-inf,
        # and then inf + (-inf) can become NaN. We therefore compute the attention
        # core in fp32 and cast back at the end.
        orig_dtype = query.dtype
        query = query.float()
        key = key.float()
        value = value.float()

        # ELU + 1 确保非负
        Q = F.elu(query) + 1
        K = F.elu(key) + 1
        
        # FP16 数值稳定性：更严格的 clamp
        Q = torch.clamp(Q, min=self.eps, max=100.0)  # FP16 safe range
        K = torch.clamp(K, min=self.eps, max=100.0)
        value = torch.clamp(value, min=-100.0, max=100.0)
        Q = torch.nan_to_num(Q, nan=self.eps, posinf=100.0, neginf=self.eps)
        K = torch.nan_to_num(K, nan=self.eps, posinf=100.0, neginf=self.eps)
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # K^T @ V
        # Scale by S to keep magnitude bounded without changing the ratio
        # (both numerator and denominator are scaled).
        seq_len = max(int(key.size(1)), 1)
        KV = torch.einsum('bshd,bshc->bhdc', K, value) / float(seq_len)  # [B, H, D, D]
        KV = torch.clamp(KV, min=-1000.0, max=1000.0)  # Prevent extreme values
        KV = torch.nan_to_num(KV, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # K sum
        Z = K.sum(dim=1) / float(seq_len)  # [B, H, D]
        Z = torch.clamp(Z, min=self.eps)  # Prevent division by zero
        Z = torch.nan_to_num(Z, nan=self.eps, posinf=1000.0, neginf=self.eps)
        
        # Q @ (K^T @ V) / (Q @ K^T @ 1)
        output = torch.einsum('blhd,bhdc->blhc', Q, KV)  # [B, L, H, D]
        normalizer = torch.einsum('blhd,bhd->blh', Q, Z).unsqueeze(-1) + self.eps
        normalizer = torch.clamp(normalizer, min=self.eps)  # Extra safety
        
        output = output / normalizer
        output = torch.nan_to_num(output, nan=0.0, posinf=100.0, neginf=-100.0)
        output = torch.clamp(output, min=-100.0, max=100.0)  # Final clamp
        
        return output.to(orig_dtype)


class KPALayer(nn.Module):
    """
    CGNet 风格的 Key Point Attention Layer
    用于增强 query 的路口感知能力
    """
    def __init__(self, d_model, nhead=4):
        super().__init__()
        
        self.dim = d_model // nhead
        self.nhead = nhead
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )
        
        # Norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x: [B, L, C] - query
            source: [B, S, C] - junction features
        
        Returns:
            enhanced: [B, L, C]
        """
        # Run KPALayer in fp32 to avoid AMP/fp16 overflow (NaNs observed in training).
        orig_dtype = x.dtype
        x_fp32 = x.float()
        source_fp32 = source.float()

        bs = x_fp32.size(0)
        query, key, value = x_fp32, source_fp32, source_fp32
        
        # Multi-head attention
        # NOTE: Under mmcv fp16 training, Linear/LayerNorm weights may be fp16.
        # Explicitly casting the projected tensors to fp32 keeps the attention core stable.
        query = self.q_proj(query).reshape(bs, -1, self.nhead, self.dim).float()  # [B, L, H, D]
        key = self.k_proj(key).reshape(bs, -1, self.nhead, self.dim).float()      # [B, S, H, D]
        value = self.v_proj(value).reshape(bs, -1, self.nhead, self.dim).float()  # [B, S, H, D]
        
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        message = self.merge(message.reshape(bs, -1, self.nhead * self.dim)).float()
        # Guard LayerNorm from inf/-inf causing inf-inf -> NaN.
        message = torch.nan_to_num(message, nan=0.0, posinf=1e4, neginf=-1e4)
        message = torch.clamp(message, min=-1e4, max=1e4)
        message = self.norm1(message)
        message = torch.nan_to_num(message, nan=0.0, posinf=50.0, neginf=-50.0)
        message = torch.clamp(message, min=-50.0, max=50.0)
        
        # FFN with concat
        message = self.mlp(torch.cat([x_fp32, message], dim=2)).float()
        message = torch.nan_to_num(message, nan=0.0, posinf=1e4, neginf=-1e4)
        message = torch.clamp(message, min=-1e4, max=1e4)
        message = self.norm2(message)
        message = torch.nan_to_num(message, nan=0.0, posinf=50.0, neginf=-50.0)
        message = torch.clamp(message, min=-50.0, max=50.0)
        
        enhanced = x_fp32 + message
        enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=50.0, neginf=-50.0)
        enhanced = torch.clamp(enhanced, min=-50.0, max=50.0)
        return enhanced.to(orig_dtype)


class JunctionAwareQuery(nn.Module):
    """
    Junction Aware Query Enhancement模块
    增强模型对路口/分叉点的感知能力
    """
    
    def __init__(self, 
                 embed_dim=256,
                 dilate_radius=9,
                 nhead=4):
        """
        Args:
            embed_dim: 特征维度
            dilate_radius: 路口热图膨胀半径
            nhead: KPALayer 的注意力头数
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.dilate_radius = dilate_radius
        self.nhead = nhead
        
        # CGNet 风格：2层 Conv2d + BN + ReLU（bias=False）
        self.junction_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # CGNet 风格：1x1 Conv 投影到热图
        self.junction_projector = nn.Sequential(
            nn.Conv2d(embed_dim, 1, 1),
            # 注意：sigmoid 在 forward 中用于 loss 计算
        )
        
        # CGNet 风格：maxpool 下采样
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # CGNet 风格：KPALayer 用于 query 增强
        self.query_enhance = KPALayer(d_model=embed_dim, nhead=nhead)
        
        # 权重初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
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
        # Allow inference usage: gt_junctions can be None (no loss), but query
        # enhancement still runs based on predicted junction features.
        
        # NaN 保护：检查输入
        if torch.isnan(bev_features).any():
            print("[JAQ] bev_features contains NaN!")
            # 返回原始查询，跳过 JAQ
            return query_embed, torch.zeros(B, 1, bev_features.shape[2], bev_features.shape[3], device=bev_features.device), None
        
        junction_feat = self.junction_decoder(bev_features)
        
        # NaN 保护
        if torch.isnan(junction_feat).any():
            print("[JAQ] junction_feat contains NaN after decoder!")
            return query_embed, torch.zeros(B, 1, bev_features.shape[2], bev_features.shape[3], device=bev_features.device), None
        
        # 生成热图（用于 loss）
        junction_heatmap = self.junction_projector(junction_feat)
        
        # CGNet 风格：maxpool 下采样 + flatten
        junction_feat_pooled = self.maxpool(junction_feat)  # [B, D, H/2, W/2]
        junction_feat_flat = junction_feat_pooled.flatten(2).permute(0, 2, 1)  # [B, H*W/4, D]
        
        # NaN check before KPALayer
        if torch.isnan(junction_feat_flat).any():
            print("[JAQ] junction_feat_flat contains NaN before KPALayer!")
            return query_embed, junction_heatmap, None
        
        if torch.isnan(query_embed).any():
            print("[JAQ] query_embed contains NaN before KPALayer!")
            return query_embed, junction_heatmap, None
        
        # CGNet 风格：KPALayer 增强 query
        enhanced_query = self.query_enhance(query_embed, junction_feat_flat)
        
        # NaN/Inf 保护
        if not torch.isfinite(enhanced_query).all():
            print("[JAQ] enhanced_query contains NaN/Inf after KPALayer!")
            print(f"  query_embed range: [{query_embed.min():.4f}, {query_embed.max():.4f}]")
            print(f"  junction_feat_flat range: [{junction_feat_flat.min():.4f}, {junction_feat_flat.max():.4f}]")
            enhanced_query = query_embed
        
        junction_loss = None
        if gt_junctions is not None:
            # Resize GT to match prediction size
            pred_sigmoid = junction_heatmap.sigmoid()  # [B, 1, H_pred, W_pred]
            
            # Resize gt_junctions to match prediction
            gt_target = gt_junctions.unsqueeze(1).float()  # [B, 1, H_gt, W_gt]
            if gt_target.shape != pred_sigmoid.shape:
                gt_target = F.interpolate(
                    gt_target, 
                    size=(pred_sigmoid.shape[2], pred_sigmoid.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            # CGNet style neg_loss
            junction_loss = self._neg_loss(pred_sigmoid, gt_target)
        
        return enhanced_query, junction_heatmap, junction_loss
    
    def _neg_loss(self, pred, gt, weights=None):
        """Modified focal loss (CornerNet style)."""
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        
        neg_weights = torch.pow(1 - gt, 4)
        
        loss = 0
        eps = 1e-6
        
        pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds
        
        num_pos = pos_inds.float().sum()
        neg_loss = neg_loss.sum()
        
        if weights is not None:
            pos_loss = (pos_loss * weights).sum()
        else:
            pos_loss = pos_loss.sum()
        
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        
        return loss
    
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
