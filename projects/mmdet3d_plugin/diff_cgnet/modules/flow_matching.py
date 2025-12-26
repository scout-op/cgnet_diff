"""
Flow Matching for Centerline Generation
替代原有的Cold Diffusion，更简单高效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码 (用于时间嵌入)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (借鉴DIVER)
    用时间条件调制特征
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.modulation = nn.Sequential(
            nn.Mish(),
            nn.Linear(d_model, d_model * 2),
        )
    
    def forward(self, x, t_emb):
        """
        x: [B, ..., D]
        t_emb: [B, D]
        """
        x = self.norm(x)
        
        # 生成scale和shift
        scale_shift = self.modulation(t_emb)
        
        # 扩展维度匹配x
        while scale_shift.dim() < x.dim():
            scale_shift = scale_shift.unsqueeze(1)
        
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # 限制scale和shift范围，防止梯度爆炸
        scale = torch.tanh(scale)  # 限制在 [-1, 1]
        shift = torch.clamp(shift, -10.0, 10.0)
        
        return x * (1 + scale) + shift


class FlowMatchingModule(nn.Module):
    """
    Flow Matching核心模块
    
    训练: 学习从噪声到目标的速度场
    推理: 单步或少步采样
    支持: log_prob计算 (用于GRPO)
    """
    def __init__(self, d_model=256, sigma_min=1e-4, sigma=0.5):
        super().__init__()
        self.d_model = d_model
        self.sigma_min = sigma_min
        self.sigma = sigma  # 噪声缩放因子，减小 noise 范围
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def get_time_embedding(self, t):
        """获取时间嵌入"""
        return self.time_mlp(t)
    
    def interpolate(self, x0, x1, t):
        """
        线性插值 (OT-CFM)
        x_t = (1-t) * x0 + t * x1
        
        Args:
            x0: 噪声 [B, ...]
            x1: 目标 [B, ...]
            t: 时间 [B]
        """
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        x_t = (1 - t) * x0 + t * x1
        return x_t
    
    def get_velocity(self, x0, x1):
        """
        计算真实速度场 (OT-CFM)
        v = x1 - x0
        """
        return x1 - x0
    
    def sample_time(self, batch_size, device):
        """采样时间 t ~ U[0, 1]"""
        return torch.rand(batch_size, device=device)
    
    def sample_noise(self, shape, device):
        """采样噪声 x0 ~ N(0, sigma^2 * I)"""
        return self.sigma * torch.randn(shape, device=device)
    
    @torch.no_grad()
    def euler_sample(self, velocity_fn, shape, device, num_steps=1, return_trajectory=False, x0_prior=None):
        """
        Euler采样
        
        Args:
            velocity_fn: 速度预测函数 v = velocity_fn(x_t, t)
            shape: 输出形状
            device: 设备
            num_steps: 采样步数 (1=单步)
            return_trajectory: 是否返回完整轨迹
            x0_prior: 可选的起点先验 [B, ...], 如果提供则在此基础上加噪声
        
        Returns:
            x_1: 生成的样本
        """
        # 从噪声开始 (可选：使用位置先验)
        noise = self.sigma * torch.randn(shape, device=device)
        if x0_prior is not None:
            # 以先验位置为中心加噪声
            x = x0_prior + noise
        else:
            x = noise
        
        if return_trajectory:
            trajectory = [x.clone()]
        
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((shape[0],), step * dt, device=device)
            v = velocity_fn(x, t)
            x = x + dt * v
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x
    
    def compute_log_prob(self, x1, velocity_fn, num_steps=10):
        """
        [DEPRECATED] 计算Flow Matching的log概率
        注意: 简化版GRPO不使用此函数，保留仅供参考
        
        通过ODE积分计算: log p(x1) = log p(x0) - ∫ div(v_t) dt
        
        Args:
            x1: [B, ...] 生成的样本
            velocity_fn: 速度函数 v = velocity_fn(x, t)
            num_steps: ODE积分步数
        
        Returns:
            log_prob: [B] log概率
        """
        B = x1.shape[0]
        device = x1.device
        
        # 初始log概率 (标准高斯)
        log_p0 = -0.5 * (x1 ** 2).sum(dim=tuple(range(1, x1.dim()))) - \
                 0.5 * np.prod(x1.shape[1:]) * np.log(2 * np.pi)
        
        # 反向ODE积分计算散度
        x = x1.clone()
        dt = 1.0 / num_steps
        
        log_det_sum = torch.zeros(B, device=device)
        
        for step in reversed(range(num_steps)):
            t = torch.full((B,), step * dt, device=device)
            
            # 计算速度和散度
            x.requires_grad_(True)
            v = velocity_fn(x, t)
            
            # 计算散度 div(v) = trace(∂v/∂x)
            # 使用Hutchinson's trace estimator
            eps = torch.randn_like(x)
            v_eps = (v * eps).sum()
            grad_v_eps = torch.autograd.grad(v_eps, x, create_graph=False)[0]
            div_v = (grad_v_eps * eps).sum(dim=tuple(range(1, x.dim())))
            
            log_det_sum = log_det_sum - dt * div_v
            
            # 反向Euler步
            with torch.no_grad():
                x = x - dt * v
                x.requires_grad_(False)
        
        log_prob = log_p0 + log_det_sum
        
        return log_prob


class FlowVelocityNet(nn.Module):
    """
    速度预测网络
    输入: 噪声坐标 + 时间 + 条件(Query特征, BEV特征)
    输出: 速度场
    """
    def __init__(self, 
                 d_model=256, 
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # 坐标编码
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # AdaLN调制层
        self.adaln_layers = nn.ModuleList([
            AdaLN(d_model) for _ in range(num_layers)
        ])
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        
        # BEV交叉注意力
        self.bev_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.bev_norm = nn.LayerNorm(d_model)
        
        # 输出头
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        
        # 初始化权重 (小权重初始化，防止初始输出过大)
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使用较小的初始化值防止梯度爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 输出层使用更小的初始化
        for m in self.velocity_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, noisy_coords, t, query_feat, bev_feat):
        """
        Args:
            noisy_coords: [B, N, P, 2] 噪声坐标 (N=num_queries, P=num_points)
            t: [B] 时间
            query_feat: [B, N, D] 或 [B, N, P, D] Query特征
                        - [B, N, D]: 每条线共享特征 (all_pts模式)
                        - [B, N, P, D]: 每个控制点独立特征 (instance_pts模式)
            bev_feat: [B, C, H, W] 或 [B, H*W, C] BEV特征
        
        Returns:
            velocity: [B, N, P, 2] 速度场
        """
        B, N, P, _ = noisy_coords.shape

        # Debug: print shapes to verify BEV layout (set env CGNET_DEBUG_FLOW_SHAPES=1)
        debug_enabled = os.environ.get('CGNET_DEBUG_FLOW_SHAPES', '').lower() in ('1', 'true', 'yes', 'y')
        is_main_process = os.environ.get('RANK', '0') in ('0', '-1')
        if debug_enabled and is_main_process:
            if not getattr(self, '_debug_shapes_printed', False):
                self._debug_shapes_printed = True
                bev_shape = tuple(bev_feat.shape) if hasattr(bev_feat, 'shape') else str(type(bev_feat))
                query_shape = tuple(query_feat.shape) if hasattr(query_feat, 'shape') else str(type(query_feat))

                bev_feat_is_hwbc = (
                    hasattr(bev_feat, 'dim')
                    and bev_feat.dim() == 3
                    and bev_feat.shape[0] != B
                    and bev_feat.shape[1] == B
                )
                if hasattr(bev_feat, 'dim') and bev_feat.dim() == 4:
                    bev_flat_shape = tuple(bev_feat.flatten(2).permute(0, 2, 1).shape)
                elif bev_feat_is_hwbc:
                    bev_flat_shape = tuple(bev_feat.permute(1, 0, 2).shape)
                else:
                    bev_flat_shape = bev_shape

                print(
                    f'[FlowVelocityNet] noisy_coords={tuple(noisy_coords.shape)} '
                    f't={tuple(t.shape)} query_feat={query_shape} bev_feat={bev_shape} bev_flat={bev_flat_shape}',
                    flush=True,
                )
                if bev_feat_is_hwbc:
                    print(
                        '[FlowVelocityNet] NOTE: converting bev_feat from [H*W, B, C] -> [B, H*W, C].',
                        flush=True,
                    )

        # 1. 时间嵌入
        t_emb = self.time_mlp(t)  # [B, D]
        
        # 2. 坐标编码
        coord_feat = self.coord_encoder(noisy_coords)  # [B, N, P, D]
        
        # 3. 融合Query特征 (改进: 使用乘法门控而非简单相加)
        # 支持两种输入格式
        if query_feat.dim() == 3:
            # [B, N, D] -> expand to [B, N, P, D]
            query_expand = query_feat.unsqueeze(2).expand(-1, -1, P, -1)
        else:
            # [B, N, P, D] -> 直接使用 (每个控制点有独立特征)
            query_expand = query_feat
        
        # 门控融合: coord_feat 调制 query_feat
        gate = torch.sigmoid(coord_feat)
        feat = gate * query_expand + (1 - gate) * coord_feat  # [B, N, P, D]
        
        # 4. 展平为序列
        feat_flat = feat.view(B, N * P, -1)  # [B, N*P, D]
        
        # 5. BEV交叉注意力
        if bev_feat.dim() == 4:
            bev_flat = bev_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        else:
            # Support both [B, H*W, C] and [H*W, B, C] layouts.
            if bev_feat.dim() == 3 and bev_feat.shape[0] != B and bev_feat.shape[1] == B:
                bev_flat = bev_feat.permute(1, 0, 2).contiguous()
            else:
                bev_flat = bev_feat
        
        # 检查并处理 NaN/Inf
        if torch.isnan(bev_flat).any() or torch.isinf(bev_flat).any():
            bev_flat = torch.nan_to_num(bev_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        
        bev_out, _ = self.bev_cross_attn(feat_flat, bev_flat, bev_flat)
        bev_out = torch.nan_to_num(bev_out, nan=0.0, posinf=1e4, neginf=-1e4)
        feat_flat = self.bev_norm(feat_flat + bev_out)
        
        # 6. Transformer + AdaLN
        for adaln, transformer in zip(self.adaln_layers, self.transformer_layers):
            feat_flat = adaln(feat_flat, t_emb)
            feat_flat = transformer(feat_flat)
            # 每层后检查数值稳定性
            feat_flat = torch.nan_to_num(feat_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 7. 输出速度
        velocity = self.velocity_head(feat_flat)  # [B, N*P, 2]
        velocity = velocity.view(B, N, P, 2)
        
        # 限制速度范围，防止梯度爆炸 (v_true 约在 [-3, 3] 范围)
        velocity = torch.clamp(velocity, -10.0, 10.0)
        
        return velocity


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching损失函数
    默认使用速度场 MSE；可选加入 x1 重建监督以提升生成质量（尤其是少步采样）。
    """
    def __init__(self,
                 recon_loss_weight=0.0,
                 recon_loss_type='l1',
                 time_sampling='uniform',
                 time_beta_a=1.0,
                 time_beta_b=1.0,
                 time_clip_eps=1e-4,
                 time_weighting='none',
                 time_weight_power=2.0,
                 time_weight_lambda=2.0,
                 time_weight_normalize=True,
                 coord_scale=(1.0, 1.0)):
        super().__init__()
        self.flow_module = FlowMatchingModule()
        self.recon_loss_weight = float(recon_loss_weight)
        self.recon_loss_type = recon_loss_type
        if self.recon_loss_type not in ('l1', 'mse'):
            raise ValueError(f'Unsupported recon_loss_type={self.recon_loss_type!r}, expected "l1" or "mse".')
        self.time_sampling = time_sampling
        self.time_beta_a = float(time_beta_a)
        self.time_beta_b = float(time_beta_b)
        self.time_clip_eps = float(time_clip_eps)
        if not (0.0 <= self.time_clip_eps < 0.5):
            raise ValueError(f'time_clip_eps must be in [0, 0.5), got {self.time_clip_eps}.')
        self.time_weighting = time_weighting
        self.time_weight_power = float(time_weight_power)
        self.time_weight_lambda = float(time_weight_lambda)
        self.time_weight_normalize = bool(time_weight_normalize)
        if self.time_sampling not in ('uniform', 'beta'):
            raise ValueError(f'Unsupported time_sampling={self.time_sampling!r}, expected "uniform" or "beta".')
        if self.time_sampling == 'beta' and (self.time_beta_a <= 0 or self.time_beta_b <= 0):
            raise ValueError(
                f'Beta params must be > 0, got time_beta_a={self.time_beta_a}, time_beta_b={self.time_beta_b}.'
            )
        if self.time_weighting not in ('none', 'linear', 'power', 'exp'):
            raise ValueError(
                f'Unsupported time_weighting={self.time_weighting!r}, expected "none|linear|power|exp".'
            )

        if coord_scale is None:
            coord_scale = (1.0, 1.0)
        if isinstance(coord_scale, torch.Tensor):
            coord_scale_tensor = coord_scale.detach().float().flatten()
        else:
            coord_scale_tensor = torch.tensor(coord_scale, dtype=torch.float32).flatten()
        if coord_scale_tensor.numel() != 2:
            raise ValueError(f'coord_scale must have 2 elements for x/y, got shape {tuple(coord_scale_tensor.shape)}.')
        self.register_buffer('coord_scale', coord_scale_tensor)

    def _sample_time(self, batch_size, device):
        if self.time_sampling == 'uniform':
            t = torch.rand(batch_size, device=device)
        else:
            dist = torch.distributions.Beta(self.time_beta_a, self.time_beta_b)
            t = dist.sample((batch_size,)).to(device)
        if self.time_clip_eps > 0:
            t = t.clamp(min=self.time_clip_eps, max=1.0 - self.time_clip_eps)
        return t

    def _time_weights(self, t):
        if self.time_weighting == 'none':
            w = torch.ones_like(t)
        elif self.time_weighting == 'linear':
            w = 1.0 - t
        elif self.time_weighting == 'power':
            w = (1.0 - t).pow(self.time_weight_power)
        else:
            w = torch.exp(-self.time_weight_lambda * t)
        if self.time_weight_normalize:
            w = w / (w.mean() + 1e-6)
        return w
    
    def forward(self, velocity_net, gt_coords, query_feat, bev_feat, pos_mask=None, x0_prior=None):
        """
        Args:
            velocity_net: 速度预测网络
            gt_coords: [B, N, P, 2] GT坐标
            query_feat: [B, N, D] 或 [B, N, P, D] Query特征
            bev_feat: BEV特征
            pos_mask: [B, N] 有效mask (可选)
            x0_prior: [B, N, P, 2] 位置先验 (可选，与推理保持一致)
        
        Returns:
            loss: 标量损失
        """
        B = gt_coords.shape[0]
        device = gt_coords.device
        
        # 1. 采样时间 (可选偏向 t≈0，匹配少步推理)
        t = self._sample_time(B, device)
        w_t = self._time_weights(t)
        
        # 2. 采样噪声 + 位置先验 (与推理保持一致)
        noise = self.flow_module.sample_noise(gt_coords.shape, device)
        if x0_prior is not None:
            x0 = x0_prior + noise  # 先验 + 噪声
        else:
            x0 = noise
        
        # 3. 插值得到x_t
        x_t = self.flow_module.interpolate(x0, gt_coords, t)
        
        # 4. 真实速度
        v_true = self.flow_module.get_velocity(x0, gt_coords)
        
        # 5. 预测速度
        v_pred = velocity_net(x_t, t, query_feat, bev_feat)
        
        # 6. 计算速度场损失（按样本加权，强调小 t）
        v_err = (v_pred - v_true) ** 2  # [B, N, P, 2]
        scale = self.coord_scale.view(1, 1, 1, 2).to(v_err.dtype)
        v_err = v_err * (scale ** 2)
        if pos_mask is not None:
            mask_v = pos_mask.unsqueeze(-1).unsqueeze(-1).expand_as(v_err)
            v_err = v_err * mask_v
            denom_v = mask_v.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            v_loss_per = v_err.sum(dim=(1, 2, 3)) / denom_v
        else:
            v_loss_per = v_err.mean(dim=(1, 2, 3))
        loss_per = v_loss_per

        # 7. 可选：重建 x1（强化 t≈0 的监督，对少步采样更友好）
        if self.recon_loss_weight > 0:
            t_view = t.view(-1, *([1] * (gt_coords.dim() - 1)))
            x1_hat = x_t + (1 - t_view) * v_pred

            if self.recon_loss_type == 'l1':
                x_err = (x1_hat - gt_coords).abs()
                x_err = x_err * scale
            else:
                x_err = (x1_hat - gt_coords) ** 2
                x_err = x_err * (scale ** 2)

            if pos_mask is not None:
                mask_x = pos_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x_err)
                x_err = x_err * mask_x
                denom_x = mask_x.sum(dim=(1, 2, 3)).clamp_min(1e-6)
                x_loss_per = x_err.sum(dim=(1, 2, 3)) / denom_x
            else:
                x_loss_per = x_err.mean(dim=(1, 2, 3))

            loss_per = loss_per + self.recon_loss_weight * x_loss_per

        # time-weighted mean
        loss = (loss_per * w_t).sum() / (w_t.sum() + 1e-6)
        
        return loss
