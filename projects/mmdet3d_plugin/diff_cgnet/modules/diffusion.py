import torch
import torch.nn as nn
import numpy as np


class ColdDiffusion(nn.Module):
    """
    Cold Diffusion模块
    使用确定性退化而非随机噪声
    """
    
    def __init__(self, 
                 num_timesteps=1000,
                 beta_schedule='cosine',
                 s=0.008):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        if beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(num_timesteps, s)
        elif beta_schedule == 'linear':
            betas = self.linear_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - alphas_cumprod))
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        余弦调度（推荐）
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """
        线性调度
        """
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def q_sample(self, x_start, t, anchors=None, noise=None):
        """
        前向扩散过程（带锚点匹配）
        
        Args:
            x_start: torch.Tensor, shape (B, N, 4, 2), GT贝塞尔控制点
            t: torch.Tensor, shape (B,), 时间步
            anchors: torch.Tensor, shape (M, 4, 2), 锚点池
            noise: torch.Tensor, 可选的噪声
        
        Returns:
            x_t: torch.Tensor, shape (B, N, 4, 2), 加噪后的控制点
        """
        B, N = x_start.shape[:2]
        device = x_start.device
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(B, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(B, 1, 1, 1)
        
        if anchors is not None:
            if anchors.dim() == 3:
                M = anchors.shape[0]
                
                matched_anchors = []
                for b in range(B):
                    x_flat = x_start[b].flatten(1)
                    anchor_flat = anchors.flatten(1)
                    
                    dist_matrix = torch.cdist(x_flat, anchor_flat, p=2)
                    
                    matched_indices = dist_matrix.argmin(dim=1)
                    
                    matched_anchor = anchors[matched_indices]
                    matched_anchors.append(matched_anchor)
                
                matched_anchors = torch.stack(matched_anchors)
            else:
                matched_anchors = anchors
            
            x_t = sqrt_alphas_cumprod_t * x_start + \
                  sqrt_one_minus_alphas_cumprod_t * matched_anchors
        else:
            if noise is None:
                noise = torch.randn_like(x_start)
            
            x_t = sqrt_alphas_cumprod_t * x_start + \
                  sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t
    
    def ddim_sample_step(self, x_t, pred_x0, t, eta=0.0):
        """
        DDIM采样步骤（快速采样）
        
        Args:
            x_t: 当前状态
            pred_x0: 预测的x0
            t: 当前时间步
            eta: DDIM参数，0表示确定性采样
        
        Returns:
            x_{t-1}: 下一个状态
        """
        alpha_t = self.alphas_cumprod[t]
        
        if t > 0:
            alpha_t_prev = self.alphas_cumprod[t - 1]
        else:
            alpha_t_prev = torch.ones_like(alpha_t)
        
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)
        
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * \
                  torch.sqrt(1 - alpha_t / alpha_t_prev)
        
        pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * \
                   (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t)
        
        x_t_prev = torch.sqrt(alpha_t_prev) * pred_x0 + pred_dir
        
        if eta > 0 and t > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise
        
        return x_t_prev
