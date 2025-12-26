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
        前向扩散过程（简化版，prepare_gt已做匹配）
        
        Args:
            x_start: torch.Tensor, shape (B, N, 4, 2), 目标控制点（已匹配）
            t: torch.Tensor, shape (B,), 时间步
            anchors: torch.Tensor, shape (B, N, 4, 2) or (N, 4, 2), 锚点
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
                anchors = anchors.unsqueeze(0).expand(B, -1, -1, -1)
            
            x_t = sqrt_alphas_cumprod_t * x_start + \
                  sqrt_one_minus_alphas_cumprod_t * anchors
        else:
            if noise is None:
                noise = torch.randn_like(x_start)
            
            x_t = sqrt_alphas_cumprod_t * x_start + \
                  sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t
    
    def ddim_sample_step(self, x_t, pred_x0, t, t_prev=None, eta=0.0):
        """
        DDIM采样步骤（快速采样）
        
        Args:
            x_t: 当前状态
            pred_x0: 预测的x0
            t: 当前时间步
            t_prev: 下一个时间步（用于跳步采样），默认为 t-1
            eta: DDIM参数，0表示确定性采样
        
        Returns:
            x_{t_prev}: 下一个状态
        """
        # DDIM sampling involves divisions by (1 - alpha_t). Under fp16 training/inference,
        # alpha_t may round to 1.0 for small t, causing 0 denominators and NaNs/Infs.
        # Compute the core in fp32 with clamps for numerical stability.
        orig_dtype = x_t.dtype
        eps = 1e-8

        alpha_t = self.alphas_cumprod[t].float()
        
        if t_prev is None:
            t_prev = t - 1 if t > 0 else 0
        
        if t_prev > 0:
            alpha_t_prev = self.alphas_cumprod[t_prev].float()
        else:
            alpha_t_prev = torch.ones_like(alpha_t, dtype=torch.float32)
        
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)
        
        one_minus_alpha_t = torch.clamp(1.0 - alpha_t, min=eps)
        one_minus_alpha_t_prev = torch.clamp(1.0 - alpha_t_prev, min=eps)

        ratio = one_minus_alpha_t_prev / one_minus_alpha_t
        ratio = torch.clamp(ratio, min=0.0, max=1e4)
        term = 1.0 - (alpha_t / torch.clamp(alpha_t_prev, min=eps))
        term = torch.clamp(term, min=0.0, max=1.0)

        sigma_t = eta * torch.sqrt(ratio) * torch.sqrt(term)
        
        pred_dir_coeff_sq = 1.0 - alpha_t_prev - sigma_t ** 2
        pred_dir_coeff_sq = torch.clamp(pred_dir_coeff_sq, min=0.0)
        pred_dir_coeff = torch.sqrt(pred_dir_coeff_sq)

        x_t_fp32 = x_t.float()
        pred_x0_fp32 = pred_x0.float()
        pred_dir = pred_dir_coeff * (
            x_t_fp32 - torch.sqrt(torch.clamp(alpha_t, min=eps)) * pred_x0_fp32
        ) / torch.sqrt(one_minus_alpha_t)
        
        x_t_prev = torch.sqrt(torch.clamp(alpha_t_prev, min=eps)) * pred_x0_fp32 + pred_dir
        
        if eta > 0 and t > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise.float()
        
        x_t_prev = torch.nan_to_num(x_t_prev, nan=0.0, posinf=0.0, neginf=0.0)
        return x_t_prev.to(orig_dtype)
