"""
GRPO (Group Relative Policy Optimization) for Centerline Generation
借鉴DIVER的强化学习优化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CenterlineReward(nn.Module):
    """
    中心线生成的Reward计算
    多维度评估生成质量
    """
    def __init__(self,
                 chamfer_weight=0.4,
                 topology_weight=0.3,
                 smoothness_weight=0.2,
                 diversity_weight=0.1,
                 pc_range=None):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.topology_weight = topology_weight
        self.smoothness_weight = smoothness_weight
        self.diversity_weight = diversity_weight
        self.pc_range = pc_range or [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]
    
    def chamfer_reward(self, pred_coords, gt_coords, pred_mask, gt_mask):
        """
        Chamfer距离作为几何精度reward (优化版)
        
        Args:
            pred_coords: [B, N, P, 2] 预测坐标
            gt_coords: [B, N, P, 2] GT坐标  
            pred_mask: [B, N] 预测有效mask
            gt_mask: [B, N] GT有效mask
        
        Returns:
            reward: [B] 每个样本的reward (越高越好)
        """
        B = pred_coords.shape[0]
        device = pred_coords.device
        rewards = []
        
        for b in range(B):
            pred_valid = pred_coords[b][pred_mask[b]]  # [M1, P, 2]
            gt_valid = gt_coords[b][gt_mask[b]]        # [M2, P, 2]
            
            if pred_valid.shape[0] == 0 or gt_valid.shape[0] == 0:
                rewards.append(torch.tensor(0.0, device=device))
                continue
            
            # 展平点
            pred_pts = pred_valid.flatten(0, 1)  # [M1*P, 2]
            gt_pts = gt_valid.flatten(0, 1)      # [M2*P, 2]
            
            # 优化: 如果点数过多，采样减少计算量
            max_pts = 200
            if pred_pts.shape[0] > max_pts:
                indices = torch.randperm(pred_pts.shape[0], device=device)[:max_pts]
                pred_pts = pred_pts[indices]
            if gt_pts.shape[0] > max_pts:
                indices = torch.randperm(gt_pts.shape[0], device=device)[:max_pts]
                gt_pts = gt_pts[indices]
            
            # Chamfer距离
            dist_pred_to_gt = torch.cdist(pred_pts, gt_pts).min(dim=1)[0].mean()
            dist_gt_to_pred = torch.cdist(gt_pts, pred_pts).min(dim=1)[0].mean()
            chamfer = (dist_pred_to_gt + dist_gt_to_pred) / 2
            
            # 转换为reward (距离越小reward越高)
            reward = torch.exp(-chamfer.clamp(max=10.0))  # 裁剪防止数值问题
            rewards.append(reward)
        
        return torch.stack(rewards)
    
    def topology_reward(self, pred_adj, gt_adj, pred_mask, gt_mask):
        """
        拓扑准确性reward (F1 score)
        
        Args:
            pred_adj: [B, N, N] 预测邻接矩阵
            gt_adj: [B, N, N] GT邻接矩阵
            pred_mask: [B, N] 预测有效mask
            gt_mask: [B, N] GT有效mask
        """
        B = pred_adj.shape[0]
        rewards = []
        
        for b in range(B):
            mask = gt_mask[b]
            if mask.sum() < 2:
                rewards.append(torch.tensor(0.5, device=pred_adj.device))
                continue
            
            # 只看有效部分
            pred_sub = pred_adj[b][mask][:, mask]
            gt_sub = gt_adj[b][mask][:, mask]
            
            # 二值化
            pred_binary = (pred_sub > 0.5).float()
            gt_binary = (gt_sub > 0.5).float()
            
            # F1 score
            tp = (pred_binary * gt_binary).sum()
            fp = (pred_binary * (1 - gt_binary)).sum()
            fn = ((1 - pred_binary) * gt_binary).sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            rewards.append(f1)
        
        return torch.stack(rewards)
    
    def smoothness_reward(self, coords, mask):
        """
        曲线平滑度reward
        通过计算二阶差分来衡量
        """
        B = coords.shape[0]
        rewards = []
        
        for b in range(B):
            valid_coords = coords[b][mask[b]]  # [M, P, 2]
            
            if valid_coords.shape[0] == 0:
                rewards.append(torch.tensor(0.5, device=coords.device))
                continue
            
            # 二阶差分 (曲率近似)
            if valid_coords.shape[1] >= 3:
                diff1 = valid_coords[:, 1:] - valid_coords[:, :-1]
                diff2 = diff1[:, 1:] - diff1[:, :-1]
                curvature = diff2.norm(dim=-1).mean()
                
                # 曲率越小越平滑
                reward = torch.exp(-curvature * 10)
            else:
                reward = torch.tensor(0.5, device=coords.device)
            
            rewards.append(reward)
        
        return torch.stack(rewards)
    
    def diversity_reward(self, candidates_coords, candidates_mask):
        """
        多样性reward (组内差异)
        鼓励生成不同的结果
        
        Args:
            candidates_coords: [K, B, N, P, 2] K个候选
            candidates_mask: [K, B, N] K个候选的mask
        """
        K, B = candidates_coords.shape[:2]
        
        if K < 2:
            return torch.zeros(K, B, device=candidates_coords.device)
        
        rewards = []
        for k in range(K):
            other_indices = [i for i in range(K) if i != k]
            
            batch_rewards = []
            for b in range(B):
                coord_k = candidates_coords[k, b]  # [N, P, 2]
                mask_k = candidates_mask[k, b]
                
                if mask_k.sum() == 0:
                    batch_rewards.append(torch.tensor(0.0, device=candidates_coords.device))
                    continue
                
                # 与其他候选的平均距离
                dists = []
                for other_k in other_indices:
                    coord_other = candidates_coords[other_k, b]
                    mask_other = candidates_mask[other_k, b]
                    
                    if mask_other.sum() == 0:
                        continue
                    
                    # 简化: 使用有效线的中心点距离
                    center_k = coord_k[mask_k].mean(dim=(0, 1))
                    center_other = coord_other[mask_other].mean(dim=(0, 1))
                    dist = (center_k - center_other).norm()
                    dists.append(dist)
                
                if len(dists) > 0:
                    avg_dist = torch.stack(dists).mean()
                    # 距离越大越好
                    div_reward = 1 - torch.exp(-avg_dist)
                else:
                    div_reward = torch.tensor(0.0, device=candidates_coords.device)
                
                batch_rewards.append(div_reward)
            
            rewards.append(torch.stack(batch_rewards))
        
        return torch.stack(rewards)  # [K, B]
    
    def forward(self, 
                pred_coords, pred_adj, pred_mask,
                gt_coords, gt_adj, gt_mask,
                candidates_coords=None, candidates_mask=None):
        """
        计算总reward
        
        Args:
            pred_coords: [B, N, P, 2]
            pred_adj: [B, N, N]
            pred_mask: [B, N]
            gt_coords: [B, N, P, 2]
            gt_adj: [B, N, N]
            gt_mask: [B, N]
            candidates_coords: [K, B, N, P, 2] 可选，用于diversity
            candidates_mask: [K, B, N]
        
        Returns:
            total_reward: [B]
            reward_dict: 各项reward
        """
        reward_dict = {}
        
        # Chamfer reward
        r_chamfer = self.chamfer_reward(pred_coords, gt_coords, pred_mask, gt_mask)
        reward_dict['chamfer'] = r_chamfer
        
        # Topology reward
        r_topo = self.topology_reward(pred_adj, gt_adj, pred_mask, gt_mask)
        reward_dict['topology'] = r_topo
        
        # Smoothness reward
        r_smooth = self.smoothness_reward(pred_coords, pred_mask)
        reward_dict['smoothness'] = r_smooth
        
        # Total (不包括diversity，那个在GRPO中单独处理)
        total = (self.chamfer_weight * r_chamfer + 
                 self.topology_weight * r_topo +
                 self.smoothness_weight * r_smooth)
        
        reward_dict['total'] = total
        
        return total, reward_dict


class GRPO(nn.Module):
    """
    Group Relative Policy Optimization (简化版 - DIVER风格)
    
    核心思想: 
    - 采样多个候选
    - 计算 reward
    - 用 reward 加权 Flow Matching Loss (不计算复杂的 log_prob)
    
    参考: DIVER (CVPR 2024)
    """
    def __init__(self,
                 num_samples=4,
                 temperature=1.0,
                 use_best_only=False):
        super().__init__()
        self.num_samples = num_samples
        self.temperature = temperature
        self.use_best_only = use_best_only  # 是否只用最佳样本
        
        self.reward_fn = CenterlineReward()
    
    def forward(self,
                sample_fn,
                flow_loss_fn,
                velocity_net,
                query_feat,
                bev_feat,
                gt_coords,
                gt_adj,
                gt_mask,
                x0_prior=None,
                cls_scores=None):
        """
        简化版 GRPO 训练步骤 (DIVER 风格)
        
        不计算真实 log_prob，而是:
        1. 采样多个候选
        2. 计算 reward
        3. 用 reward 加权 Flow Matching Loss
        
        Args:
            sample_fn: 采样函数 coords = sample_fn(query_feat, bev_feat)
            flow_loss_fn: Flow Matching Loss 函数
            velocity_net: 速度网络
            query_feat: [B, N, D] 或 [B, N, P, D] Query特征
            bev_feat: BEV特征
            gt_coords: [B, N, P, 2]
            gt_adj: [B, N, N]
            gt_mask: [B, N]
            x0_prior: [B, N, P, 2] 位置先验
            cls_scores: [B, N] 存在性分数 (可选)
        
        Returns:
            loss: GRPO损失
            log_dict: 日志信息
        """
        B = query_feat.shape[0]
        device = query_feat.device
        
        # 1. 采样 K 个候选 (detached，不需要梯度)
        candidates_coords = []
        candidates_mask = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                coords = sample_fn(query_feat, bev_feat)
                
                # 存在性 mask
                if cls_scores is not None:
                    mask = (cls_scores.squeeze(-1) > 0.3) if cls_scores.dim() == 3 else (cls_scores > 0.3)
                else:
                    mask = gt_mask
                
                candidates_coords.append(coords)
                candidates_mask.append(mask)
        
        candidates_coords = torch.stack(candidates_coords)  # [K, B, N, P, 2]
        candidates_mask = torch.stack(candidates_mask)      # [K, B, N]
        
        # 2. 计算每个候选的 reward
        rewards = []
        reward_details = []
        
        for k in range(self.num_samples):
            # 简化版: 只用 Chamfer reward
            r_chamfer = self.reward_fn.chamfer_reward(
                candidates_coords[k], gt_coords, 
                candidates_mask[k], gt_mask
            )
            r_smooth = self.reward_fn.smoothness_reward(
                candidates_coords[k], candidates_mask[k]
            )
            
            r = 0.7 * r_chamfer + 0.3 * r_smooth
            rewards.append(r)
            reward_details.append({'chamfer': r_chamfer, 'smoothness': r_smooth})
        
        rewards = torch.stack(rewards)  # [K, B]
        
        # 3. 计算权重 (softmax over samples)
        if self.use_best_only:
            # 只用最佳样本
            best_idx = rewards.argmax(dim=0)  # [B]
            weights = F.one_hot(best_idx, self.num_samples).float().T  # [K, B]
        else:
            # Softmax 加权
            weights = F.softmax(rewards / self.temperature, dim=0)  # [K, B]
        
        # 4. 加权 Flow Matching Loss
        # 对每个候选计算 flow loss，用 weight 加权
        total_loss = 0.0
        
        for k in range(self.num_samples):
            # 以候选 coords 为目标计算 flow loss
            # 这是 DIVER 的核心: 用 reward 高的样本作为监督目标
            candidate_target = candidates_coords[k].detach()  # [B, N, P, 2]
            
            # Flow Matching Loss: 学习从 noise 到 candidate 的映射
            loss_k = flow_loss_fn(
                velocity_net,
                candidate_target,
                query_feat,
                bev_feat,
                gt_mask,  # 使用 GT mask
                x0_prior=x0_prior
            )
            
            # 用 reward 加权
            weight_k = weights[k].mean()  # 取 batch 平均权重
            total_loss = total_loss + weight_k * loss_k
        
        # 5. 同时保留对 GT 的监督 (防止偏离太远)
        gt_loss = flow_loss_fn(
            velocity_net,
            gt_coords,
            query_feat,
            bev_feat,
            gt_mask,
            x0_prior=x0_prior
        )
        
        # 混合损失: candidate loss + GT loss
        alpha = 0.5  # GT 权重
        final_loss = (1 - alpha) * total_loss + alpha * gt_loss
        
        # 6. 日志信息
        log_dict = {
            'grpo/reward_mean': rewards.mean().item(),
            'grpo/reward_max': rewards.max().item(),
            'grpo/reward_min': rewards.min().item(),
            'grpo/weight_entropy': -(weights * (weights + 1e-8).log()).sum(dim=0).mean().item(),
            'grpo/chamfer_mean': torch.stack([d['chamfer'] for d in reward_details]).mean().item(),
            'grpo/smooth_mean': torch.stack([d['smoothness'] for d in reward_details]).mean().item(),
            'grpo/gt_loss': gt_loss.item(),
            'grpo/candidate_loss': total_loss.item(),
        }
        
        return final_loss, log_dict
