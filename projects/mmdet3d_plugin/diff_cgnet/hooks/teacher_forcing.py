import torch
import torch.nn as nn


class ProgressiveTrainingScheduler:
    """
    渐进式训练调度器
    管理Teacher Forcing概率和训练阶段
    """
    
    def __init__(self,
                 stage1_epochs=10,
                 stage2_epochs=20,
                 teacher_forcing_start=0.8,
                 teacher_forcing_end=0.0):
        """
        Args:
            stage1_epochs: 阶段1的epoch数（几何预热）
            stage2_epochs: 阶段2的epoch数（联合训练）
            teacher_forcing_start: TF起始概率
            teacher_forcing_end: TF结束概率
        """
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.teacher_forcing_start = teacher_forcing_start
        self.teacher_forcing_end = teacher_forcing_end
    
    def get_training_config(self, current_epoch):
        """
        根据当前epoch返回训练配置
        
        Returns:
            config: dict, 包含训练参数
        """
        if current_epoch < self.stage1_epochs:
            return {
                'stage': 1,
                'train_diffusion': True,
                'train_gnn': False,
                'teacher_forcing_prob': 1.0,
                'loss_weights': {
                    'geometry': 5.0,
                    'topology': 0.0,
                    'bezier': 0.1,
                    'direction': 0.005
                }
            }
        else:
            progress = (current_epoch - self.stage1_epochs) / self.stage2_epochs
            progress = min(progress, 1.0)
            
            tf_prob = self.teacher_forcing_start * (1 - progress) + \
                      self.teacher_forcing_end * progress
            
            return {
                'stage': 2,
                'train_diffusion': True,
                'train_gnn': True,
                'teacher_forcing_prob': max(0.0, tf_prob),
                'loss_weights': {
                    'geometry': 5.0,
                    'topology': 1.0,
                    'bezier': 0.1,
                    'direction': 0.005
                }
            }


class TeacherForcingModule(nn.Module):
    """
    Teacher Forcing模块（带噪声注入）
    """
    
    def __init__(self, noise_std=0.02):
        """
        Args:
            noise_std: 添加到GT的噪声标准差
        """
        super().__init__()
        self.noise_std = noise_std
    
    def forward(self, pred_ctrl, gt_ctrl, tf_prob, training=True):
        """
        执行Teacher Forcing
        
        Args:
            pred_ctrl: torch.Tensor, shape (B, N, 4, 2), 预测的控制点
            gt_ctrl: torch.Tensor, shape (B, M, 4, 2), GT控制点
            tf_prob: float, Teacher Forcing概率
            training: bool, 是否训练模式
        
        Returns:
            gnn_input: torch.Tensor, shape (B, N, 4, 2), GNN的输入
        """
        B, N = pred_ctrl.shape[:2]
        M = gt_ctrl.shape[1]
        
        if N != M:
            gt_ctrl_padded = torch.zeros_like(pred_ctrl)
            gt_ctrl_padded[:, :M] = gt_ctrl
            gt_ctrl = gt_ctrl_padded
        
        use_gt_mask = torch.rand(B, N, device=pred_ctrl.device) < tf_prob
        
        if training and tf_prob > 0:
            noise = torch.randn_like(gt_ctrl) * self.noise_std
            noisy_gt = gt_ctrl + noise
        else:
            noisy_gt = gt_ctrl
        
        gnn_input = torch.where(
            use_gt_mask.unsqueeze(-1).unsqueeze(-1),
            noisy_gt,
            pred_ctrl
        )
        
        return gnn_input
