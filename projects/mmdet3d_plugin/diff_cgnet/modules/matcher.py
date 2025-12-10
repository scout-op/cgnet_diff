import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from .utils import bezier_interpolate, chamfer_distance


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器
    用于在扩散训练中建立预测和GT的对应关系
    """
    
    def __init__(self, 
                 cost_class=1.0, 
                 cost_bezier=5.0,
                 cost_chamfer=0.0,
                 num_sample_points=20):
        """
        Args:
            cost_class: 分类代价权重
            cost_bezier: 贝塞尔控制点L1距离权重
            cost_chamfer: Chamfer距离权重（可选）
            num_sample_points: 用于Chamfer距离的采样点数
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bezier = cost_bezier
        self.cost_chamfer = cost_chamfer
        self.num_sample_points = num_sample_points
    
    @torch.no_grad()
    def forward(self, pred_ctrl, pred_logits, gt_ctrl, gt_labels):
        """
        执行匈牙利匹配
        
        Args:
            pred_ctrl: torch.Tensor, shape (B, N, 4, 2), 预测的控制点
            pred_logits: torch.Tensor, shape (B, N, num_classes), 预测的分类logits
            gt_ctrl: torch.Tensor, shape (B, M, 4, 2), GT控制点
            gt_labels: torch.Tensor, shape (B, M), GT标签
        
        Returns:
            indices: List[(row_ind, col_ind)], 长度为B
                     每个元素是该batch的匹配索引
        """
        B, N = pred_ctrl.shape[:2]
        M = gt_ctrl.shape[1]
        
        pred_ctrl_flat = pred_ctrl.flatten(0, 1)
        pred_prob = pred_logits.flatten(0, 1).softmax(-1)
        
        gt_ctrl_flat = gt_ctrl.flatten(0, 1)
        gt_labels_flat = gt_labels.flatten(0, 1)
        
        cost_class = -pred_prob[:, gt_labels_flat]
        
        cost_bezier = torch.cdist(
            pred_ctrl_flat.flatten(1),
            gt_ctrl_flat.flatten(1),
            p=1
        )
        
        if self.cost_chamfer > 0:
            pred_points = bezier_interpolate(
                pred_ctrl_flat, self.num_sample_points
            )
            gt_points = bezier_interpolate(
                gt_ctrl_flat, self.num_sample_points
            )
            
            cost_chamfer_matrix = torch.zeros(B * N, B * M, device=pred_ctrl.device)
            for i in range(B * N):
                for j in range(B * M):
                    cost_chamfer_matrix[i, j] = chamfer_distance(
                        pred_points[i:i+1], gt_points[j:j+1]
                    )
            cost_chamfer = cost_chamfer_matrix
        else:
            cost_chamfer = 0
        
        C = self.cost_class * cost_class + \
            self.cost_bezier * cost_bezier + \
            self.cost_chamfer * cost_chamfer
        
        C = C.view(B, N, M).cpu()
        
        indices = []
        for i in range(B):
            row_ind, col_ind = linear_sum_assignment(C[i])
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64)
            ))
        
        return indices
