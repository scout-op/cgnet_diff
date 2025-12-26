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
        B_pred, N = pred_ctrl.shape[:2]
        B_gt = gt_ctrl.shape[0]
        M = gt_ctrl.shape[1]
        
        # 确保batch大小一致
        assert B_pred == B_gt, f"Batch size mismatch: pred_ctrl has {B_pred}, gt_ctrl has {B_gt}"
        B = B_gt
        
        indices = []
        for b in range(B):
            # 获取当前batch的数据
            pred_ctrl_b = pred_ctrl[b]  # (N, 4, 2)
            pred_logits_b = pred_logits[b]  # (N, num_classes)
            gt_ctrl_b = gt_ctrl[b]  # (M, 4, 2)
            gt_labels_b = gt_labels[b]  # (M,)
            
            # 处理无效标签 (padding的-1)
            valid_mask = gt_labels_b >= 0
            if valid_mask.sum() == 0:
                # 没有有效GT，返回空匹配
                indices.append((
                    torch.tensor([], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64)
                ))
                continue
            
            # 只对有效GT计算代价
            valid_gt_labels = gt_labels_b[valid_mask]
            valid_gt_ctrl = gt_ctrl_b[valid_mask]
            M_valid = valid_gt_labels.shape[0]
            
            # 计算分类代价 (改进: 使用sigmoid而非softmax，适配单类别输出)
            # pred_logits_b: [N, 1] 或 [N, num_classes]
            if pred_logits_b.shape[-1] == 1:
                # 单类别: 使用 sigmoid，前景概率越高 cost 越低
                pred_prob_b = pred_logits_b.sigmoid()  # [N, 1]
                # GT都是前景(label=0)，cost = 1 - sigmoid(logit) = 背景概率
                cost_class_b = (1 - pred_prob_b).expand(-1, M_valid)  # [N, M_valid]
            else:
                # 多类别: 使用 softmax
                pred_prob_b = pred_logits_b.softmax(-1)  # (N, num_classes)
                cost_class_b = -pred_prob_b[:, valid_gt_labels]  # (N, M_valid)
            
            # 计算贝塞尔控制点L1距离
            cost_bezier_b = torch.cdist(
                pred_ctrl_b.flatten(1),  # (N, 8)
                valid_gt_ctrl.flatten(1),  # (M_valid, 8)
                p=1
            )  # (N, M_valid)
            
            # 计算Chamfer距离 (如果需要)
            if self.cost_chamfer > 0:
                pred_points_b = bezier_interpolate(
                    pred_ctrl_b, self.num_sample_points
                )  # (N, num_sample_points, 2)
                gt_points_b = bezier_interpolate(
                    valid_gt_ctrl, self.num_sample_points
                )  # (M_valid, num_sample_points, 2)
                
                cost_chamfer_b = torch.zeros(N, M_valid, device=pred_ctrl.device)
                for i in range(N):
                    for j in range(M_valid):
                        cost_chamfer_b[i, j] = chamfer_distance(
                            pred_points_b[i:i+1], gt_points_b[j:j+1]
                        )
            else:
                cost_chamfer_b = 0
            
            # 总代价
            C_b = self.cost_class * cost_class_b + \
                  self.cost_bezier * cost_bezier_b + \
                  self.cost_chamfer * cost_chamfer_b
            
            # 处理 NaN/Inf 值（随机初始化的权重可能产生异常值）
            C_b = torch.nan_to_num(C_b, nan=1e6, posinf=1e6, neginf=-1e6)
            
            # 匈牙利匹配
            C_b_cpu = C_b.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_b_cpu)
            
            # 将col_ind映射回原始索引 (考虑valid_mask)
            valid_indices = torch.where(valid_mask)[0]
            col_ind_mapped = valid_indices[col_ind].cpu()
            
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind_mapped, dtype=torch.int64)
            ))
        
        return indices
