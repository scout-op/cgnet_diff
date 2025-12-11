import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ..modules.diffusion import ColdDiffusion
from ..modules.matcher import HungarianMatcher
from ..modules.sampler import BezierDeformableAttention
from ..modules.gnn_advanced import AdvancedTopologyGNN
from ..modules.jaq import JunctionAwareQuery
from ..modules.bsc import BezierSpaceConnection
from ..modules.utils import (
    fit_bezier, 
    bezier_interpolate,
    normalize_coords,
    denormalize_coords,
    chamfer_distance
)
from ..hooks.teacher_forcing import ProgressiveTrainingScheduler, TeacherForcingModule


@HEADS.register_module()
class DiffusionCenterlineHead(nn.Module):
    """
    扩散中心线检测头
    核心创新：在贝塞尔控制点空间进行扩散
    """
    
    def __init__(self,
                 num_classes=1,
                 embed_dims=256,
                 num_queries=50,
                 num_ctrl_points=4,
                 num_diffusion_steps=1000,
                 num_sampling_steps=4,
                 use_cold_diffusion=True,
                 num_decoder_layers=6,
                 pc_range=[-15.0, -30.0, -5.0, 15.0, 30.0, 3.0],
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, loss_weight=2.0),
                 loss_bezier=dict(type='L1Loss', loss_weight=5.0),
                 cost_class=1.0,
                 cost_bezier=5.0,
                 self_cond_prob=0.5,
                 renewal_threshold=0.3,
                 use_gnn=True,
                 use_jaq=False,
                 use_bsc=False,
                 dilate_radius=9,
                 with_multiview_supervision=True,
                 loss_topology=dict(type='BCELoss', loss_weight=1.0)):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_gnn = use_gnn
        self.use_jaq = use_jaq
        self.use_bsc = use_bsc
        self.with_multiview_supervision = with_multiview_supervision
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_ctrl_points = num_ctrl_points
        self.num_sampling_steps = num_sampling_steps
        self.pc_range = pc_range
        self.self_cond_prob = self_cond_prob
        self.renewal_threshold = renewal_threshold
        
        self.diffusion = ColdDiffusion(
            num_timesteps=num_diffusion_steps,
            beta_schedule='cosine'
        )
        
        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_bezier=cost_bezier
        )
        
        self.teacher_forcing = TeacherForcingModule(noise_std=0.02)
        self.progressive_scheduler = ProgressiveTrainingScheduler()
        
        if self.use_gnn:
            self.gnn = AdvancedTopologyGNN(
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 2,
                num_fcs=2,
                ffn_drop=0.1,
                edge_weight=0.8,
                num_layers=6
            )
        
        if self.use_jaq:
            self.jaq = JunctionAwareQuery(
                embed_dim=embed_dims,
                dilate_radius=dilate_radius,
                use_linear_attn=True
            )
        
        if self.use_bsc:
            self.bsc = BezierSpaceConnection(
                embed_dim=embed_dims,
                num_ctrl_points=num_ctrl_points,
                num_combined_points=8
            )
        
        self._init_layers()
        
        self.anchors = None
    
    def _init_layers(self):
        """初始化网络层"""
        
        self.time_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        
        self.ctrl_encoder = nn.Sequential(
            nn.Linear(self.num_ctrl_points * 2, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True)
        )
        
        self.self_cond_encoder = nn.Sequential(
            nn.Linear(self.num_ctrl_points * 2, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True)
        )
        
        self.bezier_attn = BezierDeformableAttention(
            embed_dim=self.embed_dims,
            num_heads=8,
            num_levels=1,
            num_points=4,
            num_sample_points=10
        )
        
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embed_dims,
                nhead=8,
                dim_feedforward=self.embed_dims * 4,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ) for _ in range(6)
        ])
        
        ctrl_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.num_ctrl_points * 2)
        )
        
        cls_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.num_classes)
        )
        
        if self.with_multiview_supervision:
            num_pred = 6
            self.ctrl_branches = nn.ModuleList([
                copy.deepcopy(ctrl_head) for _ in range(num_pred)
            ])
            self.cls_branches = nn.ModuleList([
                copy.deepcopy(cls_head) for _ in range(num_pred)
            ])
        else:
            self.ctrl_head = ctrl_head
            self.cls_head = cls_head
        
        self.confidence_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, 1),
            nn.Sigmoid()
        )
    
    def init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_head[-1].bias, bias_init)
    
    def load_anchors(self, anchor_path='work_dirs/kmeans_anchors.pth'):
        """加载预生成的锚点"""
        data = torch.load(anchor_path)
        self.anchors = data['anchors']
        print(f"✅ 加载锚点: {self.anchors.shape}")
    
    def get_sinusoidal_embeddings(self, timesteps, embedding_dim):
        """
        生成正弦位置编码（用于时间嵌入）
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return emb
    
    def forward_single_step(self, noisy_ctrl, bev_features, t, self_cond=None):
        """
        单步去噪（完整版本）
        
        Args:
            noisy_ctrl: [B, N, 4, 2]
            bev_features: [B, C, H, W]
            t: [B]
            self_cond: [B, N, 4, 2], 可选的自条件
        
        Returns:
            pred_ctrl: [B, N, 4, 2]
            pred_logits: [B, N, num_classes]
            features: [B, N, D]
        """
        B, N = noisy_ctrl.shape[:2]
        device = noisy_ctrl.device
        
        time_emb = self.get_sinusoidal_embeddings(t, self.embed_dims).to(device)
        time_emb = self.time_mlp(time_emb)
        
        ctrl_flat = noisy_ctrl.flatten(2)
        ctrl_emb = self.ctrl_encoder(ctrl_flat)
        
        if self_cond is not None:
            self_cond_flat = self_cond.flatten(2)
            self_cond_emb = self.self_cond_encoder(self_cond_flat)
            ctrl_emb = ctrl_emb + self_cond_emb
        
        ctrl_emb = ctrl_emb + time_emb.unsqueeze(1)
        
        junction_loss = None
        if self.use_jaq:
            ctrl_emb, junction_heatmap, junction_loss = self.jaq(
                ctrl_emb, bev_features, gt_junctions=None
            )
        
        H, W = bev_features.shape[2:]
        spatial_shapes = torch.tensor([[H, W]], device=device)
        
        bev_sampled_features = self.bezier_attn(
            query_embed=ctrl_emb,
            ctrl_points=noisy_ctrl,
            bev_features=bev_features,
            spatial_shapes=spatial_shapes,
            pc_range=self.pc_range
        )
        
        ctrl_emb = ctrl_emb + bev_sampled_features
        
        bev_flat = bev_features.flatten(2).permute(0, 2, 1)
        
        if self.with_multiview_supervision:
            intermediate_outputs = []
            tgt = ctrl_emb
            
            for layer in self.decoder_layers:
                tgt = layer(tgt, bev_flat)
                intermediate_outputs.append(tgt)
            
            all_pred_ctrl = []
            all_pred_logits = []
            
            for lvl, output in enumerate(intermediate_outputs):
                pred_ctrl_flat = self.ctrl_branches[lvl](output)
                pred_ctrl = pred_ctrl_flat.view(B, N, self.num_ctrl_points, 2)
                pred_ctrl = torch.tanh(pred_ctrl)
                all_pred_ctrl.append(pred_ctrl)
                
                pred_logits = self.cls_branches[lvl](output)
                all_pred_logits.append(pred_logits)
            
            return all_pred_ctrl, all_pred_logits, intermediate_outputs
        else:
            tgt = ctrl_emb
            for layer in self.decoder_layers:
                tgt = layer(tgt, bev_flat)
            
            pred_ctrl_flat = self.ctrl_head(tgt)
            pred_ctrl = pred_ctrl_flat.view(B, N, self.num_ctrl_points, 2)
            pred_ctrl = torch.tanh(pred_ctrl)
            
            pred_logits = self.cls_head(tgt)
            
            return pred_ctrl, pred_logits, tgt
    
    @force_fp32(apply_to=('bev_features',))
    def forward_train(self, bev_features, gt_bboxes_list, gt_labels_list, 
                     img_metas, epoch=0, gt_topology=None):
        """
        训练前向传播（完整版本）
        """
        B = len(gt_bboxes_list)
        device = bev_features.device
        
        train_config = self.progressive_scheduler.get_training_config(epoch, verbose=True)
        
        targets, gt_labels, pos_mask = self.prepare_gt(
            gt_bboxes_list, gt_labels_list, device
        )
        
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        
        if self.anchors is None:
            anchors = self.generate_default_anchors(device)
        else:
            anchors = self.anchors.to(device)
        
        if anchors.dim() == 3:
            anchors = anchors.unsqueeze(0).expand(B, -1, -1, -1)
        
        noisy_ctrl = self.diffusion.q_sample(targets, t, anchors=anchors)
        
        self_cond = None
        if torch.rand(1).item() < self.self_cond_prob:
            with torch.no_grad():
                outputs = self.forward_single_step(
                    noisy_ctrl, bev_features, t, self_cond=None
                )
                if self.with_multiview_supervision:
                    self_cond = outputs[0][-1]
                else:
                    self_cond = outputs[0]
        
        outputs = self.forward_single_step(
            noisy_ctrl, bev_features, t, self_cond=self_cond
        )
        
        if self.with_multiview_supervision:
            all_pred_ctrl, all_pred_logits, all_features = outputs
            pred_ctrl = all_pred_ctrl[-1]
            pred_logits = all_pred_logits[-1]
            features = all_features[-1]
        else:
            pred_ctrl, pred_logits, features = outputs
        
        bsc_loss = None
        enhanced_features = features
        if self.use_bsc and train_config.get('train_bsc', True):
            bsc_loss, enhanced_features = self.bsc(
                features, pred_ctrl, None
            )
        
        pred_topology = None
        if self.use_gnn and train_config['train_gnn']:
            gnn_input = self.teacher_forcing(
                pred_ctrl, targets, 
                train_config['teacher_forcing_prob'],
                training=True
            )
            
            gnn_features = enhanced_features if self.use_bsc else gnn_input
            pred_topology, pred_topology_logits = self.gnn(gnn_features)
        
        if self.with_multiview_supervision:
            losses = self.loss_multi_layer(
                all_pred_ctrl, all_pred_logits, all_features,
                targets, gt_labels, pos_mask,
                train_config,
                pred_topology=pred_topology,
                gt_topology=gt_topology,
                bsc_loss=bsc_loss
            )
        else:
            losses = self.loss(
                pred_ctrl, pred_logits, features,
                targets, gt_labels, pos_mask,
                train_config,
                pred_topology=pred_topology,
                gt_topology=gt_topology,
                bsc_loss=bsc_loss
            )
        
        return losses
    
    def forward_test(self, bev_features, img_metas):
        """
        测试/推理前向传播（完整版本）
        """
        B = bev_features.shape[0]
        device = bev_features.device
        
        if self.anchors is None:
            x_t = self.generate_default_anchors(device)
        else:
            x_t = self.anchors.to(device)
        
        x_t = x_t.unsqueeze(0).repeat(B, 1, 1, 1)
        
        timesteps = torch.linspace(
            self.diffusion.num_timesteps - 1, 0,
            self.num_sampling_steps,
            dtype=torch.long,
            device=device
        )
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            
            outputs = self.forward_single_step(
                x_t, bev_features, t_batch, self_cond=None
            )
            
            if self.with_multiview_supervision:
                pred_x0 = outputs[0][-1]
                pred_logits = outputs[1][-1]
                features = outputs[2][-1]
            else:
                pred_x0, pred_logits, features = outputs
            
            if i < len(timesteps) - 1:
                x_t = self.diffusion.ddim_sample_step(x_t, pred_x0, t)
                
                if t > timesteps[-1] // 2:
                    x_t = self.centerline_renewal(x_t, features)
            else:
                x_t = pred_x0
        
        pred_topology = None
        if self.use_gnn:
            pred_topology, _ = self.gnn(x_t)
        
        results = self.post_process(x_t, pred_logits, pred_topology, img_metas)
        
        return results
    
    def prepare_gt(self, gt_bboxes_list, gt_labels_list, device):
        """
        准备GT数据：DiffusionDet风格的处理
        确保GT和Anchor维度对齐
        """
        B = len(gt_bboxes_list)
        N = self.num_queries
        
        if self.anchors is None:
            anchors = self.generate_default_anchors(device)
        else:
            anchors = self.anchors.to(device)
        
        targets_list = []
        labels_list = []
        mask_list = []
        matched_anchor_indices_list = []
        
        for gt_bboxes, gt_labels in zip(gt_bboxes_list, gt_labels_list):
            if hasattr(gt_bboxes, 'instance_list'):
                centerlines = gt_bboxes.instance_list
            else:
                centerlines = gt_bboxes
            
            ctrl_points = []
            for line in centerlines:
                if isinstance(line, torch.Tensor):
                    line = line.cpu().numpy()
                
                ctrl = fit_bezier(line, n_control=self.num_ctrl_points)
                ctrl_points.append(ctrl)
            
            if len(ctrl_points) == 0:
                targets = anchors.clone()
                labels = torch.zeros(N, device=device, dtype=torch.long)
                mask = torch.zeros(N, device=device, dtype=torch.bool)
                matched_indices = torch.arange(N, device=device)
            else:
                M = len(ctrl_points)
                gt_ctrl = torch.from_numpy(np.stack(ctrl_points)).float().to(device)
                gt_ctrl_norm = normalize_coords(gt_ctrl, self.pc_range)
                
                gt_flat = gt_ctrl_norm.flatten(1)
                anchor_flat = anchors.flatten(1)
                dist_matrix = torch.cdist(gt_flat, anchor_flat, p=2)
                
                matched_indices = dist_matrix.argmin(dim=1)
                
                targets = anchors.clone()
                targets[matched_indices] = gt_ctrl_norm
                
                labels = torch.zeros(N, device=device, dtype=torch.long)
                labels[matched_indices] = gt_labels[:M]
                
                mask = torch.zeros(N, device=device, dtype=torch.bool)
                mask[matched_indices] = True
            
            targets_list.append(targets)
            labels_list.append(labels)
            mask_list.append(mask)
            matched_anchor_indices_list.append(matched_indices)
        
        targets = torch.stack(targets_list)
        labels = torch.stack(labels_list)
        mask = torch.stack(mask_list)
        
        return targets, labels, mask
    
    def generate_default_anchors(self, device):
        """
        生成默认锚点（如果未加载K-Means锚点）
        """
        anchors = []
        
        x_range = self.pc_range[3] - self.pc_range[0]
        y_range = self.pc_range[4] - self.pc_range[1]
        
        for i in range(self.num_queries):
            x_offset = (i / self.num_queries - 0.5) * x_range
            
            ctrl = torch.tensor([
                [x_offset, self.pc_range[1]],
                [x_offset, self.pc_range[1] + y_range * 0.33],
                [x_offset, self.pc_range[1] + y_range * 0.67],
                [x_offset, self.pc_range[4]]
            ], device=device, dtype=torch.float32)
            
            ctrl_normalized = normalize_coords(ctrl, self.pc_range)
            anchors.append(ctrl_normalized)
        
        return torch.stack(anchors)
    
    def centerline_renewal(self, ctrl_points, features):
        """
        中心线更新：替换低质量预测
        """
        B, N = ctrl_points.shape[:2]
        device = ctrl_points.device
        
        scores = self.confidence_head(features).squeeze(-1)
        
        for b in range(B):
            mask = scores[b] > self.renewal_threshold
            num_keep = mask.sum()
            num_renew = N - num_keep
            
            if num_renew > 0:
                if self.anchors is not None:
                    new_ctrl = self.anchors[:num_renew].to(device)
                else:
                    new_ctrl = self.generate_default_anchors(device)[:num_renew]
                
                kept_ctrl = ctrl_points[b][mask]
                ctrl_points[b] = torch.cat([kept_ctrl, new_ctrl], dim=0)
        
        return ctrl_points
    
    def build_full_topology_target(self, gt_topology, pos_mask, B, N, device):
        """
        构建N×N的完整拓扑目标
        背景线对应的行/列全为0
        
        Args:
            gt_topology: 原始GT拓扑 (M×M or list)
            pos_mask: [B, N], 标记哪些位置是GT
            B, N: batch size, num_queries
            device: torch.device
        
        Returns:
            full_topology: [B, N, N], 完整的拓扑矩阵
        """
        full_topology = torch.zeros(B, N, N, device=device)
        
        if gt_topology is None:
            return full_topology
        
        for b in range(B):
            pos_indices = pos_mask[b].nonzero(as_tuple=True)[0]
            M = len(pos_indices)
            
            if M > 0 and gt_topology is not None:
                if isinstance(gt_topology, list):
                    gt_topo_b = gt_topology[b]
                else:
                    gt_topo_b = gt_topology[b] if gt_topology.dim() == 3 else gt_topology
                
                if gt_topo_b.shape[0] == M:
                    for i in range(M):
                        for j in range(M):
                            full_topology[b, pos_indices[i], pos_indices[j]] = gt_topo_b[i, j]
        
        return full_topology
    
    def loss(self, pred_ctrl, pred_logits, features,
             targets, gt_labels, pos_mask, train_config,
             pred_topology=None, gt_topology=None, bsc_loss=None):
        """
        计算损失（使用prepare_gt的匹配结果）
        """
        B, N = pred_ctrl.shape[:2]
        
        loss_dict = {}
        
        loss_bezier = 0
        num_pos = pos_mask.sum()
        
        if num_pos > 0:
            for b in range(B):
                mask_b = pos_mask[b]
                if mask_b.sum() > 0:
                    loss_bezier += F.l1_loss(
                        pred_ctrl[b][mask_b],
                        targets[b][mask_b],
                        reduction='mean'
                    )
        
        loss_bezier = loss_bezier / max(num_pos, 1)
        
        target_labels = torch.zeros(B, N, device=pred_logits.device, dtype=torch.long)
        for b in range(B):
            target_labels[b][pos_mask[b]] = gt_labels[b][pos_mask[b]]
        
        loss_cls = F.cross_entropy(
            pred_logits.view(B * N, -1),
            target_labels.view(B * N),
            reduction='mean'
        )
        
        loss_dict['loss_cls'] = loss_cls * train_config['loss_weights']['geometry']
        loss_dict['loss_bezier'] = loss_bezier * train_config['loss_weights']['geometry']
        
        if pred_topology is not None and gt_topology is not None:
            full_gt_topology = self.build_full_topology_target(
                gt_topology, pos_mask, B, N, pred_topology.device
            )
            
            loss_topology = F.binary_cross_entropy_with_logits(
                pred_topology,
                full_gt_topology,
                reduction='mean'
            )
            loss_dict['loss_topology'] = loss_topology * train_config['loss_weights']['topology']
        
        if bsc_loss is not None:
            loss_dict['loss_bsc'] = bsc_loss * train_config['loss_weights'].get('bsc', 0.1)
        
        return loss_dict
    
    def loss_multi_layer(self, all_pred_ctrl, all_pred_logits, all_features,
                        gt_ctrl, gt_labels, gt_mask, train_config,
                        pred_topology=None, gt_topology=None, bsc_loss=None):
        """
        多层监督损失（Deep Supervision）
        """
        loss_dict = {}
        
        num_layers = len(all_pred_ctrl)
        
        total_loss_cls = 0
        total_loss_bezier = 0
        
        for lvl in range(num_layers):
            layer_loss = self.loss(
                all_pred_ctrl[lvl],
                all_pred_logits[lvl],
                all_features[lvl],
                gt_ctrl, gt_labels, gt_mask,
                train_config,
                pred_topology=None,
                gt_topology=None,
                bsc_loss=None
            )
            
            total_loss_cls += layer_loss['loss_cls']
            total_loss_bezier += layer_loss['loss_bezier']
        
        loss_dict['loss_cls'] = total_loss_cls / num_layers
        loss_dict['loss_bezier'] = total_loss_bezier / num_layers
        
        if pred_topology is not None and gt_topology is not None:
            B = all_pred_ctrl[0].shape[0]
            N = all_pred_ctrl[0].shape[1]
            
            full_gt_topology = self.build_full_topology_target(
                gt_topology, gt_mask, B, N, pred_topology.device
            )
            
            loss_topology = F.binary_cross_entropy_with_logits(
                pred_topology,
                full_gt_topology,
                reduction='mean'
            )
            loss_dict['loss_topology'] = loss_topology * train_config['loss_weights']['topology']
        
        if bsc_loss is not None:
            loss_dict['loss_bsc'] = bsc_loss * train_config['loss_weights'].get('bsc', 0.1)
        
        return loss_dict
    
    def post_process(self, ctrl_points, pred_logits, pred_topology, img_metas):
        """
        后处理：转换为最终输出格式
        """
        B = ctrl_points.shape[0]
        results = []
        
        for b in range(B):
            scores = pred_logits[b].softmax(-1)[:, 0]
            
            mask = scores > 0.3
            
            ctrl_denorm = denormalize_coords(
                ctrl_points[b][mask],
                self.pc_range
            )
            
            points = bezier_interpolate(ctrl_denorm, num_points=20)
            
            result = {
                'centerlines': points.cpu().numpy(),
                'scores': scores[mask].cpu().numpy(),
                'ctrl_points': ctrl_denorm.cpu().numpy()
            }
            
            if pred_topology is not None:
                topology_masked = pred_topology[b][mask][:, mask]
                result['topology'] = (topology_masked > 0.5).cpu().numpy()
                result['topology_scores'] = topology_masked.cpu().numpy()
            
            results.append(result)
        
        return results
