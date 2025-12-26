import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
from mmdet.models import HEADS, build_loss


def _neg_loss(pred, gt, weights=None):
    """Modified focal loss (CornerNet style).
    
    Args:
        pred: (batch, c, h, w) predictions
        gt: (batch, c, h, w) ground truth
        weights: optional weights for positive samples
    """
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
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding

from ..modules.diffusion import ColdDiffusion
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
from ..hooks.epoch_hook import get_global_epoch


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
                 bev_h=200,
                 bev_w=100,
                 transformer=None,
                 positional_encoding=None,
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
                 loss_topology=dict(type='BCELoss', loss_weight=1.0),
                 force_stage2=False,  # 手动控制是否启用阶段2
                 topology_dist_threshold=2.0,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_gnn = use_gnn
        self.use_jaq = use_jaq
        self.use_bsc = use_bsc
        self.dilate_radius = dilate_radius  # 路口热图膨胀半径
        self.with_multiview_supervision = with_multiview_supervision
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_ctrl_points = num_ctrl_points
        self.num_sampling_steps = num_sampling_steps
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.transformer = transformer
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.self_cond_prob = self_cond_prob
        self.renewal_threshold = renewal_threshold
        # Endpoint distance threshold (meters) for building topology targets / init adjacency.
        # Accepts a float (explicit meters) or "auto"/None (derive from BEV resolution).
        if topology_dist_threshold is None or (
            isinstance(topology_dist_threshold, str)
            and topology_dist_threshold.lower() == 'auto'
        ):
            cell_size = max(self.real_h / self.bev_h, self.real_w / self.bev_w)
            # Heuristic: a few BEV cells to tolerate bezier-fit + discretization noise.
            self.topology_dist_threshold = float(cell_size * 6.0)
        else:
            if isinstance(topology_dist_threshold, str):
                raise ValueError(
                    f'Unsupported topology_dist_threshold={topology_dist_threshold!r}; '
                    "use a float (meters) or 'auto'."
                )
            self.topology_dist_threshold = float(topology_dist_threshold)
        
        self.force_stage2 = force_stage2
        print(f"[DiffusionCenterlineHead] force_stage2={force_stage2}")
        
        self.diffusion = ColdDiffusion(
            num_timesteps=num_diffusion_steps,
            beta_schedule='cosine'
        )
        
        self.teacher_forcing = TeacherForcingModule(noise_std=0.02)
        self.progressive_scheduler = ProgressiveTrainingScheduler()
        
        # 初始化 Focal Loss
        self.loss_cls_fn = build_loss(loss_cls)
        
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
                dilate_radius=dilate_radius
            )
        
        if self.use_bsc:
            self.bsc = BezierSpaceConnection(
                embed_dim=embed_dims,
                num_ctrl_points=num_ctrl_points,
                num_combined_points=8
            )
        
        if transformer is not None:
            from mmcv.utils import build_from_cfg
            from mmdet.models.utils.builder import TRANSFORMER
            self.transformer = build_from_cfg(transformer, TRANSFORMER)
        else:
            self.transformer = None

        # Build BEV query embedding + positional encoding for BEVFormer-style encoder.
        # This is required to actually use the configured transformer for multi-view -> BEV fusion.
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        if positional_encoding is None:
            positional_encoding = dict(
                type='LearnedPositionalEncoding',
                num_feats=self.embed_dims // 2,
                row_num_embed=self.bev_h,
                col_num_embed=self.bev_w,
            )
        self.positional_encoding = build_positional_encoding(positional_encoding)
        
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
        
        # CGNet 风格：输出 1 类，use_sigmoid=True
        cls_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, 1)  # 单类别输出
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
        
        if self.with_multiview_supervision:
            for cls_branch in self.cls_branches:
                nn.init.constant_(cls_branch[-1].bias, bias_init)
        else:
            nn.init.constant_(self.cls_head[-1].bias, bias_init)
    
    def load_anchors(self, anchor_path='work_dirs/kmeans_anchors.pth'):
        """加载预生成的锚点"""
        data = torch.load(anchor_path)
        anchors = data['anchors']  # [N, 4, 2]
        
        # 检查是否已归一化（新版脚本生成的）
        already_normalized = data.get('normalized', False)
        
        # 检查 anchor 范围
        x_min_anchor, x_max_anchor = anchors[..., 0].min().item(), anchors[..., 0].max().item()
        y_min_anchor, y_max_anchor = anchors[..., 1].min().item(), anchors[..., 1].max().item()
        
        print(f"📊 Anchor 范围: X[{x_min_anchor:.2f}, {x_max_anchor:.2f}], Y[{y_min_anchor:.2f}, {y_max_anchor:.2f}]")
        
        # 检测是否已归一化（范围在 [0, 1] 内）
        is_in_unit_range = (x_min_anchor >= 0 and x_max_anchor <= 1 and
                           y_min_anchor >= 0 and y_max_anchor <= 1)
        
        if already_normalized or is_in_unit_range:
            # 已归一化，直接使用
            print("✅ Anchor 已归一化到 [0, 1]，直接使用")
            self.anchors = anchors
        else:
            # 需要归一化
            pc_x_min, pc_x_max = self.pc_range[0], self.pc_range[3]
            pc_y_min, pc_y_max = self.pc_range[1], self.pc_range[4]
            print(f"📊 配置 pc_range: X[{pc_x_min}, {pc_x_max}], Y[{pc_y_min}, {pc_y_max}]")
            
            # 检查范围是否在 pc_range 内
            anchor_in_range = (x_min_anchor >= pc_x_min - 5 and x_max_anchor <= pc_x_max + 5 and
                              y_min_anchor >= pc_y_min - 5 and y_max_anchor <= pc_y_max + 5)
            
            if anchor_in_range:
                print("✅ Anchor 在 pc_range 范围内，使用 pc_range 归一化")
                self.anchors = normalize_coords(anchors, self.pc_range)
            else:
                print("⚠️  警告: Anchor 超出 pc_range！将缩放到 [0.1, 0.9] 范围")
                print("   建议: 重新生成 anchor")
                
                anchors_normalized = anchors.clone()
                anchors_normalized[..., 0] = (anchors[..., 0] - x_min_anchor) / (x_max_anchor - x_min_anchor + 1e-6)
                anchors_normalized[..., 1] = (anchors[..., 1] - y_min_anchor) / (y_max_anchor - y_min_anchor + 1e-6)
                anchors_normalized = anchors_normalized * 0.8 + 0.1
                self.anchors = anchors_normalized
        
        print(f"✅ 加载锚点: {self.anchors.shape}")
        print(f"✅ 最终范围: X[{self.anchors[..., 0].min():.2f}, {self.anchors[..., 0].max():.2f}], Y[{self.anchors[..., 1].min():.2f}, {self.anchors[..., 1].max():.2f}]")
    
    def get_sinusoidal_embeddings(self, timesteps, embedding_dim):
        """
        生成正弦位置编码（用于时间嵌入）
        """
        device = timesteps.device
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return emb
    
    # Align with original CGNet: the BEVFormer-style transformer is numerically
    # unstable under fp16 in this codebase (can lead to grad_norm NaNs).
    # Keep the multi-view features and prev_bev in fp32 for BEV fusion.
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """
        统一的forward接口（与DiffCGNet调用）
        
        Args:
            mlvl_feats: 多尺度特征 (list or tensor)
            img_metas: 图像元数据
            prev_bev: 历史BEV特征
            only_bev: 是否只返囮BEV特征
        
        Returns:
            bev_embed: BEV特征，shape (B, bev_h*bev_w, embed_dims)
        """
        # Fallback: if no transformer is provided, return the first level feature.
        # NOTE: This is NOT a true BEV feature; it is only kept as a last-resort
        # debug path. Production should always use the transformer.
        if self.transformer is None:
            if isinstance(mlvl_feats, (list, tuple)):
                return mlvl_feats[0]
            return mlvl_feats

        if not isinstance(mlvl_feats, (list, tuple)):
            mlvl_feats = [mlvl_feats]

        bs = mlvl_feats[0].size(0)
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=device, dtype=dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # Use BEVFormer-style encoder to fuse multi-view features into BEV tokens.
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats=mlvl_feats,
            lidar_feat=None,
            bev_queries=bev_queries,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )
        return bev_embed
    
    def forward_single_step(
        self,
        noisy_ctrl,
        bev_features,
        t,
        self_cond=None,
        gt_junctions=None,
        enable_jaq=False,
    ):
        """
        单步去噪（完整版本）
        
        Args:
            noisy_ctrl: [B, N, 4, 2]
            bev_features: [B, C, H, W]
            t: [B]
            self_cond: [B, N, 4, 2], 可选的自条件
            gt_junctions: [B, H, W], GT路口热图（用于JAQ监督）
        
        Returns:
            pred_ctrl: [B, N, 4, 2]
            pred_logits: [B, N, num_classes]
            features: [B, N, D]
        """
        B, N = noisy_ctrl.shape[:2]
        device = noisy_ctrl.device
        
        # NaN 检测：输入
        if torch.isnan(noisy_ctrl).any():
            print(f"[NaN] noisy_ctrl contains NaN at input!")
        if torch.isnan(bev_features).any():
            print(f"[NaN] bev_features contains NaN at input!")
        
        time_emb = self.get_sinusoidal_embeddings(t, self.embed_dims).to(device)
        time_emb = self.time_mlp(time_emb)
        
        if torch.isnan(time_emb).any():
            print(f"[NaN] time_emb contains NaN after time_mlp!")
        
        ctrl_flat = noisy_ctrl.flatten(2)
        ctrl_emb = self.ctrl_encoder(ctrl_flat)
        
        if torch.isnan(ctrl_emb).any():
            print(f"[NaN] ctrl_emb contains NaN after ctrl_encoder!")
        
        if self_cond is not None:
            self_cond_flat = self_cond.flatten(2)
            self_cond_emb = self.self_cond_encoder(self_cond_flat)
            ctrl_emb = ctrl_emb + self_cond_emb
        
        ctrl_emb = ctrl_emb + time_emb.unsqueeze(1)
        
        # Normalize BEV features to (B, C, H, W).
        # The transformer returns (B, H*W, C). Keep a strict contract here to
        # avoid silently treating the camera dimension as time, etc.
        if bev_features.dim() == 3:
            # (B, H*W, C) or (B, C, H*W)
            if bev_features.shape[1] == self.bev_h * self.bev_w:
                bev_features = bev_features.permute(0, 2, 1).contiguous()
                bev_features = bev_features.view(B, -1, self.bev_h, self.bev_w)
            elif bev_features.shape[2] == self.bev_h * self.bev_w:
                bev_features = bev_features.contiguous().view(B, bev_features.shape[1], self.bev_h, self.bev_w)
            else:
                raise ValueError(
                    f"Unexpected BEV token shape {tuple(bev_features.shape)}; "
                    f"expected (B, H*W, C) with H*W={self.bev_h*self.bev_w}."
                )
        elif bev_features.dim() == 4:
            # Already (B, C, H, W)
            pass
        else:
            raise ValueError(
                f"Unexpected bev_features dim={bev_features.dim()} shape={tuple(bev_features.shape)}; "
                "expected BEV tokens (B, H*W, C) from transformer."
            )
        
        H, W = bev_features.shape[2:]
        spatial_shapes = torch.tensor([[H, W]], device=device)
        
        # JAQ: Junction Aware Query enhancement
        # Enable it explicitly (to keep stage1 geometry-only training untouched),
        # but allow inference to run without GT junction supervision.
        self.junction_loss = None
        self.junction_heatmap = None
        if self.use_jaq and enable_jaq:
            ctrl_emb, self.junction_heatmap, self.junction_loss = self.jaq(
                ctrl_emb, bev_features, gt_junctions=gt_junctions
            )
            if torch.isnan(ctrl_emb).any():
                print(f"[NaN] ctrl_emb contains NaN after JAQ!")
        
        bev_sampled_features = self.bezier_attn(
            query_embed=ctrl_emb,
            ctrl_points=noisy_ctrl,
            bev_features=bev_features,
            spatial_shapes=spatial_shapes,
            pc_range=self.pc_range
        )
        
        if torch.isnan(bev_sampled_features).any():
            print(f"[NaN] bev_sampled_features contains NaN after bezier_attn!")
        
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
                pred_ctrl = torch.sigmoid(pred_ctrl)  # [0,1] to match anchors/targets
                all_pred_ctrl.append(pred_ctrl)
                
                pred_logits = self.cls_branches[lvl](output)
                # 数值稳定性：防止 logits 极端值导致 Focal Loss NaN
                pred_logits = torch.clamp(pred_logits, min=-10, max=10)
                all_pred_logits.append(pred_logits)
            
            return all_pred_ctrl, all_pred_logits, intermediate_outputs
        else:
            tgt = ctrl_emb
            for layer in self.decoder_layers:
                tgt = layer(tgt, bev_flat)
            
            pred_ctrl_flat = self.ctrl_head(tgt)
            pred_ctrl = pred_ctrl_flat.view(B, N, self.num_ctrl_points, 2)
            pred_ctrl = torch.sigmoid(pred_ctrl)  # [0,1] to match anchors/targets
            
            pred_logits = self.cls_head(tgt)
            # 数值稳定性：防止 logits 极端值导致 Focal Loss NaN
            pred_logits = torch.clamp(pred_logits, min=-10, max=10)
            
            return pred_ctrl, pred_logits, tgt
    
    @force_fp32(apply_to=('bev_features',))
    def forward_train(self, bev_features, gt_bboxes_list, gt_labels_list, 
                     img_metas, epoch=None, gt_topology=None):
        """
        训练前向传播（完整版本）
        """
        B = len(gt_bboxes_list)
        device = bev_features.device
        
        # 优先使用传入的 epoch，否则使用 Hook 设置的 current_epoch
        if epoch is None:
            epoch = getattr(self, 'current_epoch', 0)
        
        # 手动控制阶段
        if self.force_stage2:
            train_config = {
                'stage': 2,
                'train_diffusion': True,
                'train_gnn': True,
                'train_bsc': True,
                'teacher_forcing_prob': 0.5,
                'loss_weights': {
                    'geometry': 5.0,
                    'topology': 1.0,
                    'bezier': 0.1,
                    'direction': 0.005
                }
            }
        else:
            train_config = self.progressive_scheduler.get_training_config(epoch, verbose=False)
        
        # DEBUG: 每 100 个 iter 打印一次
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 100 == 1:
            mode_str = 'FORCE_STAGE2' if self.force_stage2 else f'AUTO(epoch={epoch})'
            print(f"\n[DEBUG] {mode_str}, stage={train_config['stage']}, "
                  f"train_gnn={train_config.get('train_gnn', False)}, "
                  f"use_jaq={self.use_jaq}, use_gnn={self.use_gnn}")
        
        targets, gt_labels, pos_mask, gt_junctions, gt_topology_from_data = self.prepare_gt(
            gt_bboxes_list, gt_labels_list, device
        )

        if gt_topology is None:
            gt_topology = gt_topology_from_data

        # 如果没有传入 gt_topology，生成简单的 topology GT（基于端点距离）
        if gt_topology is None and train_config.get('train_gnn', False):
            gt_topology = self.generate_simple_topology(
                targets, pos_mask, threshold=self.topology_dist_threshold
            )
        
        # DEBUG: 检查 gt_junctions
        if self._debug_counter % 100 == 1:
            print(f"[DEBUG] gt_junctions is None: {gt_junctions is None}")
            print(f"[DEBUG] gt_topology is None: {gt_topology is None}")
            if gt_junctions is not None:
                print(f"[DEBUG] gt_junctions shape: {gt_junctions.shape if hasattr(gt_junctions, 'shape') else type(gt_junctions)}")
        
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        
        if self.anchors is None:
            anchors = self.generate_default_anchors(device)
        else:
            anchors = self.anchors.to(device)
        
        if anchors.dim() == 3:
            anchors = anchors.unsqueeze(0).expand(B, -1, -1, -1)
        
        noisy_ctrl = self.diffusion.q_sample(targets, t, anchors=anchors)
        
        # 只在阶段2使用JAQ（train_gnn=True时）
        use_jaq_this_step = self.use_jaq and train_config.get('train_gnn', False)
        gt_junctions_input = gt_junctions if use_jaq_this_step else None
        
        # DEBUG
        if self._debug_counter % 100 == 1:
            print(f"[DEBUG] use_jaq_this_step={use_jaq_this_step}, gt_junctions_input is None: {gt_junctions_input is None}")
        
        self_cond = None
        if torch.rand(1).item() < self.self_cond_prob:
            with torch.no_grad():
                outputs = self.forward_single_step(
                    noisy_ctrl,
                    bev_features,
                    t,
                    self_cond=None,
                    gt_junctions=gt_junctions_input,
                    enable_jaq=use_jaq_this_step,
                )
                if self.with_multiview_supervision:
                    self_cond = outputs[0][-1]
                else:
                    self_cond = outputs[0]
        
        outputs = self.forward_single_step(
            noisy_ctrl,
            bev_features,
            t,
            self_cond=self_cond,
            gt_junctions=gt_junctions_input,
            enable_jaq=use_jaq_this_step,
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
            connectivity = None
            if gt_topology is not None:
                # Use GT connectivity (or pseudo-GT) to activate BSC training.
                connectivity = self.build_full_topology_target(
                    gt_topology, pos_mask, B, N, device
                )
            bsc_loss, enhanced_features = self.bsc(
                features, pred_ctrl, connectivity
            )
        
        pred_topology = None
        pred_topology_logits = None
        if self.use_gnn and train_config['train_gnn']:
            # Teacher forcing affects topology by providing a more reliable
            # geometry-based init adjacency for the topology GNN.
            gnn_ctrl_input = self.teacher_forcing(
                pred_ctrl, targets,
                train_config['teacher_forcing_prob'],
                training=True,
            )
            # Build init adjacency in fp32 for numerical stability under AMP.
            gnn_ctrl_input_fp32 = gnn_ctrl_input.float()
            init_adj = self.build_init_adj_from_ctrl(
                gnn_ctrl_input_fp32,
                pos_mask=pos_mask,
                threshold=self.topology_dist_threshold,
            )

            gnn_features = enhanced_features if self.use_bsc else features
            # Run GNN in fp32 to avoid fp16 overflow producing inf logits,
            # which can yield NaN in BCEWithLogits (inf - inf).
            with torch.cuda.amp.autocast(enabled=False):
                pred_topology, pred_topology_logits_list = self.gnn(
                    gnn_features.float(),
                    init_adj=init_adj.float(),
                )
            # Use the last-layer logits for BCEWithLogitsLoss-style supervision.
            pred_topology_logits = pred_topology_logits_list[-1] if isinstance(
                pred_topology_logits_list, (list, tuple)) else pred_topology_logits_list
            # Clamp and sanitize logits to keep loss finite.
            pred_topology_logits = torch.clamp(pred_topology_logits, min=-20.0, max=20.0)
            pred_topology_logits = torch.nan_to_num(
                pred_topology_logits, nan=0.0, posinf=20.0, neginf=-20.0
            )
        
        if self.with_multiview_supervision:
            losses = self.loss_multi_layer(
                all_pred_ctrl, all_pred_logits, all_features,
                targets, gt_labels, pos_mask,
                train_config,
                pred_topology=pred_topology_logits,
                gt_topology=gt_topology,
                bsc_loss=bsc_loss,
                junction_loss=self.junction_loss
            )
        else:
            losses = self.loss(
                pred_ctrl, pred_logits, features,
                targets, gt_labels, pos_mask,
                train_config,
                pred_topology=pred_topology_logits,
                gt_topology=gt_topology,
                bsc_loss=bsc_loss,
                junction_loss=self.junction_loss
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
                x_t,
                bev_features,
                t_batch,
                self_cond=None,
                gt_junctions=None,
                enable_jaq=self.use_jaq,
            )
            
            if self.with_multiview_supervision:
                pred_x0 = outputs[0][-1]
                pred_logits = outputs[1][-1]
                features = outputs[2][-1]
            else:
                pred_x0, pred_logits, features = outputs
            
            if i < len(timesteps) - 1:
                # 传入下一个时间步用于正确的跳步采样
                t_prev = timesteps[i + 1]
                x_t = self.diffusion.ddim_sample_step(x_t, pred_x0, t, t_prev=t_prev)
                
                # 只在前半段进行 renewal（t > 中间时刻）
                mid_t = timesteps[0] // 2  # 例如 999 // 2 = 499
                if t > mid_t:
                    x_t = self.centerline_renewal(x_t, features)
            else:
                x_t = pred_x0
        
        pred_topology = None
        if self.use_gnn:
            # Keep inference consistent with stage2 training: build an init adjacency
            # from predicted geometry, and mask out low-confidence queries.
            scores = pred_logits.sigmoid().squeeze(-1)  # [B,N]
            pos_mask = scores > 0.3
            init_adj = self.build_init_adj_from_ctrl(
                x_t.float(),
                pos_mask=pos_mask,
                threshold=self.topology_dist_threshold,
            )
            with torch.cuda.amp.autocast(enabled=False):
                pred_topology, _ = self.gnn(features.float(), init_adj=init_adj.float())
        
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
        gt_topology_list = []
        
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
                matched_indices = torch.full((0,), -1, device=device, dtype=torch.long)
                gt_topology_b = torch.zeros(0, 0, device=device, dtype=torch.float32)
            else:
                M = len(ctrl_points)
                gt_ctrl = torch.from_numpy(np.stack(ctrl_points)).float().to(device)
                gt_ctrl_norm = normalize_coords(gt_ctrl, self.pc_range)
                
                gt_flat = gt_ctrl_norm.flatten(1)
                anchor_flat = anchors.flatten(1)
                # Unique assignment to avoid many-to-one collisions (multiple GT -> same query).
                # Prefer Hungarian matching; fall back to a greedy unique matcher if scipy is unavailable.
                dist_matrix = torch.cdist(gt_flat, anchor_flat, p=2)  # [M, N]
                try:
                    from scipy.optimize import linear_sum_assignment
                    row_ind, col_ind = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
                    row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)
                    col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)
                except Exception:
                    with torch.no_grad():
                        m, n = dist_matrix.shape
                        flat = dist_matrix.detach().cpu().view(-1)
                        sort_idx = torch.argsort(flat)
                        used_gt = torch.zeros(m, dtype=torch.bool)
                        used_q = torch.zeros(n, dtype=torch.bool)
                        rows = []
                        cols = []
                        for k in sort_idx.tolist():
                            r = k // n
                            c = k % n
                            if (not used_gt[r]) and (not used_q[c]):
                                used_gt[r] = True
                                used_q[c] = True
                                rows.append(r)
                                cols.append(c)
                                if len(rows) == min(m, n):
                                    break
                        row_ind = torch.tensor(rows, device=device, dtype=torch.long)
                        col_ind = torch.tensor(cols, device=device, dtype=torch.long)

                targets = anchors.clone()
                targets[col_ind] = gt_ctrl_norm[row_ind]
                
                # labels 不再需要，因为我们使用 pos_mask 来区分前景/背景
                labels = torch.zeros(N, device=device, dtype=torch.long)
                labels[col_ind] = gt_labels[row_ind]
                
                mask = torch.zeros(N, device=device, dtype=torch.bool)
                mask[col_ind] = True

                matched_indices = col_ind

                # Topology GT from dataset (if provided by LiDARInstanceLines).
                gt_topology_b = None
                if hasattr(gt_bboxes, 'adj_matrix') and gt_bboxes.adj_matrix is not None:
                    adj = gt_bboxes.adj_matrix
                    if isinstance(adj, torch.Tensor):
                        adj_t = adj.to(device=device, dtype=torch.float32)
                    else:
                        adj_t = torch.from_numpy(np.asarray(adj)).to(device=device, dtype=torch.float32)
                    # Keep only assigned GT rows/cols, and order by query index (pos_indices order).
                    order = torch.argsort(col_ind)
                    row_sorted = row_ind[order]
                    gt_topology_b = adj_t[row_sorted][:, row_sorted].contiguous()
                    if gt_topology_b.numel() > 0:
                        gt_topology_b.fill_diagonal_(0.0)
            
            targets_list.append(targets)
            labels_list.append(labels)
            mask_list.append(mask)
            matched_anchor_indices_list.append(matched_indices)
            gt_topology_list.append(gt_topology_b)
        
        targets = torch.stack(targets_list)
        labels = torch.stack(labels_list)
        mask = torch.stack(mask_list)
        
        # 生成GT路口热图（用于JAQ监督）
        gt_junctions_list = []
        for gt_bboxes in gt_bboxes_list:
            junction_map = self.get_bev_keypoint(gt_bboxes)
            gt_junctions_list.append(junction_map)
        gt_junctions = torch.stack(gt_junctions_list).to(device)

        # Only return topology if it is available for the whole batch.
        if any(t is None for t in gt_topology_list):
            gt_topology_list = None

        return targets, labels, mask, gt_junctions, gt_topology_list
    
    def get_bev_keypoint(self, lines):
        """从GT中心线提取路口点生成BEV热图
        
        Args:
            lines: GT数据（包含instance_list或key_points）
        
        Returns:
            bev_keypoint_map: torch.Tensor, shape (H, W)
        """
        bev_keypoint_map = np.zeros((self.bev_h, self.bev_w))
        
        # 获取关键点（端点）
        if hasattr(lines, 'key_points'):
            keypoints = lines.key_points
        elif hasattr(lines, 'instance_list'):
            # 手动提取端点
            keypoints = []
            for instance in lines.instance_list:
                if hasattr(instance, 'coords'):
                    coords = instance.coords
                else:
                    coords = instance
                if isinstance(coords, torch.Tensor):
                    coords = coords.cpu().numpy()
                if len(coords) > 0:
                    keypoints.append(coords[0])   # 起点
                    keypoints.append(coords[-1])  # 终点
        else:
            # 无有效数据
            return torch.tensor(bev_keypoint_map, dtype=torch.float32)
        
        # 将关键点映射到BEV坐标
        for key in keypoints:
            if isinstance(key, torch.Tensor):
                key = key.cpu().numpy()
            key = np.array(key)
            
            # 坐标转换：(x, y) -> (row, col)
            # CGNet使用 [::-1] 翻转坐标
            if len(key) >= 2:
                # x -> col, y -> row
                col = (key[0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]) * self.bev_w
                row = (key[1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]) * self.bev_h
                
                col = int(np.clip(col, 0, self.bev_w - 1))
                row = int(np.clip(row, 0, self.bev_h - 1))
                
                bev_keypoint_map[row, col] = 1
        
        # 膨胀热图
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_radius, self.dilate_radius))
        bev_keypoint_map = cv2.dilate(bev_keypoint_map, kernel)
        
        return torch.tensor(bev_keypoint_map, dtype=torch.float32)
    
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
    
    def generate_simple_topology(self, targets, pos_mask, threshold=2.0):
        """
        生成简单的 topology GT（基于端点距离）
        
        Args:
            targets: [B, N, 4, 2], 控制点
            pos_mask: [B, N], 有效 mask
            threshold: float, 距离阈值（米）
        
        Returns:
            gt_topology: list of [M, M] tensors
        """
        B, N = targets.shape[:2]
        gt_topology = []
        
        for b in range(B):
            pos_indices = pos_mask[b].nonzero(as_tuple=True)[0]
            M = len(pos_indices)
            
            if M == 0:
                gt_topology.append(torch.zeros(0, 0, device=targets.device))
                continue
            
            # 获取有效的控制点
            valid_ctrl = targets[b, pos_indices]  # [M, 4, 2]
            
            # 反归一化到实际坐标
            valid_ctrl_denorm = denormalize_coords(valid_ctrl, self.pc_range)
            
            # 计算端点（起点和终点）
            start_points = valid_ctrl_denorm[:, 0]  # [M, 2]
            end_points = valid_ctrl_denorm[:, -1]   # [M, 2]
            
            # 计算端点之间的距离矩阵
            topo = torch.zeros(M, M, device=targets.device)
            
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    
                    # 检查 i 的终点是否接近 j 的起点
                    dist = torch.norm(end_points[i] - start_points[j])
                    if dist < threshold:
                        topo[i, j] = 1.0
            
            gt_topology.append(topo)
        
        return gt_topology

    def build_init_adj_from_ctrl(self, ctrl_points, pos_mask=None, threshold=2.0):
        """Build an initial directed adjacency matrix from line endpoints.

        Used to make teacher forcing affect topology training: compute init_adj
        from (partially teacher-forced) geometry and feed it into the GNN.

        Args:
            ctrl_points (Tensor): [B, N, 4, 2] normalized coords in [0, 1].
            pos_mask (Tensor | None): [B, N] bool, foreground query mask.
            threshold (float): endpoint distance threshold in meters.

        Returns:
            Tensor: [B, N, N] float32 adjacency (0/1), no self-loops.
        """
        ctrl_denorm = denormalize_coords(ctrl_points, self.pc_range)  # [B,N,4,2]
        start_points = ctrl_denorm[:, :, 0, :]   # [B,N,2]
        end_points = ctrl_denorm[:, :, -1, :]    # [B,N,2]

        dist = torch.cdist(end_points, start_points)  # [B,N,N]
        init_adj = (dist < threshold).to(dtype=torch.float32)

        eye = torch.eye(init_adj.shape[-1], device=init_adj.device, dtype=init_adj.dtype)
        init_adj = init_adj * (1.0 - eye[None, :, :])

        if pos_mask is not None:
            valid = pos_mask.to(dtype=init_adj.dtype)
            init_adj = init_adj * valid[:, :, None] * valid[:, None, :]

        return init_adj

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

    def topology_loss_pos_only(self, pred_topology_logits, gt_topology, pos_mask):
        """CGNet-style topology loss: supervise only on the positive sub-graph.

        Args:
            pred_topology_logits (Tensor): [B, N, N] logits.
            gt_topology (list[Tensor] | Tensor): list of [M, M] per batch (preferred),
                or a full [B, N, N] tensor.
            pos_mask (Tensor): [B, N] bool.

        Returns:
            Tensor: scalar loss (mean over supervised edges).
        """
        B, N = pos_mask.shape
        device = pred_topology_logits.device

        total_loss = pred_topology_logits.new_tensor(0.0)
        total_edges = 0

        for b in range(B):
            pos_idx = pos_mask[b].nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0:
                continue

            pred_sub = pred_topology_logits[b].index_select(0, pos_idx).index_select(1, pos_idx)

            if isinstance(gt_topology, list):
                gt_sub = gt_topology[b]
            else:
                if gt_topology.dim() == 3:
                    gt_sub = gt_topology[b].index_select(0, pos_idx).index_select(1, pos_idx)
                else:
                    gt_sub = gt_topology

            if gt_sub is None or gt_sub.numel() == 0:
                continue

            gt_sub = gt_sub.to(device=device, dtype=pred_sub.dtype)
            if gt_sub.shape != pred_sub.shape:
                continue

            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                pred_sub, gt_sub, reduction='sum'
            )
            total_edges += pred_sub.numel()

        if total_edges == 0:
            return pred_topology_logits.new_tensor(0.0)

        return total_loss / float(total_edges)
    
    def loss(self, pred_ctrl, pred_logits, features,
             targets, gt_labels, pos_mask, train_config,
             pred_topology=None, gt_topology=None, bsc_loss=None, junction_loss=None):
        """
        计算损失（使用prepare_gt的匹配结果）
        """
        B, N = pred_ctrl.shape[:2]
        
        loss_dict = {}
        
        # NaN 检测
        if torch.isnan(pred_ctrl).any():
            print(f"[NaN DEBUG] pred_ctrl contains NaN!")
        if torch.isnan(targets).any():
            print(f"[NaN DEBUG] targets contains NaN!")
        if torch.isnan(pred_logits).any():
            print(f"[NaN DEBUG] pred_logits contains NaN!")
        
        loss_bezier = 0
        num_pos = 0
        
        for b in range(B):
            mask_b = pos_mask[b]
            if mask_b.sum() > 0:
                loss_bezier += F.l1_loss(
                    pred_ctrl[b][mask_b],
                    targets[b][mask_b],
                    reduction='sum'
                )
                num_pos += mask_b.sum()
        
        loss_bezier = loss_bezier / max(num_pos, 1)
        
        # CGNet/mmdet 风格 Focal Loss（use_sigmoid=True）
        # NOTE: mmdet 的 FocalLoss(use_sigmoid=True) 期望 target 为「类别索引」(long)，
        # 背景类索引 = num_classes。对于 1 类任务：前景=0，背景=1。
        target_labels = torch.full(
            (B, N),
            fill_value=self.num_classes,
            device=pred_logits.device,
            dtype=torch.long,
        )
        target_labels[pos_mask] = gt_labels[pos_mask]
        label_weights = torch.ones((B, N), device=pred_logits.device, dtype=pred_logits.dtype)
        
        # 计算 avg_factor（CGNet 风格）
        num_total_pos = float(pos_mask.sum().item())
        num_total_neg = float((B * N) - num_total_pos)
        cls_avg_factor = max(num_total_pos * 1.0 + num_total_neg * 0.1, 1.0)  # bg_cls_weight=0.1
        
        # pred_logits: [B, N, 1] → [B*N, 1]
        # target_labels: [B, N] → [B*N]
        loss_cls = self.loss_cls_fn(
            pred_logits.view(B * N, -1),        # (B*N, 1)
            target_labels.view(B * N),          # (B*N,)
            label_weights.view(B * N),          # (B*N,)
            avg_factor=cls_avg_factor,
        )
        
        # NaN 保护（CGNet 也有这个）
        loss_cls = torch.nan_to_num(loss_cls, nan=0.0, posinf=10.0, neginf=-10.0)
        
        loss_dict['loss_cls'] = loss_cls * train_config['loss_weights']['geometry']
        loss_dict['loss_bezier'] = loss_bezier * train_config['loss_weights']['geometry']
        
        # 始终添加 loss_topology，即使在阶段1也显示为 0.0
        if pred_topology is not None and gt_topology is not None:
            loss_topology = self.topology_loss_pos_only(pred_topology, gt_topology, pos_mask)
            loss_topology = torch.nan_to_num(loss_topology, nan=0.0, posinf=10.0, neginf=-10.0)
            loss_dict['loss_topology'] = loss_topology * train_config['loss_weights']['topology']
        else:
            # 阶段1：GNN冻结，显示为 0.0
            loss_dict['loss_topology'] = torch.tensor(0.0, device=pred_logits.device)
        
        # 始终显示 loss_bsc
        if bsc_loss is not None:
            loss_dict['loss_bsc'] = bsc_loss * train_config['loss_weights'].get('bsc', 0.1)
        else:
            loss_dict['loss_bsc'] = torch.tensor(0.0, device=pred_logits.device)
        
        # 始终显示 loss_junction  
        if junction_loss is not None:
            loss_dict['loss_junction'] = junction_loss * train_config['loss_weights'].get('junction', 0.1)
        else:
            loss_dict['loss_junction'] = torch.tensor(0.0, device=pred_logits.device)
        
        return loss_dict
    
    def loss_multi_layer(self, all_pred_ctrl, all_pred_logits, all_features,
                        gt_ctrl, gt_labels, gt_mask, train_config,
                        pred_topology=None, gt_topology=None, bsc_loss=None, junction_loss=None):
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
        
        # 始终添加 loss_topology
        if pred_topology is not None and gt_topology is not None:
            loss_topology = self.topology_loss_pos_only(pred_topology, gt_topology, gt_mask)
            loss_topology = torch.nan_to_num(loss_topology, nan=0.0, posinf=10.0, neginf=-10.0)
            loss_dict['loss_topology'] = loss_topology * train_config['loss_weights']['topology']
        else:
            # 阶段1：GNN冻结，显示为 0.0
            device = all_pred_ctrl[0].device
            loss_dict['loss_topology'] = torch.tensor(0.0, device=device)
        
        # 始终显示 loss_bsc
        if bsc_loss is not None:
            loss_dict['loss_bsc'] = bsc_loss * train_config['loss_weights'].get('bsc', 0.1)
        else:
            device = all_pred_ctrl[0].device
            loss_dict['loss_bsc'] = torch.tensor(0.0, device=device)
        
        # 始终显示 loss_junction
        if junction_loss is not None:
            loss_dict['loss_junction'] = junction_loss * train_config['loss_weights'].get('junction', 0.1)
        else:
            device = all_pred_ctrl[0].device
            loss_dict['loss_junction'] = torch.tensor(0.0, device=device)
        
        return loss_dict
    
    def post_process(self, ctrl_points, pred_logits, pred_topology, img_metas):
        """
        后处理：转换为最终输出格式
        """
        B = ctrl_points.shape[0]
        results = []
        
        for b in range(B):
            # CGNet 风格：cls_head 输出 1 类，使用 sigmoid
            scores = pred_logits[b].sigmoid().squeeze(-1)  # [N, 1] → [N]
            
            mask = scores > 0.3
            
            ctrl_denorm = denormalize_coords(
                ctrl_points[b][mask],
                self.pc_range
            )
            
            points = bezier_interpolate(ctrl_denorm, num_points=20)
            
            # 评估代码期望 tensor 格式，它会自己调用 .numpy()
            num_lines = mask.sum().item()
            result = {
                'line': points.cpu(),  # tensor
                'line_scores': scores[mask].cpu(),  # tensor
                'line_labels': torch.zeros(num_lines, dtype=torch.long).cpu(),  # tensor
                # 额外信息（可选）
                'centerlines': points.cpu().numpy(),
                'scores': scores[mask].cpu().numpy(),
                'ctrl_points': ctrl_denorm.cpu().numpy()
            }
            
            if pred_topology is not None:
                topology_masked = pred_topology[b][mask][:, mask]
                result['topology'] = (topology_masked > 0.5).cpu().numpy()
                result['topology_scores'] = topology_masked.cpu().numpy()
                # 评估代码期望的 adj_matrix 字段
                result['adj_matrix'] = (topology_masked > 0.5).cpu().numpy()
            else:
                # 如果没有拓扑预测，返回空矩阵
                result['adj_matrix'] = np.zeros((num_lines, num_lines), dtype=np.float32)
            
            results.append(result)
        
        return results
