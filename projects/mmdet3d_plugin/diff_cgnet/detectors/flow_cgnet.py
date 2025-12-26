"""
FlowCGNet: Flow Matching + GRPO for Centerline Generation
基于CGNet改进，使用Flow Matching替代Cold Diffusion
"""

import copy
import torch
import torch.nn as nn
import os
from mmdet.models import DETECTORS, build_backbone, build_neck, build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.runner import force_fp32, auto_fp16


@DETECTORS.register_module()
class FlowCGNet(MVXTwoStageDetector):
    """
    FlowCGNet: Flow Matching中心线生成
    
    核心设计:
    1. Flow Matching生成Bezier控制点 (单步采样)
    2. MLP预测拓扑邻接矩阵
    3. GRPO强化学习优化
    """
    
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 modality='vision'):
        
        super(FlowCGNet, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained
        )
        
        from ...models.utils.grid_mask import GridMask
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode
        self.modality = modality
        
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """提取图像特征"""
        B = img.size(0)
        
        if img is not None:
            if img.dim() == 6:
                B, T, N, C, H, W = img.size()
                img = img.reshape(B * T * N, C, H, W)
            elif img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            elif img.dim() == 4:
                pass
            else:
                raise ValueError(f"Unexpected image dimension: {img.shape}")
            
            if self.use_grid_mask:
                img = self.grid_mask(img)
            
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped
    
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """提取特征"""
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats
    
    def forward_pts_train(self,
                         pts_feats,
                         gt_bboxes_3d,
                         gt_labels_3d,
                         img_metas,
                         gt_bboxes_ignore=None,
                         prev_bev=None,
                         gt_topology=None):
        """
        中心线训练前向传播 (CGNet风格)
        """
        # 使用Transformer编码BEV和Query特征
        outputs = self.pts_bbox_head(pts_feats, img_metas, prev_bev, only_bev=False)
        debug_enabled = os.environ.get('CGNET_DEBUG_FLOW_SHAPES', '').lower() in ('1', 'true', 'yes', 'y')
        is_main_process = os.environ.get('RANK', '0') in ('0', '-1')
        
        if self.pts_bbox_head.transformer is not None:
            # 有Transformer: 返回 (bev_embed, hs)
            bev_features, hs = outputs
            if debug_enabled and is_main_process and not getattr(self, '_debug_shapes_printed', False):
                self._debug_shapes_printed = True
                print(
                    f'[FlowCGNet.forward_pts_train] transformer outputs: '
                    f'bev_features={tuple(bev_features.shape)} hs={tuple(hs.shape)}',
                    flush=True,
                )
                if bev_features.dim() == 3 and bev_features.shape[0] == self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w:
                    print(
                        '[FlowCGNet.forward_pts_train] NOTE: bev_features is [H*W, B, C] from transformer.',
                        flush=True,
                    )
            # hs: [num_layers, N_total, B, D] (decoder输出格式)
            # N_total = num_instances * num_ctrl_points (instance_pts模式)
            # 使用最后一层的Query特征
            hs_last = hs[-1]  # [N_total, B, D]
            N_total, B, D = hs_last.shape
            
            # 转换为 [B, N_total, D]
            query_feat_flat = hs_last.permute(1, 0, 2)  # [B, N_total, D]
            
            # 如果是 instance_pts 模式，reshape 为 [B, num_instances, num_ctrl_points, D]
            if self.pts_bbox_head.query_embed_type == 'instance_pts':
                num_instances = self.pts_bbox_head.num_instances
                num_query_points = self.pts_bbox_head.flow_num_points
                # [B, num_instances * num_query_points, D] -> [B, num_instances, num_query_points, D]
                query_feat = query_feat_flat.view(B, num_instances, num_query_points, D)
            else:
                # all_pts 模式: [B, N, D]，每个query代表一条线
                query_feat = query_feat_flat
            if debug_enabled and is_main_process and not getattr(self, '_debug_query_shapes_printed', False):
                self._debug_query_shapes_printed = True
                print(
                    f'[FlowCGNet.forward_pts_train] query_feat={tuple(query_feat.shape)} '
                    f'query_embed_type={self.pts_bbox_head.query_embed_type}',
                    flush=True,
                )
        else:
            # 无Transformer: 直接使用embedding
            bev_features = outputs
            if isinstance(bev_features, (list, tuple)):
                bev_features = bev_features[0]
            
            B = bev_features.shape[0]
            device = bev_features.device
            
            # Query特征 (CGNet风格: embedding split成query和pos)
            if hasattr(self.pts_bbox_head, 'query_embedding'):
                query_embeds = self.pts_bbox_head.query_embedding.weight[None].expand(B, -1, -1)
                query_feat = query_embeds[..., :self.pts_bbox_head.embed_dims]
                query_pos = query_embeds[..., self.pts_bbox_head.embed_dims:]
                query_feat = query_feat + query_pos
            else:
                if bev_features.dim() == 4:
                    query_feat = bev_features.flatten(2).permute(0, 2, 1)
                    query_feat = query_feat[:, :self.pts_bbox_head.num_queries, :]
                else:
                    query_feat = bev_features[:, :self.pts_bbox_head.num_queries, :]
        
        # 训练
        losses = self.pts_bbox_head.forward_train(
            bev_features=bev_features,
            query_feat=query_feat,
            gt_bboxes_list=gt_bboxes_3d,
            gt_labels_list=gt_labels_3d,
            img_metas=img_metas,
            gt_topology=gt_topology,
        )
        
        return losses
    
    @force_fp32(apply_to=('img', 'points'))
    def forward_train(self,
                     points=None,
                     img_metas=None,
                     gt_bboxes_3d=None,
                     gt_labels_3d=None,
                     gt_labels=None,
                     gt_bboxes=None,
                     img=None,
                     proposals=None,
                     gt_bboxes_ignore=None,
                     gt_topology=None):
        """训练入口"""
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        losses = dict()
        
        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            gt_topology=gt_topology,
        )
        
        losses.update(losses_pts)
        
        return losses
    
    def forward_pts_test(self, pts_feats, img_metas, prev_bev=None):
        """测试前向传播 (CGNet风格)"""
        outputs = self.pts_bbox_head(pts_feats, img_metas, prev_bev, only_bev=False)
        
        if self.pts_bbox_head.transformer is not None:
            # 有Transformer
            bev_features, hs = outputs
            hs_last = hs[-1]  # [N_total, B, D]
            N_total, B, D = hs_last.shape
            query_feat_flat = hs_last.permute(1, 0, 2)  # [B, N_total, D]
            
            if self.pts_bbox_head.query_embed_type == 'instance_pts':
                num_instances = self.pts_bbox_head.num_instances
                num_query_points = self.pts_bbox_head.flow_num_points
                query_feat = query_feat_flat.view(B, num_instances, num_query_points, D)
            else:
                query_feat = query_feat_flat
        else:
            # 无Transformer
            bev_features = outputs
            if isinstance(bev_features, (list, tuple)):
                bev_features = bev_features[0]
            
            B = bev_features.shape[0]
            
            if hasattr(self.pts_bbox_head, 'query_embedding'):
                query_embeds = self.pts_bbox_head.query_embedding.weight[None].expand(B, -1, -1)
                query_feat = query_embeds[..., :self.pts_bbox_head.embed_dims]
                query_pos = query_embeds[..., self.pts_bbox_head.embed_dims:]
                query_feat = query_feat + query_pos
            else:
                if bev_features.dim() == 4:
                    query_feat = bev_features.flatten(2).permute(0, 2, 1)
                    query_feat = query_feat[:, :self.pts_bbox_head.num_queries, :]
                else:
                    query_feat = bev_features[:, :self.pts_bbox_head.num_queries, :]
        
        results = self.pts_bbox_head.forward_test(
            bev_features=bev_features,
            query_feat=query_feat,
            img_metas=img_metas,
        )
        
        return results
    
    def forward_test(self, img_metas, img=None, **kwargs):
        """测试入口"""
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list')
        
        img = [img] if img is None else img
        
        # 处理 img_metas 格式 (可能是 list of list 或 list of dict)
        if isinstance(img_metas[0], list):
            img_metas_inner = img_metas[0][0]
        else:
            img_metas_inner = img_metas[0]
        
        scene_token = img_metas_inner.get('scene_token', None)
        if scene_token != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        
        self.prev_frame_info['scene_token'] = scene_token
        
        img = img[0]
        img_metas = img_metas[0]
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        results = self.forward_pts_test(
            img_feats, 
            img_metas,
            prev_bev=self.prev_frame_info.get('prev_bev')
        )
        
        return results
    
    def simple_test(self, img_metas, img=None, **kwargs):
        """简单测试"""
        return self.forward_test(img_metas, img=img, **kwargs)
