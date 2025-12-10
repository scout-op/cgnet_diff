import torch
import torch.nn as nn
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.runner import force_fp32, auto_fp16

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


@DETECTORS.register_module()
class DiffCGNet(MVXTwoStageDetector):
    """
    DiffCGNet: 基于扩散模型的中心线图生成
    
    核心创新:
    1. 在贝塞尔控制点空间进行Cold Diffusion
    2. 渐进式训练策略（Teacher Forcing）
    3. 端到端生成几何和拓扑
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
                 modality='vision',
                 anchor_path='work_dirs/kmeans_anchors.pth'):
        
        super(DiffCGNet, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained
        )
        
        from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
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
        
        if hasattr(self.pts_bbox_head, 'load_anchors'):
            try:
                self.pts_bbox_head.load_anchors(anchor_path)
            except:
                print(f"⚠️  未找到锚点文件: {anchor_path}")
                print("   将使用默认锚点")
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """提取图像特征"""
        B = img.size(0)
        
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            
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
                         epoch=0):
        """
        中心线训练前向传播
        """
        outs = self.pts_bbox_head(
            pts_feats, 
            img_metas, 
            prev_bev
        )
        
        bev_features = outs if isinstance(outs, torch.Tensor) else outs[0]
        
        loss_inputs = dict(
            bev_features=bev_features,
            gt_bboxes_list=gt_bboxes_3d,
            gt_labels_list=gt_labels_3d,
            img_metas=img_metas,
            epoch=epoch
        )
        
        losses = self.pts_bbox_head.forward_train(**loss_inputs)
        
        return losses
    
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
                     img_depth=None,
                     img_mask=None,
                     epoch=0):
        """
        完整训练前向传播
        """
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        
        prev_img_metas = [each[:-1] for each in img_metas]
        img_metas = [each[-1] for each in img_metas]
        
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        else:
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev,
            epoch=epoch
        )
        
        losses.update(losses_pts)
        
        return losses
    
    def forward_test(self, img_metas, img=None, **kwargs):
        """
        测试前向传播
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        
        img = [img] if img is None else img
        
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev']
        )
        
        self.prev_frame_info['prev_bev'] = new_prev_bev
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']
        
        return bbox_results
    
    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """
        简单测试
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        bbox_list = [dict() for _ in range(len(img_metas))]
        
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return new_prev_bev, bbox_list
    
    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """
        中心线测试
        """
        outs = self.pts_bbox_head(x, img_metas, prev_bev)
        
        bev_features = outs if isinstance(outs, torch.Tensor) else outs[0]
        
        bbox_list = self.pts_bbox_head.forward_test(
            bev_features, img_metas
        )
        
        bbox_results = [
            dict(boxes_3d=result['centerlines'], 
                 scores_3d=result['scores'])
            for result in bbox_list
        ]
        
        return outs, bbox_results
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """
        获取历史BEV特征（用于时序信息）
        """
        self.eval()
        
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True
                )
        
        self.train()
        return prev_bev
