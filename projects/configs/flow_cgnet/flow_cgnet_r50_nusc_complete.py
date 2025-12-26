"""
FlowCGNet配置文件 (完整版 - 对齐原始CGNet)
Flow Matching + GRPO for Centerline Generation
包含所有原始CGNet的配置项
"""

_base_ = [
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# 点云范围 (与原始CGNet一致)
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

# 图像归一化 (与原始CGNet一致)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 类别
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_classes = ['centerline']
num_map_classes = len(map_classes)

# ===== 训练参数 (可手动调整) =====
batch_size = 12       # 每个GPU的batch size
num_workers = 8      # 数据加载线程数

# BEV尺寸
bev_h_ = 200
bev_w_ = 100
queue_length = 1

# 固定点数
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
nums_control_pts = 4
eval_use_same_gt_sample_num_flag = True

# 维度
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1

# 输入模态
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

# 模型配置
model = dict(
    type='FlowCGNet',
    use_grid_mask=True,
    video_test_mode=False,
    # ===== 预训练权重 (重要!) =====
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='FlowCenterlineHead',
        num_classes=num_map_classes,
        embed_dims=_dim_,
        num_queries=50,
        num_pts_per_line=fixed_ptsnum_per_pred_line,
        num_ctrl_points=nums_control_pts,
        num_flow_layers=4,
        num_gnn_layers=6,
        num_sampling_steps=5,  # 增加采样步数，提高精度
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        use_grpo=True,   # 启用简化版 GRPO (DIVER 风格)
        grpo_samples=4,  # 减少采样数，提高效率
        grpo_start_epoch=12,
        query_embed_type='instance_pts',  # CGNet风格
        # ===== JAQ (Junction-Aware Query) =====
        use_jaq=True,
        loss_kp_weight=0.1,  # 降低权重，使各loss量级接近
        dilate_radius=3,
        # ===== 完整Transformer配置 (CGNet风格) =====
        transformer=dict(
            type='JAPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3, 5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                )
            ),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                )
            ),
        ),
        # ===== bbox_coder (CGNet风格后处理) =====
        bbox_coder=dict(
            type='CGNetNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            score_threshold=0.3,
            max_num=50,
            adj_threshold=0.9,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        # ===== Positional Encoding (Learned, CGNet风格) =====
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        # ===== 损失函数 =====
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_flow_weight=1.0,  # 从5.0降到1.0，防止梯度爆炸
        loss_topo_weight=1.0,
    ),
    # ===== train_cfg (包含assigner) =====
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
            pc_range=point_cloud_range))),
    test_cfg=dict(pts=dict()),
)

# ===== 数据集配置 (与原始CGNet一致) =====
dataset_type = 'CustomNuScenesLocalMapDataset'
# 注意: pkl文件可能在 anns/ 或 train/ 目录下，根据实际情况调整
ann_root = 'data/nuscenes/train/'  # 或 'data/nuscenes/anns/'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesNus', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]


data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        only_centerline=True,
        nums_control_pts=nums_control_pts,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_root + 'nuscenes_infos_temporal_val.pkl',
        map_ann_file=ann_root + 'nuscenes_map_anns_val_centerline.json',
        graph_ann_file=ann_root + 'nuscenes_graph_anns_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        only_centerline=True,
        nums_control_pts=nums_control_pts,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_root + 'nuscenes_infos_temporal_val.pkl',
        map_ann_file=ann_root + 'nuscenes_map_anns_val_centerline.json',
        graph_ann_file=ann_root + 'nuscenes_graph_anns_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        only_centerline=True,
        nums_control_pts=nums_control_pts,
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# ===== 优化器 (与原始CGNet一致) =====
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # 降低学习率，防止梯度爆炸
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))  # 加强梯度裁剪

# ===== 学习率调度 =====
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# ===== 训练配置 (110 epochs与原始CGNet一致) =====
total_epochs = 110

# ===== 评估 (包含所有指标) =====
evaluation = dict(interval=110, pipeline=test_pipeline, metric=['chamfer', 'openlane', 'topology'])

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# ===== 日志 =====
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# ===== FP16训练 =====l
fp16 = dict(loss_scale=512.)

# ===== 检查点 =====
checkpoint_config = dict(interval=1, save_last=True)

# ===== 其他 =====
find_unused_parameters = True
seed = 1234

# ===== 自定义Hooks (GRPO需要InjectEpochHook注入current_epoch) =====
custom_hooks = [
    dict(type='InjectEpochHook', priority='NORMAL'),
]
# ===== 从头训练 (之前的 checkpoint 已损坏) =====
# load_from = None
resume_from = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/cg/fg/work_dirs/flow_cgnet_r50_nusc_complete_v4/epoch_12.pth'
work_dir = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/cg/fg/work_dirs/flow_cgnet_r50_nusc_complete_v4'
