"""
FlowCGNet配置文件 (完整版)
Flow Matching + GRPO + 完整Transformer
以精度为主，集成原始CGNet的所有组件
"""

_base_ = [
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# 点云范围
point_cloud_range = [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]

# BEV尺寸
bev_h_ = 200
bev_w_ = 100

# 类别
num_classes = 1
map_classes = ['centerline']

# 维度
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1

# 固定点数
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
nums_control_pts = 4

# 模型配置
model = dict(
    type='FlowCGNet',
    use_grid_mask=True,
    modality='vision',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        with_cp=True,
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type='FlowCenterlineHead',
        num_classes=num_classes,
        embed_dims=_dim_,
        num_queries=50,
        num_pts_per_line=fixed_ptsnum_per_pred_line,
        num_ctrl_points=nums_control_pts,
        num_flow_layers=4,
        num_gnn_layers=6,
        num_sampling_steps=1,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        use_grpo=True,
        grpo_samples=8,
        grpo_start_epoch=12,
        query_embed_type='all_pts',
        # 完整Transformer配置 (CGNet风格)
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
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_pos_dim_,
            normalize=True,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_flow_weight=5.0,
        loss_topo_weight=1.0,
    ),
    train_cfg=dict(pts=dict()),
    test_cfg=dict(pts=dict()),
)

# 数据集配置
dataset_type = 'NuScenesMapDataset'
data_root = 'data/nuscenes/'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='NormalizeMultiViewImage', **dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False
    )),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=map_classes),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiViewImage', **dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False
    )),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=map_classes, with_label=False),
            dict(type='CustomCollect3D', keys=['img']),
        ]
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=map_classes,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=map_classes,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=map_classes,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        box_type_3d='LiDAR',
    ),
)

# 优化器
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率调度
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# 训练配置
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# 检查点
checkpoint_config = dict(interval=1)

# 日志
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# 评估
evaluation = dict(interval=1, pipeline=test_pipeline)

# 预训练
load_from = None
resume_from = None

# 工作目录
work_dir = 'work_dirs/flow_cgnet_r50_nusc_full'
