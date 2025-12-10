_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 100
queue_length = 1
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
nums_control_pts = 4
batch_size = 4

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

map_classes = ['centerline']
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

model = dict(
    type='DiffCGNet',
    use_grid_mask=True,
    video_test_mode=False,
    modality='vision',
    anchor_path='work_dirs/kmeans_anchors.pth',
    
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)
    ),
    
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    
    pts_bbox_head=dict(
        type='DiffusionCenterlineHead',
        num_classes=1,
        embed_dims=256,
        num_queries=50,
        num_ctrl_points=4,
        num_diffusion_steps=1000,
        num_sampling_steps=4,
        use_cold_diffusion=True,
        num_decoder_layers=6,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
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
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=4),
                            embed_dims=256,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                   'ffn', 'norm')))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bezier=dict(
            type='L1Loss',
            loss_weight=5.0
        ),
        cost_class=1.0,
        cost_bezier=5.0,
        self_cond_prob=0.5,
        renewal_threshold=0.3
    )
)

dataset_type = 'CustomNuScenesLocalMapDataset'
ann_root = 'data/nuscenes/anns/'
data_root = 'data/nuscenes'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1600, 900),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='CustomCollect3D', keys=['img'])
         ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
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
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        only_centerline=True,
        nums_control_pts=nums_control_pts,
        box_type_3d='LiDAR'
    ),
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
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=map_classes,
        only_centerline=True,
        nums_control_pts=nums_control_pts,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1
    ),
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
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=map_classes,
        only_centerline=True,
        nums_control_pts=nums_control_pts,
        classes=class_names,
        modality=input_modality
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline, metric=['chamfer', 'openlane', 'topology'])

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

fp16 = dict(loss_scale=512.)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

checkpoint_config = dict(interval=6, save_last=True)

work_dir = None
find_unused_parameters = True
seed = 1234
