# 🔍 CGNet配置对比分析

## 关键发现

通过对比`true_cgnet`（可运行版本）和我们的实现，发现以下关键配置：

---

## 📊 核心配置差异

### **1. 数据集类型** ⚠️ **关键**

```python
# 可运行版本
dataset_type = 'CustomNuScenesLocalMapDataset'  # ← 特定的数据集类
ann_file = 'data/nuscenes/anns/nuscenes_infos_temporal_train.pkl'

# 我们的版本
dataset_type = 'NuScenesCenterlineDataset'  # ← 可能不存在
ann_file = 'data/nuscenes/nuscenes_infos_temporal_train.pkl'
```

**需要修改**: 使用`CustomNuScenesLocalMapDataset`

---

### **2. Transformer配置** ✅ **已正确**

```python
# 可运行版本使用
transformer=dict(
    type='JAPerceptionTransformer',  # Junction Aware
    encoder=dict(
        type='BEVFormerEncoder',
        num_layers=1,  # ← 注意只有1层
        transformerlayers=dict(
            type='BEVFormerLayer',
            attn_cfgs=[
                dict(type='TemporalSelfAttention'),
                dict(type='GeometrySptialCrossAttention',
                     attention=dict(type='GeometryKernelAttention'))
            ]
        )
    ),
    decoder=dict(
        type='MapTRDecoder',
        num_layers=6
    )
)

# 我们的版本
transformer=dict(
    type='MapTRPerceptionTransformer',  # ← 需要改为JAPerceptionTransformer
    encoder=dict(
        type='BEVFormerEncoder',
        num_layers=3,  # ← 改为1
        ...
    )
)
```

**需要修改**: 
- Transformer type改为`JAPerceptionTransformer`
- Encoder layers改为1

---

### **3. Head配置** ⚠️

```python
# 可运行版本
pts_bbox_head=dict(
    type='CGTopoHead',  # ← CGNet原版的Head
    num_query=900,  # 50 * 18 (num_vec * num_pts)
    num_vec=50,
    num_pts_per_vec=20,
    nums_ctp=4,  # 贝塞尔控制点数
    dilate_radius=9,  # JAQ模块参数
    edge_weight=0.8,  # GNN参数
    ...
)

# 我们的版本
pts_bbox_head=dict(
    type='DiffusionCenterlineHead',  # ← 我们的扩散Head
    num_queries=50,
    num_ctrl_points=4,
    ...
)
```

---

### **4. 数据Pipeline** ⚠️ **关键**

```python
# 可运行版本
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),  # ← 缩放
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

# 我们的版本
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

**缺少**: 
- `ObjectRangeFilter`
- `ObjectNameFilter`
- `RandomScaleImageMultiViewImage`

---

### **5. 数据集参数** ⚠️ **关键**

```python
# 可运行版本的数据集参数
data.train = dict(
    type='CustomNuScenesLocalMapDataset',
    ann_file='data/nuscenes/anns/nuscenes_infos_temporal_train.pkl',
    bev_size=(200, 100),  # ← BEV尺寸
    pc_range=point_cloud_range,
    fixed_ptsnum_per_line=20,  # ← 固定点数
    eval_use_same_gt_sample_num_flag=True,
    padding_value=-10000,  # ← padding值
    map_classes=['centerline'],
    only_centerline=True,  # ← 只用中心线
    nums_control_pts=4,  # ← 控制点数
    queue_length=1,
    ...
)
```

---

## 🎯 需要立即修改的配置

### **修改清单**

1. ✅ **Transformer type**: `JAPerceptionTransformer`
2. ✅ **Encoder layers**: 1（不是3）
3. ⚠️ **Dataset type**: `CustomNuScenesLocalMapDataset`
4. ⚠️ **Ann file path**: `data/nuscenes/anns/...`
5. ⚠️ **添加数据集参数**: `bev_size`, `fixed_ptsnum_per_line`等
6. ⚠️ **添加pipeline**: `ObjectRangeFilter`, `RandomScaleImageMultiViewImage`

---

## 📋 关键参数对照表

| 参数 | 可运行版本 | 我们的版本 | 需要修改 |
|------|-----------|-----------|---------|
| dataset_type | CustomNuScenesLocalMapDataset | NuScenesCenterlineDataset | ✅ 是 |
| ann_file | anns/nuscenes_infos... | nuscenes_infos... | ✅ 是 |
| transformer.type | JAPerceptionTransformer | MapTRPerceptionTransformer | ✅ 是 |
| encoder.num_layers | 1 | 3 | ✅ 是 |
| bev_h | 200 | 未设置 | ✅ 是 |
| bev_w | 100 | 未设置 | ✅ 是 |
| nums_control_pts | 4 | 4 | ✅ 正确 |
| only_centerline | True | 未设置 | ✅ 是 |

---

## 🚀 立即行动

需要更新配置文件，使用可运行版本的设置！
