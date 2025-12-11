# 🚀 服务器部署指南

## 📁 服务器数据结构

```bash
/data/roadnet_data/cg/CGNet/mmdetection3d/
├── data/
│   └── nuscenes/
│       ├── anns/  ✅ 预处理标注（关键）
│       │   ├── nuscenes_infos_temporal_train.pkl (2.6GB)
│       │   ├── nuscenes_infos_temporal_val.pkl (580MB)
│       │   ├── nuscenes_map_anns_val_centerline.json
│       │   └── nuscenes_graph_anns_val.pkl
│       ├── samples/  # 原始图像
│       ├── sweeps/
│       ├── maps/
│       └── v1.0-trainval/
```

---

## ✅ 数据已预处理，无需重新处理

**检查清单**:
- [x] 训练标注存在
- [x] 验证标注存在
- [x] 中心线标注存在
- [x] 拓扑标注存在

**结论**: 直接使用即可！

---

## 🚀 部署步骤

### **Step 1: 上传代码**

```bash
# 在本地打包
cd /home/subobo/ro/e2e/CGNet
tar -czf diffcgnet_code.tar.gz \
    projects/mmdet3d_plugin/diff_cgnet/ \
    configs/diff_cgnet/ \
    tools/generate_anchors.py \
    tools/train_diff_cgnet.sh \
    tools/debug_forward.py \
    tools/eval_diff_cgnet.py \
    tools/visualize_diff_cgnet.py \
    START_HERE.sh \
    READY_FOR_TRAINING.md

# 上传
scp diffcgnet_code.tar.gz server:/data/roadnet_data/cg/CGNet/mmdetection3d/
```

### **Step 2: 在服务器解压**

```bash
# SSH到服务器
ssh server

# 解压
cd /data/roadnet_data/cg/CGNet/mmdetection3d/
tar -xzf diffcgnet_code.tar.gz

# 验证
ls projects/mmdet3d_plugin/diff_cgnet/
ls configs/diff_cgnet/
```

### **Step 3: 验证数据路径**

```bash
# 检查数据文件
ls -lh data/nuscenes/anns/*.pkl

# 应该看到:
# nuscenes_infos_temporal_train.pkl (2.6GB)
# nuscenes_infos_temporal_val.pkl (580MB)
# nuscenes_graph_anns_val.pkl
```

### **Step 4: 生成锚点**

```bash
cd /data/roadnet_data/cg/CGNet/mmdetection3d/

# 运行锚点生成（路径已修复，会自动找anns/目录）
python tools/generate_anchors.py \
    --data-root data/nuscenes \
    --num-clusters 50 \
    --degree 3 \
    --output work_dirs/kmeans_anchors.pth \
    --visualize
```

**预期输出**:
```
Loading from: data/nuscenes/anns/nuscenes_infos_temporal_train.pkl
Found 28130 samples
Processing centerlines: 100%
收集到 ~300,000 条有效中心线
聚类完成！
✅ 锚点已保存到: work_dirs/kmeans_anchors.pth
```

### **Step 5: 开始训练**

```bash
# 8卡训练
bash tools/train_diff_cgnet.sh \
    configs/diff_cgnet/diff_cgnet_r50_nusc.py 8
```

---

## 📋 路径配置总结

### **锚点生成脚本**

```python
# tools/generate_anchors.py (已修复)

优先查找:
✅ data/nuscenes/anns/nuscenes_infos_temporal_train.pkl

备选路径:
1. data/nuscenes/nuscenes_infos_temporal_train.pkl
2. data/nuscenes/nuscenes_centerline_infos_train.pkl
3. data/nuscenes/nuscenes_infos_train.pkl

结论: ✅ 已适配服务器结构
```

### **配置文件**

```python
# configs/diff_cgnet/diff_cgnet_r50_nusc.py

ann_root = 'data/nuscenes/anns/'  ✅ 正确
data_root = 'data/nuscenes'        ✅ 正确

train.ann_file = ann_root + 'nuscenes_infos_temporal_train.pkl'  ✅
val.ann_file = ann_root + 'nuscenes_infos_temporal_val.pkl'      ✅

结论: ✅ 配置文件路径正确
```

---

## ✅ 总结

**路径调整**: ✅ **已完成**

**修改内容**:
- generate_anchors.py: 优先查找`anns/`目录
- 支持多个备选路径
- 兼容本地和服务器

**无需其他调整**:
- ✅ 配置文件路径已正确
- ✅ 数据已预处理
- ✅ 直接可用

**在服务器上执行**:
```bash
cd /data/roadnet_data/cg/CGNet/mmdetection3d/
python tools/generate_anchors.py --data-root data/nuscenes --visualize
```

**路径会自动找到正确的文件！** 🎯
