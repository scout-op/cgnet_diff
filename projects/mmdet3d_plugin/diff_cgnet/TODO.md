# ✅ 实施清单

## Phase 0: 预处理与验证 ✅

- [x] 创建项目结构
- [x] 实现几何工具 (`modules/utils.py`)
- [x] 实现扩散模块 (`modules/diffusion.py`)
- [x] 实现匈牙利匹配器 (`modules/matcher.py`)
- [x] 实现贝塞尔Deformable Attention (`modules/sampler.py`)
- [x] 实现Teacher Forcing (`hooks/teacher_forcing.py`)
- [x] 创建锚点生成工具 (`tools/generate_anchors.py`)
- [x] 创建单元测试 (`tests/test_modules.py`)

## Phase 1: 核心模块实现 🔄

### 立即执行（按顺序）:

#### 1. 运行单元测试 ⏰ 现在
```bash
cd /home/subobo/ro/e2e/CGNet
python projects/mmdet3d_plugin/diff_cgnet/tests/test_modules.py
```

#### 2. 生成锚点 ⏰ 今天
```bash
python tools/generate_anchors.py \
    --data-root data/nuscenes \
    --num-clusters 50 \
    --visualize
```

#### 3. 实现扩散检测头 ⏰ 明天
- [ ] `dense_heads/diff_head.py`
  - [ ] 时间嵌入
  - [ ] 去噪网络
  - [ ] Self-Conditioning
  - [ ] Centerline Renewal

#### 4. 实现主检测器 ⏰ 后天
- [ ] `detectors/diff_cgnet.py`
  - [ ] 继承CGNet
  - [ ] 集成扩散模块
  - [ ] forward_train
  - [ ] forward_test

#### 5. 配置文件 ⏰ 第4天
- [ ] `configs/diff_cgnet/diff_cgnet_r50_nusc.py`

#### 6. Sanity Check ⏰ 第5天
- [ ] 过拟合1个样本
- [ ] Loss < 0.01
- [ ] 可视化验证

## Phase 2: 优化与训练 📅 Week 2

- [ ] 小规模训练（10%数据）
- [ ] 调整超参数
- [ ] 全量训练（24 epoch）
- [ ] 评估指标

## Phase 3: 实验对比 📅 Week 3-4

- [ ] 消融实验
- [ ] 对比实验
- [ ] 可视化

## 🎯 当前状态

**已完成**: 基础模块实现
**下一步**: 运行单元测试
**预计时间**: 3-4周完成全部

## 📝 注意事项

1. ⚠️ 每次修改后运行单元测试
2. ⚠️ Sanity Check必须通过才能全量训练
3. ⚠️ 注意坐标系转换
4. ⚠️ 检查梯度是否正常
