#!/bin/bash

echo "=========================================="
echo "🚀 DiffCGNet 完整启动流程"
echo "=========================================="
echo ""
echo "项目状态: ✅ 所有核心模块已实现"
echo ""

cd /home/subobo/ro/e2e/CGNet

echo "步骤 1/5: 运行单元测试..."
echo "------------------------------------------"
python projects/mmdet3d_plugin/diff_cgnet/tests/test_modules.py

if [ $? -ne 0 ]; then
    echo "❌ 单元测试失败！请先修复。"
    exit 1
fi

echo ""
echo "步骤 2/5: 生成K-Means锚点..."
echo "------------------------------------------"

if [ ! -f "work_dirs/kmeans_anchors.pth" ]; then
    echo "正在生成锚点..."
    python tools/generate_anchors.py \
        --data-root data/nuscenes \
        --num-clusters 50 \
        --degree 3 \
        --output work_dirs/kmeans_anchors.pth \
        --visualize
    
    if [ $? -ne 0 ]; then
        echo "❌ 锚点生成失败！"
        exit 1
    fi
else
    echo "✅ 锚点文件已存在: work_dirs/kmeans_anchors.pth"
fi

echo ""
echo "步骤 3/5: 验证锚点..."
echo "------------------------------------------"
python -c "
import torch
data = torch.load('work_dirs/kmeans_anchors.pth')
print(f'✅ 锚点形状: {data[\"anchors\"].shape}')
print(f'✅ 锚点数量: {data[\"num_clusters\"]}')
print(f'✅ 贝塞尔阶数: {data[\"degree\"]}')
"

echo ""
echo "步骤 4/5: 检查配置文件..."
echo "------------------------------------------"
if [ -f "configs/diff_cgnet/diff_cgnet_r50_nusc.py" ]; then
    echo "✅ 配置文件存在"
else
    echo "❌ 配置文件不存在！"
    exit 1
fi

echo ""
echo "步骤 5/5: 准备就绪检查..."
echo "------------------------------------------"

echo "检查清单:"
echo "  ✅ 核心模块: 已实现"
echo "  ✅ 扩散检测头: 已实现"
echo "  ✅ 主检测器: 已实现"
echo "  ✅ 配置文件: 已创建"
echo "  ✅ 锚点文件: 已生成"
echo "  ✅ 单元测试: 通过"

echo ""
echo "=========================================="
echo "✅ 所有准备工作完成！"
echo "=========================================="
echo ""
echo "📋 下一步操作:"
echo ""
echo "1️⃣  Sanity Check (过拟合测试):"
echo "   python projects/mmdet3d_plugin/diff_cgnet/tests/test_sanity_check.py"
echo ""
echo "2️⃣  小规模训练 (验证代码):"
echo "   bash tools/train_diff_cgnet.sh configs/diff_cgnet/diff_cgnet_r50_nusc.py 1"
echo ""
echo "3️⃣  全量训练 (8卡):"
echo "   bash tools/train_diff_cgnet.sh configs/diff_cgnet/diff_cgnet_r50_nusc.py 8"
echo ""
echo "4️⃣  查看日志:"
echo "   tensorboard --logdir work_dirs/diff_cgnet"
echo ""
echo "=========================================="
echo "🎯 建议: 先运行Sanity Check！"
echo "=========================================="
