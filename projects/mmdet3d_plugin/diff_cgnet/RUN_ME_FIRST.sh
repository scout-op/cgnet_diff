#!/bin/bash

echo "=========================================="
echo "🚀 DiffCGNet 快速验证脚本"
echo "=========================================="
echo ""

cd /home/subobo/ro/e2e/CGNet

echo "Step 1: 运行单元测试..."
echo "------------------------------------------"
python projects/mmdet3d_plugin/diff_cgnet/tests/test_modules.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 单元测试通过！"
    echo ""
else
    echo ""
    echo "❌ 单元测试失败！请先修复错误。"
    exit 1
fi

echo "Step 2: 检查数据..."
echo "------------------------------------------"
if [ -f "data/nuscenes/nuscenes_infos_temporal_train.pkl" ]; then
    echo "✅ 找到数据文件: nuscenes_infos_temporal_train.pkl"
elif [ -f "data/nuscenes/nuscenes_infos_train.pkl" ]; then
    echo "✅ 找到数据文件: nuscenes_infos_train.pkl"
else
    echo "❌ 未找到数据文件！"
    echo "   请确保数据在: data/nuscenes/"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 基础验证完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 生成锚点:"
echo "   python tools/generate_anchors.py --visualize"
echo ""
echo "2. 查看可视化:"
echo "   打开 work_dirs/anchors_visualization.png"
echo ""
echo "3. 实现检测头:"
echo "   编辑 projects/mmdet3d_plugin/diff_cgnet/dense_heads/diff_head.py"
echo ""
echo "=========================================="
