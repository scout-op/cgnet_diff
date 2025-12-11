#!/usr/bin/env python
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'projects'))

print("=" * 60)
print("测试导入 DiffCGNet")
print("=" * 60)

try:
    print("\n1. 导入 mmdet3d_plugin...")
    import mmdet3d_plugin
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. 导入 diff_cgnet 模块...")
    from mmdet3d_plugin import diff_cgnet
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. 导入 DiffCGNet 类...")
    from mmdet3d_plugin.diff_cgnet import DiffCGNet
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n4. 检查 DETECTORS 注册表...")
    from mmdet.models import DETECTORS
    print(f"   注册的检测器: {list(DETECTORS.module_dict.keys())[:10]}...")
    if 'DiffCGNet' in DETECTORS.module_dict:
        print("   ✓ DiffCGNet 已注册")
    else:
        print("   ✗ DiffCGNet 未注册")
        print(f"   可用的检测器: {[k for k in DETECTORS.module_dict.keys() if 'Diff' in k or 'CG' in k]}")
except Exception as e:
    print(f"   ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
