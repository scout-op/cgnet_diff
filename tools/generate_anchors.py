import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
import argparse
import matplotlib.pyplot as plt
import sys
import os
from math import factorial


def comb(n, k):
    """组合数 C(n,k)"""
    return factorial(n) // (factorial(k) * factorial(n - k))


def fit_bezier(points, n_control=4):
    """
    将点序列拟合为贝塞尔曲线控制点
    
    Args:
        points: np.ndarray, shape (n_points, 2)
        n_control: int, 控制点数量（默认4，即三次贝塞尔）
    
    Returns:
        control_points: np.ndarray, shape (n_control, 2)
    """
    if len(points) < 10:
        points = np.linspace(points[0], points[-1], num=10)
    
    n_points = len(points)
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * \
                      np.power(1 - t[i], n_control - 1 - j) * \
                      np.power(t[i], j)
    
    A_BE = A[:, 1:-1]
    points_BE = points - np.stack([
        (A[:, 0] * points[0][0] + A[:, -1] * points[-1][0]),
        (A[:, 0] * points[0][1] + A[:, -1] * points[-1][1])
    ]).T
    
    try:
        conts = np.linalg.lstsq(A_BE, points_BE, rcond=None)
    except:
        raise Exception("Bezier fitting failed! Check if points are valid.")
    
    res = conts[0]
    fin_res = np.r_[[points[0]], res, [points[-1]]]
    
    return fin_res


def bezier_interpolate_np(ctrl_points, num_points=50):
    """
    贝塞尔曲线插值（numpy版本）
    """
    n_control = len(ctrl_points)
    degree = n_control - 1
    
    t = np.linspace(0, 1, num_points)
    points = np.zeros((num_points, 2))
    
    for i in range(n_control):
        coef = comb(degree, i) * np.power(1 - t, degree - i) * np.power(t, i)
        points += coef[:, np.newaxis] * ctrl_points[i]
    
    return points


def parse_args():
    parser = argparse.ArgumentParser(description='Generate K-Means anchors for centerlines')
    parser.add_argument('--data-root', default='data/nuscenes', help='NuScenes data root')
    parser.add_argument('--num-clusters', type=int, default=50, help='Number of anchor clusters')
    parser.add_argument('--degree', type=int, default=3, help='Bezier curve degree (3 for cubic)')
    parser.add_argument('--output', default='work_dirs/kmeans_anchors.pth', help='Output path')
    parser.add_argument('--visualize', action='store_true', help='Visualize anchors')
    return parser.parse_args()


def generate_kmeans_anchors(data_root, num_clusters=50, degree=3):
    """
    生成K-Means聚类锚点
    
    Args:
        data_root: str, 数据根目录
        num_clusters: int, 聚类数量
        degree: int, 贝塞尔曲线阶数
    
    Returns:
        anchors: np.ndarray, shape (num_clusters, degree+1, 2)
    """
    print("="*60)
    print("Step 1: 加载训练集中心线...")
    print("="*60)
    
    all_bezier_ctrl = []
    
    pkl_path = os.path.join(data_root, 'nuscenes_centerline_infos_train.pkl')
    
    if not os.path.exists(pkl_path):
        print(f"Warning: {pkl_path} not found!")
        print("Trying alternative paths...")
        
        pkl_path = os.path.join(data_root, 'anns', 'nuscenes_infos_temporal_train.pkl')
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(data_root, 'nuscenes_infos_temporal_train.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(data_root, 'nuscenes_infos_train.pkl')
    
    print(f"Loading from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'infos' in data:
        infos = data['infos']
    else:
        infos = data
    
    print(f"Found {len(infos)} samples")
    
    print(f"\n示例数据结构:")
    if len(infos) > 0:
        sample = infos[0]
        print(f"  Keys: {list(sample.keys())[:10]}")
        if 'gt_bboxes_3d' in sample:
            print(f"  gt_bboxes_3d type: {type(sample['gt_bboxes_3d'])}")
            if hasattr(sample['gt_bboxes_3d'], '__dict__'):
                print(f"  gt_bboxes_3d attrs: {list(vars(sample['gt_bboxes_3d']).keys())[:5]}")
    
    for info in tqdm(infos, desc="Processing centerlines"):
        if 'gt_bboxes_3d' not in info:
            continue
        
        centerlines = info['gt_bboxes_3d']
        
        if hasattr(centerlines, 'instance_list'):
            centerlines = centerlines.instance_list
        elif hasattr(centerlines, 'fixed_num'):
            centerlines = centerlines.fixed_num
        elif isinstance(centerlines, dict):
            if 'instance_list' in centerlines:
                centerlines = centerlines['instance_list']
            elif 'vectors' in centerlines:
                centerlines = centerlines['vectors']
        elif isinstance(centerlines, (list, tuple)):
            pass
        else:
            continue
        
        if not isinstance(centerlines, (list, tuple)):
            continue
        
        for line in centerlines:
            if isinstance(line, torch.Tensor):
                line = line.cpu().numpy()
            elif not isinstance(line, np.ndarray):
                continue
            
            if len(line) < 2:
                continue
            
            try:
                ctrl = fit_bezier(line, n_control=degree+1)
                all_bezier_ctrl.append(ctrl.flatten())
            except Exception as e:
                continue
    
    if len(all_bezier_ctrl) == 0:
        print("\n❗ 错误: 未收集到任何中心线！")
        print("\n请检查:")
        print("  1. 数据文件路径是否正确")
        print("  2. gt_bboxes_3d的数据格式")
        print("  3. 是否包含中心线数据")
        print("\n试用其他数据文件:")
        print("  python tools/generate_anchors.py --data-root data/nuscenes --use-centerline-file")
        raise ValueError("未找到任何中心线数据")
    
    all_bezier_ctrl = np.array(all_bezier_ctrl)
    print(f"\n收集到 {len(all_bezier_ctrl)} 条有效中心线")
    
    print("\n" + "="*60)
    print("Step 2: K-Means聚类...")
    print("="*60)
    
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=1
    )
    kmeans.fit(all_bezier_ctrl)
    
    anchors = kmeans.cluster_centers_
    anchors = anchors.reshape(num_clusters, degree+1, 2)
    
    print(f"\n聚类完成！Inertia: {kmeans.inertia_:.2f}")
    
    return anchors


def visualize_anchors(anchors, save_path='work_dirs/anchors_visualization.png'):
    """
    可视化锚点
    
    Args:
        anchors: np.ndarray, shape (num_clusters, 4, 2)
        save_path: str, 保存路径
    """
    print("\n" + "="*60)
    print("Step 4: 可视化锚点...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        ax.set_xlim(-15, 15)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Anchor Visualization {idx+1}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        start_idx = idx * len(anchors) // 4
        end_idx = (idx + 1) * len(anchors) // 4
        
        for anchor in anchors[start_idx:end_idx]:
            points = bezier_interpolate_np(anchor, num_points=50)
            
            ax.plot(points[:, 0], points[:, 1], 'b-', alpha=0.5, linewidth=1)
            ax.plot(anchor[:, 0], anchor[:, 1], 'ro', markersize=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存到: {save_path}")
    plt.close()


def main():
    args = parse_args()
    
    anchors = generate_kmeans_anchors(
        data_root=args.data_root,
        num_clusters=args.num_clusters,
        degree=args.degree
    )
    
    print("\n" + "="*60)
    print("Step 3: 保存锚点...")
    print("="*60)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    torch.save({
        'anchors': torch.from_numpy(anchors).float(),
        'num_clusters': args.num_clusters,
        'degree': args.degree,
        'description': 'K-Means clustered Bezier anchors for centerline diffusion'
    }, args.output)
    
    print(f"✅ 锚点已保存到: {args.output}")
    print(f"   - 数量: {args.num_clusters}")
    print(f"   - 阶数: {args.degree}")
    print(f"   - 形状: ({args.num_clusters}, {args.degree+1}, 2)")
    
    if args.visualize:
        visualize_anchors(anchors)
    
    print("\n" + "="*60)
    print("✅ 锚点生成完成！")
    print("="*60)


if __name__ == '__main__':
    main()
