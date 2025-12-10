import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
import argparse
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from projects.mmdet3d_plugin.diff_cgnet.modules.utils import fit_bezier


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
    
    pkl_path = os.path.join(data_root, 'nuscenes_infos_temporal_train.pkl')
    
    if not os.path.exists(pkl_path):
        print(f"Warning: {pkl_path} not found!")
        print("Trying alternative path...")
        pkl_path = os.path.join(data_root, 'nuscenes_infos_train.pkl')
    
    print(f"Loading from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'infos' in data:
        infos = data['infos']
    else:
        infos = data
    
    print(f"Found {len(infos)} samples")
    
    for info in tqdm(infos, desc="Processing centerlines"):
        if 'gt_bboxes_3d' not in info:
            continue
        
        centerlines = info['gt_bboxes_3d']
        
        if hasattr(centerlines, 'instance_list'):
            centerlines = centerlines.instance_list
        
        for line in centerlines:
            if isinstance(line, torch.Tensor):
                line = line.cpu().numpy()
            
            if len(line) < 2:
                continue
            
            try:
                ctrl = fit_bezier(line, n_control=degree+1)
                all_bezier_ctrl.append(ctrl.flatten())
            except:
                continue
    
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
    
    from projects.mmdet3d_plugin.diff_cgnet.modules.utils import bezier_interpolate
    
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
            ctrl_tensor = torch.from_numpy(anchor).float().unsqueeze(0)
            points = bezier_interpolate(ctrl_tensor, num_points=50)
            points = points.squeeze(0).numpy()
            
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
