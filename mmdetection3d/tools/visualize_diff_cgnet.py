import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import pickle
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize DiffCGNet predictions')
    parser.add_argument('--results', required=True, help='prediction results file (.pkl)')
    parser.add_argument('--gt-file', required=True, help='ground truth file (.pkl)')
    parser.add_argument('--output-dir', default='work_dirs/visualizations', 
                       help='output directory')
    parser.add_argument('--num-samples', type=int, default=50, 
                       help='number of samples to visualize')
    parser.add_argument('--show-topology', action='store_true',
                       help='visualize topology connections')
    parser.add_argument('--save-video', action='store_true',
                       help='generate video')
    return parser.parse_args()


def draw_centerlines(ax, centerlines, color='blue', linewidth=2, alpha=0.8, label=None):
    """
    绘制中心线
    
    Args:
        ax: matplotlib axis
        centerlines: List[np.ndarray], shape (num_points, 2)
        color: str
        linewidth: float
        alpha: float
        label: str
    """
    for idx, line in enumerate(centerlines):
        if len(line) < 2:
            continue
        
        ax.plot(line[:, 0], line[:, 1], 
               color=color, linewidth=linewidth, alpha=alpha,
               label=label if idx == 0 else None)


def draw_topology(ax, centerlines, topology, color='red', alpha=0.5):
    """
    绘制拓扑连接
    
    Args:
        ax: matplotlib axis
        centerlines: List[np.ndarray]
        topology: np.ndarray, [N, N]
        color: str
        alpha: float
    """
    N = len(centerlines)
    
    for i in range(N):
        for j in range(N):
            if topology[i, j] > 0.5:
                if len(centerlines[i]) > 0 and len(centerlines[j]) > 0:
                    start = centerlines[i][-1]
                    end = centerlines[j][0]
                    
                    ax.annotate('',
                               xy=end, xytext=start,
                               arrowprops=dict(
                                   arrowstyle='->',
                                   color=color,
                                   alpha=alpha,
                                   lw=2
                               ))


def visualize_sample(pred_result, gt_data, output_path, show_topology=True):
    """
    可视化单个样本
    
    Args:
        pred_result: dict, 预测结果
        gt_data: dict, GT数据
        output_path: str, 保存路径
        show_topology: bool, 是否显示拓扑
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    pc_range = [-15, -30, 15, 30]
    
    for ax in axes:
        ax.set_xlim(pc_range[0], pc_range[2])
        ax.set_ylim(pc_range[1], pc_range[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    draw_centerlines(axes[0], gt_data['centerlines'], 
                    color='green', linewidth=2.5, label='GT')
    
    if show_topology and 'topology' in gt_data:
        draw_topology(axes[0], gt_data['centerlines'], 
                     gt_data['topology'], color='darkgreen')
    
    axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
    draw_centerlines(axes[1], pred_result['centerlines'],
                    color='blue', linewidth=2.5, label='Pred')
    
    if show_topology and 'topology' in pred_result:
        draw_topology(axes[1], pred_result['centerlines'],
                     pred_result['topology'], color='darkblue')
    
    axes[2].set_title('Comparison', fontsize=14, fontweight='bold')
    draw_centerlines(axes[2], gt_data['centerlines'],
                    color='green', linewidth=2, alpha=0.6, label='GT')
    draw_centerlines(axes[2], pred_result['centerlines'],
                    color='blue', linewidth=2, alpha=0.6, label='Pred')
    
    for ax in axes:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading predictions from: {args.results}")
    with open(args.results, 'rb') as f:
        pred_results = pickle.load(f)
    
    print(f"Loading ground truth from: {args.gt_file}")
    with open(args.gt_file, 'rb') as f:
        gt_data = pickle.load(f)
    
    num_samples = min(args.num_samples, len(pred_results))
    
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    for idx in range(num_samples):
        output_path = os.path.join(args.output_dir, f'sample_{idx:04d}.png')
        
        visualize_sample(
            pred_results[idx],
            gt_data[idx],
            output_path,
            show_topology=args.show_topology
        )
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{num_samples}")
    
    print(f"\n✅ Visualizations saved to: {args.output_dir}")
    
    if args.save_video:
        print("\nGenerating video...")
        generate_video(args.output_dir)


def generate_video(image_dir, output_path='visualization.mp4', fps=5):
    """生成视频"""
    import glob
    
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    if len(images) == 0:
        print("No images found!")
        return
    
    frame = cv2.imread(images[0])
    h, w, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"✅ Video saved to: {output_path}")


if __name__ == '__main__':
    main()
