import argparse
import pickle
import sys
import os

sys.path.insert(0, 'projects/mmdet3d_plugin')

from diff_cgnet.evaluation.centerline_metrics import evaluate_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DiffCGNet predictions')
    parser.add_argument('--results', required=True, help='prediction results (.pkl)')
    parser.add_argument('--gt-file', required=True, help='ground truth file (.pkl)')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[0.5, 1.0, 1.5],
                       help='evaluation thresholds')
    parser.add_argument('--output', default='evaluation_results.txt',
                       help='output file for results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("🎯 DiffCGNet Evaluation")
    print("="*70)
    
    print(f"\nLoading predictions: {args.results}")
    with open(args.results, 'rb') as f:
        pred_results = pickle.load(f)
    
    print(f"Loading ground truth: {args.gt_file}")
    with open(args.gt_file, 'rb') as f:
        gt_data = pickle.load(f)
    
    print(f"\nNumber of samples: {len(pred_results)}")
    print(f"Evaluation thresholds: {args.thresholds}")
    
    print("\n" + "-"*70)
    print("Computing metrics...")
    print("-"*70)
    
    metrics = evaluate_all_metrics(pred_results, gt_data, args.thresholds)
    
    print("\n" + "="*70)
    print("📊 Evaluation Results")
    print("="*70)
    
    results_text = []
    results_text.append("="*70)
    results_text.append("DiffCGNet Evaluation Results")
    results_text.append("="*70)
    results_text.append("")
    
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():15s}: {value:.4f}")
        results_text.append(f"{metric_name.upper():15s}: {value:.4f}")
    
    print("\n" + "="*70)
    
    with open(args.output, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"\n✅ Results saved to: {args.output}")
    
    return metrics


if __name__ == '__main__':
    metrics = main()
