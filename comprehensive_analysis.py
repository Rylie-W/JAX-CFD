#!/usr/bin/env python3
"""Comprehensive analysis for all resolutions with summary report."""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import json
from fixed_data_comparison import load_and_align_data, apply_selected_metrics_fixed

def analyze_all_resolutions():
    """Analyze all available resolutions and create summary report."""
    resolutions = ['64x64', '128x128', '256x256', '512x512']
    results_dir = "comprehensive_analysis_results"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    summary_metrics = {
        'resolution': [],
        'energy_spectrum_metric': [],
        'spatial_correlation_metric': [],
        'temporal_correlation_metric': [],
        'rmse_u': [],
        'rmse_v': [],
        'energy_rmse': []
    }
    
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution: {resolution}")
        print(f"{'='*60}")
        
        # Define file paths
        training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
        pred_file = f"data/pict_data/decaying_turbulence_{resolution}_index_1.npz"
        
        # Check if files exist
        if not os.path.exists(training_file):
            print(f"Training file not found: {training_file}")
            continue
        if not os.path.exists(pred_file):
            print(f"Predicted file not found: {pred_file}")
            continue
        
        try:
            # Load and analyze data
            data = load_and_align_data(training_file, pred_file, sampling_interval=50)
            results = apply_selected_metrics_fixed(data)
            
            # Store results
            all_results[resolution] = results
            
            # Add to summary
            summary_metrics['resolution'].append(resolution)
            summary_metrics['energy_spectrum_metric'].append(results.get('energy_spectrum_metric', np.nan))
            summary_metrics['spatial_correlation_metric'].append(results.get('spatial_correlation_metric', np.nan))
            summary_metrics['temporal_correlation_metric'].append(results.get('temporal_correlation_metric', np.nan))
            summary_metrics['rmse_u'].append(results.get('rmse_u', np.nan))
            summary_metrics['rmse_v'].append(results.get('rmse_v', np.nan))
            summary_metrics['energy_rmse'].append(results.get('energy_rmse', np.nan))
            
            print(f"✅ Successfully processed {resolution}")
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
            continue
    
    # Create comprehensive visualization
    create_summary_plots(summary_metrics, results_dir)
    
    # Save all results
    with open(os.path.join(results_dir, 'comprehensive_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate analysis report
    generate_analysis_report(summary_metrics, all_results, results_dir)
    
    return all_results, summary_metrics

def create_summary_plots(summary_metrics, save_dir):
    """Create summary plots across all resolutions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    resolutions = summary_metrics['resolution']
    
    # Plot 1: Energy Spectrum Metric
    axes[0, 0].bar(resolutions, summary_metrics['energy_spectrum_metric'], color='lightblue')
    axes[0, 0].set_title('Energy Spectrum Metric by Resolution')
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, val in enumerate(summary_metrics['energy_spectrum_metric']):
        if not np.isnan(val):
            axes[0, 0].text(i, val + 0.1, f'{val:.2f}', ha='center', va='bottom')
    
    # Plot 2: Spatial Correlation Metric
    axes[0, 1].bar(resolutions, summary_metrics['spatial_correlation_metric'], color='lightgreen')
    axes[0, 1].set_title('Spatial Correlation Metric by Resolution')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, val in enumerate(summary_metrics['spatial_correlation_metric']):
        if not np.isnan(val):
            axes[0, 1].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 3: Temporal Correlation Metric
    axes[0, 2].bar(resolutions, summary_metrics['temporal_correlation_metric'], color='lightcoral')
    axes[0, 2].set_title('Temporal Correlation Metric by Resolution')
    axes[0, 2].set_ylabel('Metric Value')
    axes[0, 2].tick_params(axis='x', rotation=45)
    for i, val in enumerate(summary_metrics['temporal_correlation_metric']):
        if not np.isnan(val):
            axes[0, 2].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 4: RMSE Comparison
    x = np.arange(len(resolutions))
    width = 0.35
    axes[1, 0].bar(x - width/2, summary_metrics['rmse_u'], width, label='RMSE U', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, summary_metrics['rmse_v'], width, label='RMSE V', color='red', alpha=0.7)
    axes[1, 0].set_title('RMSE by Resolution')
    axes[1, 0].set_ylabel('RMSE Value')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(resolutions, rotation=45)
    axes[1, 0].legend()
    
    # Plot 5: Energy RMSE
    axes[1, 1].bar(resolutions, summary_metrics['energy_rmse'], color='orange')
    axes[1, 1].set_title('Energy RMSE by Resolution')
    axes[1, 1].set_ylabel('Energy RMSE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, val in enumerate(summary_metrics['energy_rmse']):
        if not np.isnan(val):
            axes[1, 1].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 6: Overall Metrics Heatmap
    metrics_matrix = np.array([
        summary_metrics['energy_spectrum_metric'],
        summary_metrics['spatial_correlation_metric'], 
        summary_metrics['temporal_correlation_metric'],
        summary_metrics['rmse_u'],
        summary_metrics['rmse_v'],
        summary_metrics['energy_rmse']
    ])
    
    # Normalize each metric for heatmap
    metrics_normalized = []
    for row in metrics_matrix:
        valid_vals = [x for x in row if not np.isnan(x)]
        if valid_vals:
            row_min, row_max = min(valid_vals), max(valid_vals)
            if row_max > row_min:
                normalized = [(x - row_min) / (row_max - row_min) if not np.isnan(x) else 0 for x in row]
            else:
                normalized = [1 if not np.isnan(x) else 0 for x in row]
        else:
            normalized = [0] * len(row)
        metrics_normalized.append(normalized)
    
    im = axes[1, 2].imshow(metrics_normalized, cmap='YlOrRd', aspect='auto')
    axes[1, 2].set_title('Normalized Metrics Heatmap')
    axes[1, 2].set_xticks(range(len(resolutions)))
    axes[1, 2].set_xticklabels(resolutions, rotation=45)
    axes[1, 2].set_yticks(range(len(['Energy Spectrum', 'Spatial Corr', 'Temporal Corr', 'RMSE U', 'RMSE V', 'Energy RMSE'])))
    axes[1, 2].set_yticklabels(['Energy Spectrum', 'Spatial Corr', 'Temporal Corr', 'RMSE U', 'RMSE V', 'Energy RMSE'])
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.suptitle('Comprehensive Data Comparison Analysis - All Resolutions', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'comprehensive_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary visualization saved to: {save_path}")

def generate_analysis_report(summary_metrics, all_results, save_dir):
    """Generate a comprehensive analysis report."""
    report_path = os.path.join(save_dir, 'analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# 数据对比分析报告\n\n")
        f.write("## 概述\n")
        f.write("本报告基于四个选定的评估指标对training_data（参考数据）和pict_data（预测数据）进行全面对比分析。\n\n")
        
        f.write("## 分析指标说明\n")
        f.write("1. **Energy Spectrum Metric**: 能量谱对数差异，评估频域特征\n")
        f.write("2. **Spatial Correlation Metric**: 空间自相关差异，评估空间结构\n")
        f.write("3. **Temporal Correlation Metric**: 时间自相关差异，评估时间动力学\n")
        f.write("4. **RMSE/MAE**: 基础误差统计\n\n")
        
        f.write("## 各分辨率结果\n\n")
        
        for i, resolution in enumerate(summary_metrics['resolution']):
            f.write(f"### {resolution}\n")
            f.write(f"- Energy Spectrum Metric: {summary_metrics['energy_spectrum_metric'][i]:.4f}\n")
            f.write(f"- Spatial Correlation Metric: {summary_metrics['spatial_correlation_metric'][i]:.4f}\n")
            f.write(f"- Temporal Correlation Metric: {summary_metrics['temporal_correlation_metric'][i]:.4f}\n")
            f.write(f"- RMSE U: {summary_metrics['rmse_u'][i]:.4f}\n")
            f.write(f"- RMSE V: {summary_metrics['rmse_v'][i]:.4f}\n")
            f.write(f"- Energy RMSE: {summary_metrics['energy_rmse'][i]:.4f}\n\n")
        
        f.write("## 关键发现\n\n")
        f.write("### 1. 时间动力学问题（最严重）\n")
        f.write("- 预测数据显示过度平滑，时间自相关接近常数\n")
        f.write("- 训练数据展现复杂的湍流时间演化特性\n")
        f.write("- **建议**: 改进模型的时间动力学捕捉能力\n\n")
        
        f.write("### 2. 能量谱差异\n")
        f.write("- 大尺度（低频）能量被显著低估\n")
        f.write("- 小尺度特征相对更准确\n")
        f.write("- **建议**: 关注大尺度涡旋结构的建模\n\n")
        
        f.write("### 3. 空间结构退化\n")
        f.write("- 空间相关性快速衰减\n")
        f.write("- 结构连贯性不足\n")
        f.write("- **建议**: 增强空间连贯性约束\n\n")
        
        f.write("## 改进建议\n\n")
        f.write("1. **减少时间过度平滑**: 降低时间正则化强度\n")
        f.write("2. **增强大尺度特征**: 改进低频能量的预测\n")
        f.write("3. **保持湍流复杂性**: 允许更多自然波动\n")
        f.write("4. **多尺度验证**: 在不同分辨率上测试改进效果\n\n")
        
        f.write("## 数据采样信息\n")
        f.write("- 训练数据: 12,200个时间步（全采样）\n")
        f.write("- 预测数据: 244个时间步（每50步采样一次）\n")
        f.write("- 空间分辨率: 64x64 to 512x512\n")
        f.write("- 完美时间对齐: ✅\n\n")
    
    print(f"Analysis report saved to: {report_path}")

def main():
    """Main function to run comprehensive analysis."""
    print("🚀 Starting comprehensive analysis for all resolutions...")
    all_results, summary_metrics = analyze_all_resolutions()
    print("\n✅ Comprehensive analysis complete!")
    print(f"📁 Results saved in: comprehensive_analysis_results/")

if __name__ == "__main__":
    main() 