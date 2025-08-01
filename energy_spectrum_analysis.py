#!/usr/bin/env python3
"""Detailed energy spectrum analysis and interpretation."""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def create_energy_spectrum_explanation():
    """Create detailed explanation of energy spectrum analysis."""
    
    # Load the results from 64x64 analysis
    with open('fixed_comparison_results/fixed_results_64x64.json', 'r') as f:
        results = json.load(f)
    
    training_spectrum = np.array(results['training_spectrum_avg'])
    pred_spectrum = np.array(results['pred_spectrum_avg'])
    
    # Create wavenumber array
    k_values = np.arange(len(training_spectrum))
    
    # Create comprehensive energy spectrum analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Raw energy spectra comparison
    ax1 = axes[0, 0]
    valid_training = training_spectrum > 0
    valid_pred = pred_spectrum > 0
    valid_both = valid_training & valid_pred
    
    if np.sum(valid_both) > 1:
        ax1.loglog(k_values[valid_training], training_spectrum[valid_training], 
                  'b-', linewidth=3, label='Training (Reference)', marker='o')
        ax1.loglog(k_values[valid_pred], pred_spectrum[valid_pred], 
                  'r--', linewidth=3, label='Prediction', marker='s')
        
        # Add Kolmogorov -5/3 reference line
        k_ref = k_values[5:15]
        E_ref = training_spectrum[5] * (k_ref/k_values[5])**(-5/3)
        ax1.loglog(k_ref, E_ref, 'k:', linewidth=2, alpha=0.7, label='k^(-5/3) 理论')
    
    ax1.set_xlabel('波数 k')
    ax1.set_ylabel('能量 E(k)')
    ax1.set_title('能量谱对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('大尺度涡旋\n(能量注入)', xy=(2, 1e6), xytext=(1, 1e7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, color='green', weight='bold')
    ax1.annotate('小尺度涡旋\n(能量耗散)', xy=(25, 1e2), xytext=(20, 1e4),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                fontsize=12, color='orange', weight='bold')
    
    # Plot 2: Energy ratio (Prediction/Training)
    ax2 = axes[0, 1]
    ratio = pred_spectrum / (training_spectrum + 1e-10)  # Avoid division by zero
    valid_ratio = valid_both & (ratio > 0) & (ratio < 1000)
    
    if np.sum(valid_ratio) > 1:
        ax2.semilogx(k_values[valid_ratio], ratio[valid_ratio], 'g-', linewidth=2, marker='d')
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='完美匹配')
        ax2.fill_between(k_values[valid_ratio], 0.5, 2.0, alpha=0.2, color='gray', label='可接受范围')
    
    ax2.set_xlabel('波数 k')
    ax2.set_ylabel('能量比值 (预测/训练)')
    ax2.set_title('相对能量比较')
    ax2.set_ylim([0.01, 100])
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy distribution by scale
    ax3 = axes[1, 0]
    
    # Divide spectrum into scale ranges
    large_scale = slice(1, 8)    # k = 1-7 (大尺度)
    medium_scale = slice(8, 20)  # k = 8-19 (中尺度)
    small_scale = slice(20, None) # k = 20+ (小尺度)
    
    scales = ['大尺度\n(k=1-7)', '中尺度\n(k=8-19)', '小尺度\n(k=20+)']
    training_energy = [
        np.sum(training_spectrum[large_scale]),
        np.sum(training_spectrum[medium_scale]), 
        np.sum(training_spectrum[small_scale])
    ]
    pred_energy = [
        np.sum(pred_spectrum[large_scale]),
        np.sum(pred_spectrum[medium_scale]),
        np.sum(pred_spectrum[small_scale])
    ]
    
    x = np.arange(len(scales))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, training_energy, width, label='训练数据', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, pred_energy, width, label='预测数据', color='red', alpha=0.7)
    
    ax3.set_xlabel('尺度范围')
    ax3.set_ylabel('总能量')
    ax3.set_title('不同尺度的能量分布')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scales)
    ax3.legend()
    ax3.set_yscale('log')
    
    # Add value labels on bars
    for bar1, bar2, val1, val2 in zip(bars1, bars2, training_energy, pred_energy):
        ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() * 1.1,
                f'{val1:.0e}', ha='center', va='bottom', rotation=0, fontsize=10)
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() * 1.1,
                f'{val2:.0e}', ha='center', va='bottom', rotation=0, fontsize=10)
    
    # Plot 4: Physical interpretation
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate key metrics
    total_energy_training = np.sum(training_spectrum[valid_training])
    total_energy_pred = np.sum(pred_spectrum[valid_pred])
    energy_ratio_total = total_energy_pred / total_energy_training
    
    large_scale_ratio = np.sum(pred_spectrum[large_scale]) / np.sum(training_spectrum[large_scale])
    small_scale_ratio = np.sum(pred_spectrum[small_scale]) / np.sum(training_spectrum[small_scale])
    
    # Create text summary
    summary_text = f"""
📊 能量谱分析总结

🔍 总体发现:
• 总能量比值: {energy_ratio_total:.3f}
• 预测数据总能量明显偏低

📏 尺度分析:
• 大尺度比值: {large_scale_ratio:.3f}
• 小尺度比值: {small_scale_ratio:.3f}

⚠️  关键问题:
1. 大尺度能量严重不足
   → 主要涡旋结构丢失
   
2. 能量分布失衡
   → 物理过程不正确

💡 物理解释:
• 训练数据: 真实湍流能量级联
• 预测数据: 过度耗散，缺乏大涡旋

🎯 改进方向:
• 增强大尺度结构保持
• 减少数值耗散
• 改进边界条件处理
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('能量谱详细分析 - 64x64分辨率', fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save the plot
    save_path = 'energy_spectrum_detailed_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"详细能量谱分析图保存到: {save_path}")
    
    return {
        'total_energy_ratio': energy_ratio_total,
        'large_scale_ratio': large_scale_ratio,
        'small_scale_ratio': small_scale_ratio,
        'training_energy': training_energy,
        'pred_energy': pred_energy
    }

def explain_energy_spectrum_theory():
    """Create theoretical explanation of energy spectrum."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Theoretical energy spectrum
    k = np.logspace(0, 2, 100)
    
    # Idealized energy spectrum with different regions
    E_theory = np.zeros_like(k)
    
    # Energy injection range (k < 5)
    injection_mask = k < 5
    E_theory[injection_mask] = 1000 * k[injection_mask]**2
    
    # Inertial range (5 <= k <= 50)
    inertial_mask = (k >= 5) & (k <= 50)
    E_theory[inertial_mask] = 10000 * k[inertial_mask]**(-5/3)
    
    # Dissipation range (k > 50)
    dissipation_mask = k > 50
    E_theory[dissipation_mask] = 1e8 * k[dissipation_mask]**(-7)
    
    ax1 = axes[0, 0]
    ax1.loglog(k, E_theory, 'b-', linewidth=3, label='理论能量谱')
    
    # Add region annotations
    ax1.fill_between(k[injection_mask], 1, E_theory[injection_mask], alpha=0.3, color='green', label='能量注入区')
    ax1.fill_between(k[inertial_mask], 1, E_theory[inertial_mask], alpha=0.3, color='yellow', label='惯性区 (k^-5/3)')
    ax1.fill_between(k[dissipation_mask], 1, E_theory[dissipation_mask], alpha=0.3, color='red', label='耗散区')
    
    ax1.set_xlabel('波数 k (1/长度)', fontsize=12)
    ax1.set_ylabel('能量密度 E(k)', fontsize=12)
    ax1.set_title('理论湍流能量谱', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Physical interpretation
    ax2 = axes[0, 1]
    ax2.axis('off')
    
    theory_text = """
🌊 湍流能量谱理论

📚 Kolmogorov理论 (1941):
能量在不同尺度间级联传递

🔄 能量流动:
大涡旋 → 中等涡旋 → 小涡旋 → 热能

📏 三个区域:

1️⃣ 能量注入区 (低k):
   • 大尺度涡旋结构
   • 主要能量来源
   • E(k) ∝ k²

2️⃣ 惯性区 (中k):
   • 能量级联传递
   • Kolmogorov -5/3 定律
   • E(k) ∝ k^(-5/3)

3️⃣ 耗散区 (高k):
   • 粘性效应主导
   • 能量转化为热
   • E(k) ∝ k^(-7) 或更陡
    """
    
    ax2.text(0.05, 0.95, theory_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Plot 3: What energy spectrum tells us
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    diagnostic_text = """
🔬 能量谱诊断功能

✅ 好的预测应该:
• 保持正确的能量分布
• 遵循物理定律 (k^-5/3)
• 各尺度比例合理

❌ 常见问题及征象:

1. 数值耗散过强:
   → 高频能量过快衰减
   → 谱在惯性区过陡

2. 大尺度结构丢失:
   → 低频能量不足
   → 主要涡旋消失

3. 非物理行为:
   → 谱形状不合理
   → 出现能量堆积

🎯 我们的发现:
预测数据显示严重的大尺度
能量不足，表明模型在保持
主要涡旋结构方面存在问题
    """
    
    ax3.text(0.05, 0.95, diagnostic_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
    
    # Plot 4: Improvement strategies
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    improvement_text = """
🚀 基于能量谱的改进策略

📈 提升大尺度能量:
• 减少低频数值耗散
• 改进边界条件
• 增大计算域大小

⚖️ 平衡能量分布:
• 调整时间步长
• 优化空间离散格式
• 控制人工粘性

🎯 模型优化:
• 保持物理约束
• 增强大涡旋保持
• 改进亚格子尺度模型

📊 验证指标:
• 总能量守恒
• 谱斜率接近-5/3
• 各尺度比例合理

💡 实际应用:
能量谱分析帮助我们识别
模型的物理合理性，是
湍流仿真质量评估的
黄金标准之一
    """
    
    ax4.text(0.05, 0.95, improvement_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('湍流能量谱：理论、诊断与改进', fontsize=16, weight='bold')
    plt.tight_layout()
    
    save_path = 'energy_spectrum_theory.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"能量谱理论解释图保存到: {save_path}")

def main():
    """Generate comprehensive energy spectrum analysis."""
    print("🌊 生成能量谱详细分析...")
    
    # Create detailed analysis of our results
    metrics = create_energy_spectrum_explanation()
    
    # Create theoretical background
    explain_energy_spectrum_theory()
    
    print("\n📊 分析完成！生成了以下文件:")
    print("• energy_spectrum_detailed_analysis.png - 我们数据的详细分析")
    print("• energy_spectrum_theory.png - 理论背景和改进建议")
    
    print(f"\n🔍 关键发现:")
    print(f"• 总能量比值: {metrics['total_energy_ratio']:.3f} (预测/训练)")
    print(f"• 大尺度能量比值: {metrics['large_scale_ratio']:.3f}")
    print(f"• 小尺度能量比值: {metrics['small_scale_ratio']:.3f}")
    
    print(f"\n💡 结论: 预测模型严重低估了大尺度涡旋的能量，需要改进大尺度结构的保持能力。")

if __name__ == "__main__":
    main() 