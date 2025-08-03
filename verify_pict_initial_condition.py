#!/usr/bin/env python3
"""Verify if PICT initial condition matches training data first step after warmup."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

def verify_initial_condition_hypothesis():
    """Verify if PICT starts from training data's post-warmup state."""
    
    print("="*60)
    print("验证假设：PICT的初始条件是training warmup后的第一步")
    print("="*60)
    
    # Load data
    training_file = "data/training_data/decaying_turbulence_v2_64x64_index_1.npz"
    pict_file = "data/pict_data/decaying_turbulence_64x64_index_1.npz"
    
    training_data = np.load(training_file)
    pict_data = np.load(pict_file)
    
    # Extract first timesteps
    training_u_0 = training_data['u'][0]  # First step after warmup
    training_v_0 = training_data['v'][0]
    pict_u_0 = pict_data['u'][0]          # Initial condition for PICT
    pict_v_0 = pict_data['v'][0]
    
    print(f"数据形状:")
    print(f"  Training第0步: u={training_u_0.shape}, v={training_v_0.shape}")
    print(f"  PICT第0步: u={pict_u_0.shape}, v={pict_v_0.shape}")
    
    # Check if they are identical or very close
    is_identical_u = np.allclose(training_u_0, pict_u_0, rtol=1e-10, atol=1e-10)
    is_identical_v = np.allclose(training_v_0, pict_v_0, rtol=1e-10, atol=1e-10)
    
    # Try different tolerance levels
    tolerances = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    
    print(f"\n精确度检查:")
    print(f"  完全相同 (tolerance=1e-10): U={is_identical_u}, V={is_identical_v}")
    
    print(f"\n不同容差下的相同性检查:")
    print("  容差      | U相同 | V相同")
    print("  ----------|-------|-------")
    for tol in tolerances:
        u_close = np.allclose(training_u_0, pict_u_0, rtol=tol, atol=tol)
        v_close = np.allclose(training_v_0, pict_v_0, rtol=tol, atol=tol)
        print(f"  {tol:8.0e}  |  {str(u_close):4s}  |  {str(v_close):4s}")
    
    # Compute detailed statistics
    diff_u = training_u_0 - pict_u_0
    diff_v = training_v_0 - pict_v_0
    
    print(f"\n详细差异统计:")
    print(f"  U差异: min={np.min(diff_u):.2e}, max={np.max(diff_u):.2e}")
    print(f"        mean={np.mean(diff_u):.2e}, std={np.std(diff_u):.2e}")
    print(f"        RMSE={np.sqrt(np.mean(diff_u**2)):.2e}")
    
    print(f"  V差异: min={np.min(diff_v):.2e}, max={np.max(diff_v):.2e}")
    print(f"        mean={np.mean(diff_v):.2e}, std={np.std(diff_v):.2e}")
    print(f"        RMSE={np.sqrt(np.mean(diff_v**2)):.2e}")
    
    # Check correlation
    corr_u = np.corrcoef(training_u_0.flatten(), pict_u_0.flatten())[0, 1]
    corr_v = np.corrcoef(training_v_0.flatten(), pict_v_0.flatten())[0, 1]
    
    print(f"\n相关性分析:")
    print(f"  U相关系数: {corr_u:.10f}")
    print(f"  V相关系数: {corr_v:.10f}")
    
    # Check field statistics
    print(f"\n场统计比较:")
    print("  变量 | 数据集   | 均值      | 标准差    | 最小值    | 最大值")
    print("  -----|----------|-----------|-----------|-----------|----------")
    print(f"   U   | Training | {np.mean(training_u_0):8.5f} | {np.std(training_u_0):8.5f} | {np.min(training_u_0):8.5f} | {np.max(training_u_0):8.5f}")
    print(f"   U   | PICT     | {np.mean(pict_u_0):8.5f} | {np.std(pict_u_0):8.5f} | {np.min(pict_u_0):8.5f} | {np.max(pict_u_0):8.5f}")
    print(f"   V   | Training | {np.mean(training_v_0):8.5f} | {np.std(training_v_0):8.5f} | {np.min(training_v_0):8.5f} | {np.max(training_v_0):8.5f}")
    print(f"   V   | PICT     | {np.mean(pict_v_0):8.5f} | {np.std(pict_v_0):8.5f} | {np.min(pict_v_0):8.5f} | {np.max(pict_v_0):8.5f}")
    
    # Look at multiple timesteps to see evolution pattern
    print(f"\n时间演化模式分析:")
    print("检查PICT的后续步骤是否与training的对应步骤匹配...")
    
    n_check = min(5, pict_data['u'].shape[0])
    print(f"\n前{n_check}步的RMSE比较:")
    print("  步骤 | U-RMSE   | V-RMSE   | U-相关   | V-相关")
    print("  -----|----------|----------|----------|----------")
    
    for i in range(n_check):
        if i < training_data['u'].shape[0]:
            train_u = training_data['u'][i]
            train_v = training_data['v'][i]
            pict_u = pict_data['u'][i]
            pict_v = pict_data['v'][i]
            
            rmse_u = np.sqrt(np.mean((train_u - pict_u)**2))
            rmse_v = np.sqrt(np.mean((train_v - pict_v)**2))
            corr_u = np.corrcoef(train_u.flatten(), pict_u.flatten())[0, 1]
            corr_v = np.corrcoef(train_v.flatten(), pict_v.flatten())[0, 1]
            
            print(f"   {i:2d}  | {rmse_u:8.5f} | {rmse_v:8.5f} | {corr_u:8.5f} | {corr_v:8.5f}")
    
    # Create visualization
    create_comparison_visualization(training_u_0, training_v_0, pict_u_0, pict_v_0, diff_u, diff_v)
    
    # Final assessment
    print(f"\n{'='*60}")
    print("结论评估:")
    
    # Determine if hypothesis is likely correct
    max_diff_u = np.max(np.abs(diff_u))
    max_diff_v = np.max(np.abs(diff_v))
    
    if max_diff_u < 1e-10 and max_diff_v < 1e-10:
        print("✅ 假设正确：PICT初始条件与training第0步完全相同")
    elif max_diff_u < 1e-6 and max_diff_v < 1e-6 and corr_u > 0.999 and corr_v > 0.999:
        print("✅ 假设很可能正确：PICT初始条件与training第0步非常接近")
        print("   差异可能来自数值精度或微小的实现差异")
    elif corr_u > 0.95 and corr_v > 0.95:
        print("⚠️ 假设部分正确：PICT与training有高相关性但存在显著差异")
        print("   可能使用了相同的初始化方法但参数略有不同")
    else:
        print("❌ 假设错误：PICT初始条件与training第0步显著不同")
        print("   它们可能来自不同的初始化过程")
    
    print(f"{'='*60}")

def create_comparison_visualization(training_u, training_v, pict_u, pict_v, diff_u, diff_v):
    """Create visualization comparing the initial conditions."""
    
    output_dir = Path("initial_condition_verification")
    output_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Common color scales
    u_vmin = min(np.min(training_u), np.min(pict_u))
    u_vmax = max(np.max(training_u), np.max(pict_u))
    v_vmin = min(np.min(training_v), np.min(pict_v))
    v_vmax = max(np.max(training_v), np.max(pict_v))
    
    # Training U
    plt.subplot(3, 4, 1)
    im = plt.imshow(training_u, cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
    plt.title('Training U (t=0)\n(warmup后第1步)')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    # PICT U
    plt.subplot(3, 4, 2)
    im = plt.imshow(pict_u, cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
    plt.title('PICT U (t=0)\n(初始条件)')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    # U difference
    plt.subplot(3, 4, 3)
    diff_u_max = max(abs(np.min(diff_u)), abs(np.max(diff_u)))
    if diff_u_max > 0:
        im = plt.imshow(diff_u, cmap='RdBu_r', vmin=-diff_u_max, vmax=diff_u_max)
    else:
        im = plt.imshow(diff_u, cmap='RdBu_r')
    plt.title(f'U差异\nRMSE={np.sqrt(np.mean(diff_u**2)):.2e}')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    # U correlation scatter
    plt.subplot(3, 4, 4)
    plt.scatter(pict_u.flatten()[::20], training_u.flatten()[::20], alpha=0.6, s=1)
    plt.xlabel('PICT U')
    plt.ylabel('Training U')
    plt.title('U相关性散点图')
    corr_u = np.corrcoef(training_u.flatten(), pict_u.flatten())[0, 1]
    plt.text(0.05, 0.95, f'r = {corr_u:.6f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Training V
    plt.subplot(3, 4, 5)
    im = plt.imshow(training_v, cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
    plt.title('Training V (t=0)\n(warmup后第1步)')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    # PICT V
    plt.subplot(3, 4, 6)
    im = plt.imshow(pict_v, cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
    plt.title('PICT V (t=0)\n(初始条件)')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    # V difference
    plt.subplot(3, 4, 7)
    diff_v_max = max(abs(np.min(diff_v)), abs(np.max(diff_v)))
    if diff_v_max > 0:
        im = plt.imshow(diff_v, cmap='RdBu_r', vmin=-diff_v_max, vmax=diff_v_max)
    else:
        im = plt.imshow(diff_v, cmap='RdBu_r')
    plt.title(f'V差异\nRMSE={np.sqrt(np.mean(diff_v**2)):.2e}')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    # V correlation scatter
    plt.subplot(3, 4, 8)
    plt.scatter(pict_v.flatten()[::20], training_v.flatten()[::20], alpha=0.6, s=1)
    plt.xlabel('PICT V')
    plt.ylabel('Training V')
    plt.title('V相关性散点图')
    corr_v = np.corrcoef(training_v.flatten(), pict_v.flatten())[0, 1]
    plt.text(0.05, 0.95, f'r = {corr_v:.6f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Velocity magnitude comparison
    plt.subplot(3, 4, 9)
    training_mag = np.sqrt(training_u**2 + training_v**2)
    pict_mag = np.sqrt(pict_u**2 + pict_v**2)
    im = plt.imshow(training_mag, cmap='viridis')
    plt.title('Training速度模长')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    im = plt.imshow(pict_mag, cmap='viridis')
    plt.title('PICT速度模长')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    mag_diff = training_mag - pict_mag
    mag_diff_max = max(abs(np.min(mag_diff)), abs(np.max(mag_diff)))
    if mag_diff_max > 0:
        im = plt.imshow(mag_diff, cmap='RdBu_r', vmin=-mag_diff_max, vmax=mag_diff_max)
    else:
        im = plt.imshow(mag_diff, cmap='RdBu_r')
    plt.title('速度模长差异')
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')
    
    # Summary statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    summary_text = f"""初始条件验证总结

U分量:
• RMSE: {np.sqrt(np.mean(diff_u**2)):.2e}
• 最大差异: {np.max(np.abs(diff_u)):.2e}
• 相关系数: {corr_u:.6f}

V分量:
• RMSE: {np.sqrt(np.mean(diff_v**2)):.2e}
• 最大差异: {np.max(np.abs(diff_v)):.2e}
• 相关系数: {corr_v:.6f}

结论:
{"完全相同" if np.max(np.abs(diff_u)) < 1e-10 and np.max(np.abs(diff_v)) < 1e-10 
 else "非常接近" if np.max(np.abs(diff_u)) < 1e-6 and np.max(np.abs(diff_v)) < 1e-6
 else "高度相关" if corr_u > 0.95 and corr_v > 0.95
 else "存在差异"}
"""
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    output_file = output_dir / 'initial_condition_verification.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 可视化保存到: {output_file}")
    plt.close()

if __name__ == "__main__":
    verify_initial_condition_hypothesis()