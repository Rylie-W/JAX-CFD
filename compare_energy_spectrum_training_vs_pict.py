#!/usr/bin/env python3
"""
对比training_data和pict_data的energy spectrum
不包含warmup数据，只分析simulation阶段
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import Dict, List, Tuple
import seaborn as sns

def compute_energy_spectrum_2d(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算2D速度场的能量谱
    
    Args:
        u: x方向速度分量 (time, height, width)
        v: y方向速度分量 (time, height, width)
    
    Returns:
        k: 波数数组
        E_k: 能量谱数组
    """
    nt, nx, ny = u.shape
    
    # 计算时间平均能量谱
    energy_spectra = []
    
    for t in range(nt):
        # 计算速度场的FFT
        u_fft = np.fft.fft2(u[t])
        v_fft = np.fft.fft2(v[t])
        
        # 计算能量密度
        energy_density = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2)
        
        # 获取波数
        kx = np.fft.fftfreq(nx, 1.0) * nx
        ky = np.fft.fftfreq(ny, 1.0) * ny
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # 径向平均
        k_max = int(min(nx, ny) // 3)  # 避免混叠
        k_bins = np.arange(1, k_max + 1)  # 从k=1开始
        energy_spectrum = np.zeros(len(k_bins))
        
        for i, k in enumerate(k_bins):
            mask = (k_magnitude >= k - 0.5) & (k_magnitude < k + 0.5)
            if np.sum(mask) > 0:
                energy_spectrum[i] = np.mean(energy_density[mask])
        
        energy_spectra.append(energy_spectrum)
    
    # 时间平均
    avg_energy_spectrum = np.mean(energy_spectra, axis=0)
    
    return k_bins, avg_energy_spectrum

def load_training_data(data_dir: str, max_files: int = 3, max_frames_per_file: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    内存高效地加载training data（JAX模拟数据）
    """
    print(f"🔍 内存高效加载training data从: {data_dir}")
    
    # 查找所有文件并排序
    pattern = os.path.join(data_dir, "kolmogorov_256x256_step_*_index_1.npz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise ValueError(f"未找到匹配的文件: {pattern}")
    
    # 大幅减少文件数量以避免内存问题
    files = files[:max_files]
    print(f"  找到 {len(glob.glob(pattern))} 个文件，使用前 {len(files)} 个 (内存优化)")
    
    all_u = []
    all_v = []
    
    for i, file_path in enumerate(files):
        print(f"  正在加载 ({i+1}/{len(files)}): {os.path.basename(file_path)}")
        try:
            data = np.load(file_path)
            u = data['u']  # (time, x, y)
            v = data['v']  # (time, x, y)
            
            print(f"    原始形状: {u.shape}")
            
            # 采样减少时间帧数
            if u.shape[0] > max_frames_per_file:
                step = u.shape[0] // max_frames_per_file
                indices = range(0, u.shape[0], step)[:max_frames_per_file]
                u = u[indices]
                v = v[indices]
                print(f"    采样后形状: {u.shape}")
            
            all_u.append(u)
            all_v.append(v)
            
        except Exception as e:
            print(f"    ⚠️ 加载失败: {e}")
            continue
    
    if not all_u:
        raise ValueError("没有成功加载任何training data文件")
    
    # 合并所有数据
    u_combined = np.concatenate(all_u, axis=0)
    v_combined = np.concatenate(all_v, axis=0)
    
    print(f"  ✅ Training data加载完成: {u_combined.shape}")
    return u_combined, v_combined

def load_pict_data(data_dir: str, max_files: int = 3, max_frames_per_file: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    内存高效地加载PICT data，过滤掉warmup文件
    """
    print(f"🔍 内存高效加载PICT data从: {data_dir}")
    
    # 查找simulation阶段的文件（不包含warmup）
    pattern = os.path.join(data_dir, "turbulence_step*_256x256_index_1.npz")
    files = glob.glob(pattern)
    
    # 过滤掉可能的warmup文件
    simulation_files = []
    for file_path in files:
        basename = os.path.basename(file_path)
        # 确保不是warmup文件
        if 'warmup' not in basename and 'final' not in basename:
            simulation_files.append(file_path)
    
    simulation_files = sorted(simulation_files)
    
    if not simulation_files:
        raise ValueError(f"未找到simulation阶段的PICT文件")
    
    # 大幅减少文件数量以避免内存问题
    simulation_files = simulation_files[:max_files]
    print(f"  找到 {len(glob.glob(pattern))} 个simulation文件，使用前 {len(simulation_files)} 个 (内存优化)")
    
    all_u = []
    all_v = []
    
    for i, file_path in enumerate(simulation_files):
        print(f"  正在加载 ({i+1}/{len(simulation_files)}): {os.path.basename(file_path)}")
        try:
            data = np.load(file_path)
            u = data['u']  # (time, x, y)
            v = data['v']  # (time, x, y)
            
            print(f"    原始形状: {u.shape}")
            
            # 采样减少时间帧数
            if u.shape[0] > max_frames_per_file:
                step = u.shape[0] // max_frames_per_file
                indices = range(0, u.shape[0], step)[:max_frames_per_file]
                u = u[indices]
                v = v[indices]
                print(f"    采样后形状: {u.shape}")
            
            all_u.append(u)
            all_v.append(v)
            
        except Exception as e:
            print(f"    ⚠️ 加载失败: {e}")
            continue
    
    if not all_u:
        raise ValueError("没有成功加载任何PICT数据文件")
    
    # 合并所有数据
    u_combined = np.concatenate(all_u, axis=0)
    v_combined = np.concatenate(all_v, axis=0)
    
    print(f"  ✅ PICT data加载完成: {u_combined.shape}")
    return u_combined, v_combined

def plot_energy_spectrum_comparison(k_training, E_training, k_pict, E_pict, save_path: str = None):
    """
    绘制energy spectrum对比图
    """
    plt.figure(figsize=(12, 8))
    
    # 主图：对数尺度
    plt.subplot(2, 2, 1)
    plt.loglog(k_training, E_training, 'b-', linewidth=2, label='Training Data (JAX)')
    plt.loglog(k_pict, E_pict, 'r-', linewidth=2, label='PICT Data')
    
    # 添加理论参考线
    k_ref = np.logspace(0, np.log10(max(k_training[-1], k_pict[-1])), 50)
    # Kolmogorov -5/3律
    E_ref = k_ref**(-5/3)
    E_ref = E_ref * np.mean(E_training[5:15]) / np.mean(E_ref[5:15])  # 归一化
    plt.loglog(k_ref, E_ref, 'k--', alpha=0.7, label='$k^{-5/3}$ (Kolmogorov)')
    
    plt.xlabel('Wavenumber $k$')
    plt.ylabel('Energy Spectrum $E(k)$')
    plt.title('Energy Spectrum Comparison (Log-Log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 线性尺度对比
    plt.subplot(2, 2, 2)
    plt.plot(k_training, E_training, 'b-', linewidth=2, label='Training Data (JAX)')
    plt.plot(k_pict, E_pict, 'r-', linewidth=2, label='PICT Data')
    plt.xlabel('Wavenumber $k$')
    plt.ylabel('Energy Spectrum $E(k)$')
    plt.title('Energy Spectrum Comparison (Linear)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 比值图
    plt.subplot(2, 2, 3)
    # 插值到相同的波数网格
    k_common = np.arange(1, min(len(k_training), len(k_pict)) + 1)
    E_training_interp = np.interp(k_common, k_training, E_training)
    E_pict_interp = np.interp(k_common, k_pict, E_pict)
    
    ratio = E_training_interp / (E_pict_interp + 1e-10)  # 避免除零
    plt.semilogx(k_common, ratio, 'g-', linewidth=2)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.7)
    plt.xlabel('Wavenumber $k$')
    plt.ylabel('Ratio: Training/PICT')
    plt.title('Energy Spectrum Ratio')
    plt.grid(True, alpha=0.3)
    
    # 统计信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 计算一些统计量
    total_energy_training = np.sum(E_training)
    total_energy_pict = np.sum(E_pict)
    
    peak_k_training = k_training[np.argmax(E_training)]
    peak_k_pict = k_pict[np.argmax(E_pict)]
    
    stats_text = f"""
    统计对比:
    
    总能量:
    • Training: {total_energy_training:.3e}
    • PICT: {total_energy_pict:.3e}
    • 比值: {total_energy_training/total_energy_pict:.3f}
    
    峰值波数:
    • Training: k = {peak_k_training}
    • PICT: k = {peak_k_pict}
    
    数据点数:
    • Training: {len(k_training)} 个波数
    • PICT: {len(k_pict)} 个波数
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 图表已保存: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🔬 Training Data vs PICT Data Energy Spectrum Comparison")
    print("=" * 60)
    
    # 数据路径
    training_data_dir = "/Users/yiwei/Projects/Python/thesis/JAX-CFD/data/training_data/256"
    pict_data_dir = "/Users/yiwei/Projects/Python/thesis/JAX-CFD/data/pict_data/kf_256"
    
    # 检查路径
    if not os.path.exists(training_data_dir):
        raise ValueError(f"Training data目录不存在: {training_data_dir}")
    if not os.path.exists(pict_data_dir):
        raise ValueError(f"PICT data目录不存在: {pict_data_dir}")
    
    try:
        # 加载training data
        print("\n📁 第1步: 加载Training Data")
        u_training, v_training = load_training_data(training_data_dir, max_files=34)
        
        # 加载PICT data  
        print("\n📁 第2步: 加载PICT Data")
        u_pict, v_pict = load_pict_data(pict_data_dir, max_files=34)
        
        # 计算energy spectrum
        print("\n📊 第3步: 计算Energy Spectrum")
        print("  计算Training data能量谱...")
        k_training, E_training = compute_energy_spectrum_2d(u_training, v_training)
        
        print("  计算PICT data能量谱...")
        k_pict, E_pict = compute_energy_spectrum_2d(u_pict, v_pict)
        
        # 绘制对比图
        print("\n📈 第4步: 生成对比图表")
        save_path = "energy_spectrum_training_vs_pict_256x256.png"
        plot_energy_spectrum_comparison(k_training, E_training, k_pict, E_pict, save_path)
        
        # 输出数值对比
        print("\n📋 数值对比结果:")
        print(f"  Training总能量: {np.sum(E_training):.3e}")
        print(f"  PICT总能量: {np.sum(E_pict):.3e}")
        print(f"  能量比值 (Training/PICT): {np.sum(E_training)/np.sum(E_pict):.3f}")
        
        # 保存数据
        results = {
            'k_training': k_training,
            'E_training': E_training,
            'k_pict': k_pict,
            'E_pict': E_pict,
            'total_energy_training': np.sum(E_training),
            'total_energy_pict': np.sum(E_pict),
            'energy_ratio': np.sum(E_training)/np.sum(E_pict)
        }
        
        results_file = "energy_spectrum_comparison_results_256x256.npz"
        np.savez(results_file, **results)
        print(f"💾 数值结果已保存: {results_file}")
        
        print("\n✅ 分析完成!")
        
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
