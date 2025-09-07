#!/usr/bin/env python3
"""
多分辨率空间涡度相关性分析
支持warmup和simulation阶段的区分
以2048为参考，比较所有其他分辨率
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
import glob
import zipfile
from typing import Dict, List, Tuple
import matplotlib.patches as patches

def find_data_files_with_phases(data_path: str, resolution: int, max_steps: int = None) -> Dict[str, List[Tuple[int, str]]]:
    """
    查找数据文件，区分warmup和simulation阶段
    
    Warmup文件：
    - turbulence_warmup_step{step}_{resolution}x{resolution}_index_1.npz
    - turbulence_warmup_final_{resolution}x{resolution}_index_1.npz (最后900多步)
    
    Simulation文件：turbulence_step{step}_{resolution}x{resolution}_index_1.npz (无warmup)
    
    Returns:
        {'warmup': [(step, file_path), ...], 'simulation': [(step, file_path), ...]}
    """
    # 查找所有可能的文件 (包括warmup_final文件，处理不同命名格式)
    step_pattern1 = f"{data_path}/turbulence*step*_{resolution}x{resolution}_index_1.npz"
    step_pattern2 = f"{data_path}/turbulence_{resolution}_*step*_{resolution}x{resolution}_index_1.npz"
    
    # warmup_final文件的不同格式
    final_pattern1 = f"{data_path}/turbulence_warmup_final_{resolution}x{resolution}_index_1.npz"
    final_pattern2 = f"{data_path}/turbulence_{resolution}_warmup_final_{resolution}x{resolution}_index_1.npz"
    
    step_files1 = glob.glob(step_pattern1)
    step_files2 = glob.glob(step_pattern2)
    final_files1 = glob.glob(final_pattern1)
    final_files2 = glob.glob(final_pattern2)
    
    all_files = step_files1 + step_files2 + final_files1 + final_files2
    
    phases = {'warmup': [], 'simulation': []}
    
    # 分类处理文件
    for file in all_files:
        try:
            basename = os.path.basename(file)
            
            if 'warmup' in basename:
                if 'warmup_final' in basename:
                    # Warmup最后阶段文件的格式:
                    # turbulence_warmup_final_{resolution}x{resolution}_index_1.npz
                    # turbulence_{resolution}_warmup_final_{resolution}x{resolution}_index_1.npz
                    step_num = 10900  # 代表最后的warmup步数
                    
                    if max_steps is None or step_num <= max_steps:
                        phases['warmup'].append((step_num, file))
                        
                elif 'warmup_step' in basename:
                    # Warmup阶段文件的可能格式:
                    # turbulence_warmup_step{step}_{resolution}x{resolution}_index_1.npz
                    # turbulence_{resolution}_warmup_step{step}_{resolution}x{resolution}_index_1.npz
                    step_part = basename.split('warmup_step')[1].split('_')[0]
                    step_num = int(step_part)
                    
                    if max_steps is None or step_num <= max_steps:
                        phases['warmup'].append((step_num, file))
                    
            else:
                # Simulation阶段文件: 可能的格式
                # turbulence_step{step}_{resolution}x{resolution}_index_1.npz
                # turbulence_{resolution}_step{step}_{resolution}x{resolution}_index_1.npz
                if '_step' in basename:
                    # 找到最后一个 '_step' 的位置
                    step_parts = basename.split('_step')
                    if len(step_parts) >= 2:
                        step_part = step_parts[-1].split('_')[0]  # 取最后一个step后的数字
                        step_num = int(step_part)
                        
                        if max_steps is None or step_num <= max_steps:
                            phases['simulation'].append((step_num, file))
                    
        except (IndexError, ValueError):
            print(f"Warning: Could not extract step number from {basename}")
            continue
    
    # 排序
    phases['warmup'].sort(key=lambda x: x[0])
    phases['simulation'].sort(key=lambda x: x[0])
    
    print(f"Resolution {resolution}: found {len(phases['warmup'])} warmup files, {len(phases['simulation'])} simulation files")
    
    # 打印一些示例文件名用于验证
    if phases['warmup']:
        print(f"  Warmup example: {os.path.basename(phases['warmup'][0][1])}")
    if phases['simulation']:
        print(f"  Simulation example: {os.path.basename(phases['simulation'][0][1])}")
    
    return phases

def load_velocity_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载速度数据，包含错误处理"""
    try:
        data = np.load(file_path)
        u = data['u']  # (time, x, y)
        v = data['v']  # (time, x, y)
        return u, v
    except (zipfile.BadZipFile, OSError, IOError) as e:
        print(f"❌ Error loading file {os.path.basename(file_path)}: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error loading file {os.path.basename(file_path)}: {e}")
        raise

def downsample_to_target_resolution(u: np.ndarray, v: np.ndarray, 
                                  original_res: int, target_res: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用区域平均下采样"""
    if original_res == target_res:
        return u, v
    
    if original_res % target_res != 0:
        raise ValueError(f"Cannot downsample {original_res} to {target_res}: not divisible")
    
    downsample_factor = original_res // target_res
    
    # 使用区域平均下采样
    u_reshaped = u.reshape(u.shape[0], target_res, downsample_factor, target_res, downsample_factor)
    v_reshaped = v.reshape(v.shape[0], target_res, downsample_factor, target_res, downsample_factor)
    
    u_down = u_reshaped.mean(axis=(2, 4))
    v_down = v_reshaped.mean(axis=(2, 4))
    
    return u_down, v_down

def compute_vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """计算涡度场"""
    dvdx = np.gradient(v, axis=-1)  # ∂v/∂x
    dudy = np.gradient(u, axis=-2)  # ∂u/∂y  
    vorticity = dvdx - dudy
    return vorticity

def compute_spatial_correlation_timeseries(vort_ref: np.ndarray, vort_comp: np.ndarray) -> List[float]:
    """计算空间相关性时间序列"""
    time_steps = min(vort_ref.shape[0], vort_comp.shape[0])
    correlations = []
    
    for t in range(time_steps):
        vort_ref_flat = vort_ref[t].flatten()
        vort_comp_flat = vort_comp[t].flatten()
        
        # 计算空间相关性
        if np.std(vort_ref_flat) > 1e-10 and np.std(vort_comp_flat) > 1e-10:
            corr = np.corrcoef(vort_ref_flat, vort_comp_flat)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
            else:
                correlations.append(0.0)
        else:
            correlations.append(0.0)
    
    return correlations

def analyze_multiresolution_correlation(base_dir: str, reference_resolution: int, 
                                      comparison_resolutions: List[int], max_steps: int = None) -> Dict:
    """分析多分辨率空间涡度相关性"""
    print(f"🔬 Multi-Resolution Spatial Vorticity Correlation Analysis")
    print(f"Base directory: {base_dir}")
    print(f"Reference: {reference_resolution}x{reference_resolution}")
    print(f"Comparisons: {comparison_resolutions}")
    
    # 查找参考数据文件
    ref_path = f"{base_dir}/{reference_resolution}"
    ref_files = find_data_files_with_phases(ref_path, reference_resolution, max_steps)
    
    # 查找比较数据文件
    comp_files_dict = {}
    for res in comparison_resolutions:
        comp_path = f"{base_dir}/{res}"
        if os.path.exists(comp_path):
            comp_files_dict[res] = find_data_files_with_phases(comp_path, res, max_steps)
        else:
            print(f"Warning: Path not found for resolution {res}: {comp_path}")
    
    # 准备结果结构
    results = {
        'reference_resolution': reference_resolution,
        'comparison_resolutions': comparison_resolutions,
        'phases': {
            'warmup': {'steps': [], 'correlations': {res: [] for res in comparison_resolutions}},
            'simulation': {'steps': [], 'correlations': {res: [] for res in comparison_resolutions}}
        }
    }
    
    # 处理每个阶段
    for phase in ['warmup', 'simulation']:
        print(f"\n📊 Processing {phase} phase...")
        
        # 找到共同的步数
        ref_steps = {step: path for step, path in ref_files[phase]}
        common_steps_dict = {res: set() for res in comparison_resolutions}
        
        for res in comparison_resolutions:
            if res in comp_files_dict:
                comp_steps = {step: path for step, path in comp_files_dict[res][phase]}
                common_steps_dict[res] = set(ref_steps.keys()) & set(comp_steps.keys())
        
        # 获取所有分辨率都有的共同步数
        if common_steps_dict:
            all_common_steps = set.intersection(*[steps for steps in common_steps_dict.values() if steps])
            all_common_steps = sorted(all_common_steps)
        else:
            all_common_steps = []
        
        if not all_common_steps:
            print(f"  No common steps found for {phase} phase")
            continue
        
        print(f"  Found {len(all_common_steps)} common steps: {all_common_steps}")
        results['phases'][phase]['steps'] = all_common_steps
        
        # 分析每个步数
        for step in all_common_steps:
            print(f"  Processing {phase} step {step}...")
            
            # 加载参考数据
            ref_file = ref_steps[step]
            try:
                ref_u, ref_v = load_velocity_data(ref_file)
            except Exception as e:
                print(f"    ⚠️  Skipping step {step} - Reference file corrupted: {os.path.basename(ref_file)}")
                continue
            
            # 为每个比较分辨率计算相关性
            for res in comparison_resolutions:
                if res in comp_files_dict and step in {s: p for s, p in comp_files_dict[res][phase]}:
                    comp_files = {s: p for s, p in comp_files_dict[res][phase]}
                    comp_file = comp_files[step]
                    
                    # 加载比较数据
                    try:
                        comp_u, comp_v = load_velocity_data(comp_file)
                    except Exception as e:
                        print(f"    ⚠️  Skipping resolution {res} at step {step} - Comparison file corrupted: {os.path.basename(comp_file)}")
                        continue
                    
                    # 下采样到目标分辨率（使用较小的分辨率）
                    target_res = min(reference_resolution, res)
                    
                    # 下采样参考数据
                    ref_u_down, ref_v_down = downsample_to_target_resolution(
                        ref_u, ref_v, reference_resolution, target_res)
                    ref_vorticity = compute_vorticity(ref_u_down, ref_v_down)
                    
                    # 下采样比较数据
                    comp_u_down, comp_v_down = downsample_to_target_resolution(
                        comp_u, comp_v, res, target_res)
                    comp_vorticity = compute_vorticity(comp_u_down, comp_v_down)
                    
                    # 计算空间相关性
                    correlations = compute_spatial_correlation_timeseries(ref_vorticity, comp_vorticity)
                    avg_correlation = np.mean(correlations)
                    
                    results['phases'][phase]['correlations'][res].append(avg_correlation)
                    
                    print(f"    {res}x{res}: correlation = {avg_correlation:.4f}")
                else:
                    results['phases'][phase]['correlations'][res].append(np.nan)
    
    return results

def create_phase_aware_correlation_plot(results: Dict, save_path: str = None):
    """创建区分阶段的相关性演化图表"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # 定义颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['comparison_resolutions'])))
    
    # 合并所有步数并标记阶段
    all_steps = []
    all_phases = []
    all_correlations = {res: [] for res in results['comparison_resolutions']}
    
    # 添加warmup数据
    warmup_steps = results['phases']['warmup']['steps']
    for step in warmup_steps:
        all_steps.append(step)
        all_phases.append('warmup')
    
    for res in results['comparison_resolutions']:
        warmup_corrs = results['phases']['warmup']['correlations'][res]
        all_correlations[res].extend(warmup_corrs)
    
    # 添加simulation数据
    sim_steps = results['phases']['simulation']['steps']
    for step in sim_steps:
        all_steps.append(step)
        all_phases.append('simulation')
    
    for res in results['comparison_resolutions']:
        sim_corrs = results['phases']['simulation']['correlations'][res]
        all_correlations[res].extend(sim_corrs)
    
    # 创建连续的x轴索引
    x_indices = list(range(len(all_steps)))
    
    # 找到阶段分界点
    warmup_indices = [i for i, phase in enumerate(all_phases) if phase == 'warmup']
    sim_indices = [i for i, phase in enumerate(all_phases) if phase == 'simulation']
    
    # 添加背景色区分阶段
    if warmup_indices:
        ax.axvspan(min(warmup_indices)-0.5, max(warmup_indices)+0.5, 
                  alpha=0.15, color='blue', label='Warmup Phase')
    
    if sim_indices:
        ax.axvspan(min(sim_indices)-0.5, max(sim_indices)+0.5, 
                  alpha=0.15, color='orange', label='Simulation Phase')
    
    # 绘制每个分辨率的相关性曲线
    for i, res in enumerate(results['comparison_resolutions']):
        correlations = all_correlations[res]
        
        # 过滤NaN值
        valid_indices = [j for j, corr in enumerate(correlations) if not np.isnan(corr)]
        valid_x = [x_indices[j] for j in valid_indices]
        valid_corrs = [correlations[j] for j in valid_indices]
        
        if valid_corrs:
            ax.plot(valid_x, valid_corrs, 'o-', color=colors[i], 
                   label=f'{res}×{res}', linewidth=3, markersize=6, alpha=0.8)
    
    # 设置x轴标签（显示实际步数）
    if len(x_indices) <= 20:
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(step) for step in all_steps], rotation=45)
    else:
        # 太多点时，只显示部分标签
        step_indices = x_indices[::len(x_indices)//10]
        ax.set_xticks(step_indices)
        ax.set_xticklabels([str(all_steps[i]) for i in step_indices], rotation=45)
    
    # 设置图表属性
    ax.set_xlabel('Time Steps', fontsize=14, weight='bold')
    ax.set_ylabel('Spatial Vorticity Correlation', fontsize=14, weight='bold')
    ax.set_title(f'Multi-Resolution Spatial Vorticity Correlation Evolution\n'
                f'Reference: {results["reference_resolution"]}×{results["reference_resolution"]} '
                f'(Warmup + Simulation Phases)', fontsize=16, weight='bold')
    
    ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # 美化图表
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Multi-resolution correlation plot saved: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Multi-resolution spatial vorticity correlation analysis with phase separation')
    parser.add_argument('--base_dir', type=str, default='/Volumes/T7',
                       help='Base directory containing resolution subdirectories')
    parser.add_argument('--reference_resolution', type=int, default=2048,
                       help='Reference resolution')
    parser.add_argument('--comparison_resolutions', nargs='+', type=int, 
                       default=[64, 128, 256, 512, 1024],
                       help='Comparison resolutions')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum step number to analyze')
    parser.add_argument('--save_dir', type=str, default='multiresolution_correlation_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # 运行分析
        results = analyze_multiresolution_correlation(
            args.base_dir, args.reference_resolution, 
            args.comparison_resolutions, args.max_steps)
        
        # 创建图表
        plot_path = f"{args.save_dir}/multiresolution_spatial_correlation_with_phases.png"
        create_phase_aware_correlation_plot(results, plot_path)
        
        # 保存结果
        results_path = f"{args.save_dir}/multiresolution_correlation_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Analysis completed!")
        print(f"📁 Results directory: {args.save_dir}")
        print(f"📈 Plot: {plot_path}")
        print(f"📄 Results: {results_path}")
        
        # 打印摘要
        print(f"\n📊 Summary:")
        for phase in ['warmup', 'simulation']:
            if results['phases'][phase]['steps']:
                print(f"\n{phase.capitalize()} phase:")
                for res in args.comparison_resolutions:
                    corrs = results['phases'][phase]['correlations'][res]
                    if corrs and not all(np.isnan(corrs)):
                        avg_corr = np.nanmean(corrs)
                        print(f"  {res}×{res}: mean correlation = {avg_corr:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
