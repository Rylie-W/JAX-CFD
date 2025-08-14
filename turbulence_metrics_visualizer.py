import os
# Force JAX to use CPU to avoid CUDA/cuDNN issues
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import jax
import jax.numpy as jnp
from scipy import stats
from typing import Dict, Tuple, List
import json
from pathlib import Path

# Additional JAX configuration for CPU
jax.config.update('jax_platform_name', 'cpu')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class TurbulenceMetricsVisualizer:
    """湍流物理指标可视化类"""
    
    def __init__(self):
        self.results = {}
        
    def load_data(self, training_file: str, pict_file: str, max_timesteps: int = 1000):
        """加载和对齐数据"""
        print(f"Loading training data: {training_file}")
        training_data = np.load(training_file)
        
        print(f"Loading PICT data: {pict_file}")
        pict_data = np.load(pict_file)
        
        # 提取速度场
        training_u = training_data['u'][:max_timesteps]
        training_v = training_data['v'][:max_timesteps]
        pict_u = pict_data['u'][:max_timesteps]
        pict_v = pict_data['v'][:max_timesteps]
        
        # 确保时间维度一致
        min_time = min(training_u.shape[0], pict_u.shape[0])
        training_u = training_u[:min_time]
        training_v = training_v[:min_time]
        pict_u = pict_u[:min_time]
        pict_v = pict_v[:min_time]
        
        print(f"Data shapes - Training: {training_u.shape}, PICT: {pict_u.shape}")
        
        return {
            'training_u': training_u,
            'training_v': training_v,
            'pict_u': pict_u,
            'pict_v': pict_v,
            'resolution': training_data.get('resolution', training_u.shape[-1]),
            'timesteps': min_time
        }
    
    def compute_energy_spectrum(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算能量谱"""
        # 时间平均
        u_mean = jnp.mean(u, axis=0)
        v_mean = jnp.mean(v, axis=0)
        
        # 2D FFT
        u_fft = jnp.fft.fft2(u_mean)
        v_fft = jnp.fft.fft2(v_mean)
        
        # 能量密度
        energy_density = (jnp.abs(u_fft)**2 + jnp.abs(v_fft)**2) / 2
        
        # 波数网格
        nx, ny = u_mean.shape
        kx = jnp.fft.fftfreq(nx, 1.0) * nx
        ky = jnp.fft.fftfreq(ny, 1.0) * ny
        kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
        k_magnitude = jnp.sqrt(kx_grid**2 + ky_grid**2)
        
        # 径向平均
        k_max = int(min(nx, ny) // 2)
        k_bins = jnp.linspace(0, k_max, k_max + 1)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        energy_spectrum = jnp.zeros(len(k_centers))
        
        for i in range(len(k_bins) - 1):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
            if jnp.sum(mask) > 0:
                energy_spectrum = energy_spectrum.at[i].set(jnp.mean(energy_density[mask]))
        
        return k_centers, energy_spectrum
    
    def compute_structure_functions(self, u: np.ndarray, v: np.ndarray, max_r: int = 32) -> Dict:
        """计算结构函数"""
        # 时间和空间平均
        u_mean = jnp.mean(u, axis=0)
        v_mean = jnp.mean(v, axis=0)
        
        r_values = jnp.arange(1, min(max_r, u_mean.shape[0]//4))
        s2_u = jnp.zeros(len(r_values))
        s2_v = jnp.zeros(len(r_values))
        s3_u = jnp.zeros(len(r_values))  # 三阶结构函数
        
        for i, r in enumerate(r_values):
            # 纵向结构函数
            du_x = u_mean[r:, :] - u_mean[:-r, :]
            dv_x = v_mean[r:, :] - v_mean[:-r, :]
            
            s2_u = s2_u.at[i].set(jnp.mean(du_x**2))
            s2_v = s2_v.at[i].set(jnp.mean(dv_x**2))
            s3_u = s3_u.at[i].set(jnp.mean(du_x**3))
        
        return {
            'r_values': r_values,
            's2_u': s2_u,
            's2_v': s2_v,
            's3_u': s3_u
        }
    
    def compute_vorticity_statistics(self, u: np.ndarray, v: np.ndarray) -> Dict:
        """计算涡度场统计"""
        # 计算涡度: ω = ∂v/∂x - ∂u/∂y
        dvdx = jnp.gradient(v, axis=-1)
        dudy = jnp.gradient(u, axis=-2)
        vorticity = dvdx - dudy
        
        # 计算涡度统计量
        vort_flat = vorticity.flatten()
        
        return {
            'mean': float(jnp.mean(vorticity)),
            'std': float(jnp.std(vorticity)),
            'skewness': float(stats.skew(vort_flat)),
            'kurtosis': float(stats.kurtosis(vort_flat)),
            'rms': float(jnp.sqrt(jnp.mean(vorticity**2))),
            'max_abs': float(jnp.max(jnp.abs(vorticity)))
        }
    
    def compute_energy_dissipation_rate(self, u: np.ndarray, v: np.ndarray, viscosity: float = 1e-3) -> Dict:
        """计算能量耗散率"""
        # 计算速度梯度
        dudx = jnp.gradient(u, axis=-1)
        dudy = jnp.gradient(u, axis=-2)
        dvdx = jnp.gradient(v, axis=-1)
        dvdy = jnp.gradient(v, axis=-2)
        
        # 应变率张量分量
        s11 = dudx
        s12 = 0.5 * (dudy + dvdx)
        s22 = dvdy
        
        # 耗散率: ε = 2ν * Sij * Sij
        dissipation = 2 * viscosity * (s11**2 + 2*s12**2 + s22**2)
        
        # 统计量
        dissip_flat = dissipation.flatten()
        
        return {
            'mean': float(jnp.mean(dissipation)),
            'std': float(jnp.std(dissipation)),
            'max': float(jnp.max(dissipation)),
            'skewness': float(stats.skew(dissip_flat)),
            'kurtosis': float(stats.kurtosis(dissip_flat))
        }
    
    def compute_reynolds_number_local(self, u: np.ndarray, v: np.ndarray, viscosity: float = 1e-3) -> Dict:
        """计算局部雷诺数相关量"""
        # Taylor microscale Reynolds number
        # Re_λ = u_rms * λ / ν
        # where λ = sqrt(15 * ν * u_rms^2 / ε)
        
        u_rms = jnp.sqrt(jnp.mean(u**2 + v**2))
        
        # 计算耗散率
        dissip_stats = self.compute_energy_dissipation_rate(u, v, viscosity)
        epsilon = dissip_stats['mean']
        
        if epsilon > 0:
            taylor_lambda = jnp.sqrt(15 * viscosity * u_rms**2 / epsilon)
            re_lambda = u_rms * taylor_lambda / viscosity
        else:
            taylor_lambda = 0.0
            re_lambda = 0.0
        
        return {
            'u_rms': float(u_rms),
            'taylor_lambda': float(taylor_lambda),
            'reynolds_lambda': float(re_lambda),
            'epsilon': epsilon
        }
    
    def compute_integral_scales(self, u: np.ndarray, v: np.ndarray) -> Dict:
        """计算积分长度和时间尺度"""
        # 空间积分尺度
        u_mean = jnp.mean(u, axis=0)
        
        # 自相关函数
        autocorr = []
        max_lag = min(20, u_mean.shape[-1]//4)
        
        for lag in range(max_lag):
            shifted = jnp.roll(u_mean, lag, axis=-1)
            corr = jnp.corrcoef(u_mean.flatten(), shifted.flatten())[0, 1]
            if jnp.isnan(corr):
                corr = 0.0
            autocorr.append(corr)
        
        autocorr = jnp.array(autocorr)
        
        # 积分长度尺度 (第一个零点或积分到某个阈值)
        integral_length = 0.0
        for i, corr in enumerate(autocorr):
            if corr > 0.1:  # 积分到相关性降到0.1
                integral_length += 1.0
            else:
                break
        
        # 时间积分尺度
        temporal_autocorr = []
        max_time_lag = min(20, u.shape[0]//4)
        u_flat = jnp.mean(u, axis=(-2, -1))
        
        for lag in range(max_time_lag):
            if lag == 0:
                corr = 1.0
            else:
                corr = jnp.corrcoef(u_flat[:-lag], u_flat[lag:])[0, 1]
                if jnp.isnan(corr):
                    corr = 0.0
            temporal_autocorr.append(corr)
        
        temporal_autocorr = jnp.array(temporal_autocorr)
        
        integral_time = 0.0
        for i, corr in enumerate(temporal_autocorr):
            if corr > 0.1:
                integral_time += 1.0
            else:
                break
        
        return {
            'spatial_integral_length': float(integral_length),
            'temporal_integral_scale': float(integral_time),
            'spatial_autocorr': autocorr.tolist(),
            'temporal_autocorr': temporal_autocorr.tolist()
        }
    
    def compute_intermittency_measures(self, field: np.ndarray) -> Dict:
        """计算间歇性测量"""
        field_flat = field.flatten()
        
        # 去除均值
        field_fluct = field_flat - jnp.mean(field_flat)
        
        # 计算矩
        var = jnp.var(field_fluct)
        if var > 0:
            skewness = jnp.mean(field_fluct**3) / (var**(3/2))
            kurtosis = jnp.mean(field_fluct**4) / (var**2)
            flatness = kurtosis - 3.0  # 超峰度
        else:
            skewness = 0.0
            kurtosis = 3.0
            flatness = 0.0
        
        # 间歇性参数
        if var > 0:
            # 基于四阶矩的间歇性测量
            intermittency = flatness / 3.0  # 标准化的间歇性参数
        else:
            intermittency = 0.0
        
        return {
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'flatness': float(flatness),
            'intermittency_parameter': float(intermittency)
        }
    
    def convert_to_json_serializable(self, obj):
        """将JAX数组和其他不可序列化对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # JAX arrays and numpy arrays
            return obj.tolist()
        elif isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif hasattr(obj, '__array__'):  # JAX arrays that might not have tolist
            return np.array(obj).tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Try to convert to float, otherwise convert to string
            try:
                return float(obj)
            except:
                return str(obj)
    
    def analyze_all_metrics(self, data: Dict) -> Dict:
        """运行所有物理指标分析"""
        print("Running comprehensive turbulence metrics analysis...")
        
        training_u, training_v = data['training_u'], data['training_v']
        pict_u, pict_v = data['pict_u'], data['pict_v']
        
        results = {}
        
        # 1. 基础误差指标
        print("  1. Basic error metrics...")
        results['basic_metrics'] = {
            'rmse_u': float(jnp.sqrt(jnp.mean((training_u - pict_u)**2))),
            'rmse_v': float(jnp.sqrt(jnp.mean((training_v - pict_v)**2))),
            'correlation_u': float(jnp.corrcoef(training_u.flatten(), pict_u.flatten())[0, 1]),
            'correlation_v': float(jnp.corrcoef(training_v.flatten(), pict_v.flatten())[0, 1])
        }
        
        # 2. 能量谱分析
        print("  2. Energy spectrum analysis...")
        k_train, E_train = self.compute_energy_spectrum(training_u, training_v)
        k_pict, E_pict = self.compute_energy_spectrum(pict_u, pict_v)
        
        # 能量谱指标
        valid_mask = (E_train > 1e-10) & (E_pict > 1e-10)
        if jnp.sum(valid_mask) > 0:
            log_diff = jnp.abs(jnp.log(E_pict) - jnp.log(E_train))
            energy_metric = jnp.mean(log_diff[valid_mask])
        else:
            energy_metric = float('inf')
        
        results['energy_spectrum'] = {
            'k_values': k_train.tolist(),
            'E_training': E_train.tolist(),
            'E_pict': E_pict.tolist(),
            'log_difference_metric': float(energy_metric)
        }
        
        # 3. 结构函数分析
        print("  3. Structure functions...")
        struct_train = self.compute_structure_functions(training_u, training_v)
        struct_pict = self.compute_structure_functions(pict_u, pict_v)
        
        results['structure_functions'] = {
            'training': struct_train,
            'pict': struct_pict
        }
        
        # 4. 涡度统计
        print("  4. Vorticity statistics...")
        vort_train = self.compute_vorticity_statistics(training_u, training_v)
        vort_pict = self.compute_vorticity_statistics(pict_u, pict_v)
        
        results['vorticity_stats'] = {
            'training': vort_train,
            'pict': vort_pict
        }
        
        # 5. 能量耗散率
        print("  5. Energy dissipation rate...")
        dissip_train = self.compute_energy_dissipation_rate(training_u, training_v)
        dissip_pict = self.compute_energy_dissipation_rate(pict_u, pict_v)
        
        results['dissipation'] = {
            'training': dissip_train,
            'pict': dissip_pict
        }
        
        # 6. 雷诺数分析
        print("  6. Reynolds number analysis...")
        re_train = self.compute_reynolds_number_local(training_u, training_v)
        re_pict = self.compute_reynolds_number_local(pict_u, pict_v)
        
        results['reynolds_analysis'] = {
            'training': re_train,
            'pict': re_pict
        }
        
        # 7. 积分尺度
        print("  7. Integral scales...")
        scales_train = self.compute_integral_scales(training_u, training_v)
        scales_pict = self.compute_integral_scales(pict_u, pict_v)
        
        results['integral_scales'] = {
            'training': scales_train,
            'pict': scales_pict
        }
        
        # 8. 间歇性分析
        print("  8. Intermittency analysis...")
        inter_train_u = self.compute_intermittency_measures(training_u)
        inter_pict_u = self.compute_intermittency_measures(pict_u)
        
        results['intermittency'] = {
            'training': inter_train_u,
            'pict': inter_pict_u
        }
        
        self.results = results
        return results
    
    def create_physics_visualization(self, data: Dict, save_dir: str = "turbulence_metrics_results"):
        """创建物理指标可视化"""
        if not self.results:
            raise ValueError("No analysis results found. Run analyze_all_metrics first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建大型图形
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. 能量谱对比
        ax1 = fig.add_subplot(gs[0, 0])
        k_vals = np.array(self.results['energy_spectrum']['k_values'])
        E_train = np.array(self.results['energy_spectrum']['E_training'])
        E_pict = np.array(self.results['energy_spectrum']['E_pict'])
        
        valid_idx = (k_vals > 0) & (E_train > 0) & (E_pict > 0)
        ax1.loglog(k_vals[valid_idx], E_train[valid_idx], 'b-', label='Training', linewidth=2)
        ax1.loglog(k_vals[valid_idx], E_pict[valid_idx], 'r--', label='PICT', linewidth=2)
        
        # 添加Kolmogorov -5/3参考线
        if len(k_vals[valid_idx]) > 5:
            k_ref = k_vals[valid_idx][5:]
            E_ref = E_train[valid_idx][5] * (k_ref/k_vals[valid_idx][5])**(-5/3)
            ax1.loglog(k_ref, E_ref, 'k:', alpha=0.7, label='k^(-5/3)')
        
        ax1.set_xlabel('Wavenumber k')
        ax1.set_ylabel('Energy E(k)')
        ax1.set_title('Energy Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 结构函数对比
        ax2 = fig.add_subplot(gs[0, 1])
        r_vals_train = np.array(self.results['structure_functions']['training']['r_values'])
        s2_train = np.array(self.results['structure_functions']['training']['s2_u'])
        s2_pict = np.array(self.results['structure_functions']['pict']['s2_u'])
        
        ax2.loglog(r_vals_train, s2_train, 'b-', label='Training S₂(r)', linewidth=2)
        ax2.loglog(r_vals_train, s2_pict, 'r--', label='PICT S₂(r)', linewidth=2)
        
        # 添加2/3标度参考线
        if len(r_vals_train) > 3:
            r_ref = r_vals_train[3:]
            s2_ref = s2_train[3] * (r_ref/r_vals_train[3])**(2/3)
            ax2.loglog(r_ref, s2_ref, 'k:', alpha=0.7, label='r^(2/3)')
        
        ax2.set_xlabel('Separation r')
        ax2.set_ylabel('Structure Function S₂(r)')
        ax2.set_title('Second-Order Structure Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 涡度统计对比
        ax3 = fig.add_subplot(gs[0, 2])
        vort_train = self.results['vorticity_stats']['training']
        vort_pict = self.results['vorticity_stats']['pict']
        
        stats_names = ['RMS', 'Std', 'Skewness', 'Kurtosis']
        train_vals = [vort_train['rms'], vort_train['std'], 
                     vort_train['skewness'], vort_train['kurtosis']]
        pict_vals = [vort_pict['rms'], vort_pict['std'],
                    vort_pict['skewness'], vort_pict['kurtosis']]
        
        x_pos = np.arange(len(stats_names))
        width = 0.35
        
        ax3.bar(x_pos - width/2, train_vals, width, label='Training', alpha=0.8)
        ax3.bar(x_pos + width/2, pict_vals, width, label='PICT', alpha=0.8)
        ax3.set_xlabel('Statistic')
        ax3.set_ylabel('Value')
        ax3.set_title('Vorticity Statistics')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(stats_names)
        ax3.legend()
        
        # 4. 雷诺数对比
        ax4 = fig.add_subplot(gs[0, 3])
        re_train = self.results['reynolds_analysis']['training']
        re_pict = self.results['reynolds_analysis']['pict']
        
        re_metrics = ['u_rms', 'Re_λ', 'λ', 'ε']
        re_train_vals = [re_train['u_rms'], re_train['reynolds_lambda'], 
                        re_train['taylor_lambda'], re_train['epsilon']]
        re_pict_vals = [re_pict['u_rms'], re_pict['reynolds_lambda'],
                       re_pict['taylor_lambda'], re_pict['epsilon']]
        
        # 归一化显示
        re_train_norm = np.array(re_train_vals) / np.max(re_train_vals)
        re_pict_norm = np.array(re_pict_vals) / np.max(re_pict_vals)
        
        x_pos = np.arange(len(re_metrics))
        ax4.bar(x_pos - width/2, re_train_norm, width, label='Training', alpha=0.8)
        ax4.bar(x_pos + width/2, re_pict_norm, width, label='PICT', alpha=0.8)
        ax4.set_xlabel('Reynolds Metrics')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Reynolds Number Analysis')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(re_metrics)
        ax4.legend()
        
        # 5-8. 速度场快照
        ax5 = fig.add_subplot(gs[1, 0])
        im1 = ax5.imshow(data['training_u'][0], cmap='RdBu_r', origin='lower')
        ax5.set_title('Training U-velocity (t=0)')
        plt.colorbar(im1, ax=ax5, shrink=0.8)
        
        ax6 = fig.add_subplot(gs[1, 1])
        im2 = ax6.imshow(data['pict_u'][0], cmap='RdBu_r', origin='lower')
        ax6.set_title('PICT U-velocity (t=0)')
        plt.colorbar(im2, ax=ax6, shrink=0.8)
        
        ax7 = fig.add_subplot(gs[1, 2])
        diff = data['training_u'][0] - data['pict_u'][0]
        im3 = ax7.imshow(diff, cmap='RdBu_r', origin='lower')
        ax7.set_title('U-velocity Difference')
        plt.colorbar(im3, ax=ax7, shrink=0.8)
        
        # 涡度场对比
        ax8 = fig.add_subplot(gs[1, 3])
        # 计算涡度
        dvdx = np.gradient(data['training_v'][0], axis=-1)
        dudy = np.gradient(data['training_u'][0], axis=-2)
        vorticity_train = dvdx - dudy
        
        im4 = ax8.imshow(vorticity_train, cmap='RdBu_r', origin='lower')
        ax8.set_title('Training Vorticity (t=0)')
        plt.colorbar(im4, ax=ax8, shrink=0.8)
        
        # 9. 时间演化对比
        ax9 = fig.add_subplot(gs[2, 0])
        timesteps = np.arange(data['timesteps'])
        
        # 动能演化
        ke_train = 0.5 * np.mean(data['training_u']**2 + data['training_v']**2, axis=(-2, -1))
        ke_pict = 0.5 * np.mean(data['pict_u']**2 + data['pict_v']**2, axis=(-2, -1))
        
        ax9.plot(timesteps, ke_train, 'b-', label='Training', linewidth=2)
        ax9.plot(timesteps, ke_pict, 'r--', label='PICT', linewidth=2)
        ax9.set_xlabel('Time step')
        ax9.set_ylabel('Kinetic Energy')
        ax9.set_title('Kinetic Energy Evolution')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. 空间相关性对比
        ax10 = fig.add_subplot(gs[2, 1])
        spat_corr_train = np.array(self.results['integral_scales']['training']['spatial_autocorr'])
        spat_corr_pict = np.array(self.results['integral_scales']['pict']['spatial_autocorr'])
        spat_lags = np.arange(len(spat_corr_train))
        
        ax10.plot(spat_lags, spat_corr_train, 'b-', label='Training', linewidth=2)
        ax10.plot(spat_lags, spat_corr_pict, 'r--', label='PICT', linewidth=2)
        ax10.set_xlabel('Spatial lag')
        ax10.set_ylabel('Autocorrelation')
        ax10.set_title('Spatial Autocorrelation')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # 11. 时间相关性对比
        ax11 = fig.add_subplot(gs[2, 2])
        temp_corr_train = np.array(self.results['integral_scales']['training']['temporal_autocorr'])
        temp_corr_pict = np.array(self.results['integral_scales']['pict']['temporal_autocorr'])
        temp_lags = np.arange(len(temp_corr_train))
        
        ax11.plot(temp_lags, temp_corr_train, 'b-', label='Training', linewidth=2)
        ax11.plot(temp_lags, temp_corr_pict, 'r--', label='PICT', linewidth=2)
        ax11.set_xlabel('Time lag')
        ax11.set_ylabel('Autocorrelation')
        ax11.set_title('Temporal Autocorrelation')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 12. 间歇性对比
        ax12 = fig.add_subplot(gs[2, 3])
        inter_train = self.results['intermittency']['training']
        inter_pict = self.results['intermittency']['pict']
        
        inter_names = ['Skewness', 'Kurtosis', 'Flatness', 'Intermittency']
        inter_train_vals = [inter_train['skewness'], inter_train['kurtosis'],
                           inter_train['flatness'], inter_train['intermittency_parameter']]
        inter_pict_vals = [inter_pict['skewness'], inter_pict['kurtosis'],
                          inter_pict['flatness'], inter_pict['intermittency_parameter']]
        
        x_pos = np.arange(len(inter_names))
        ax12.bar(x_pos - width/2, inter_train_vals, width, label='Training', alpha=0.8)
        ax12.bar(x_pos + width/2, inter_pict_vals, width, label='PICT', alpha=0.8)
        ax12.set_xlabel('Measure')
        ax12.set_ylabel('Value')
        ax12.set_title('Intermittency Measures')
        ax12.set_xticks(x_pos)
        ax12.set_xticklabels(inter_names, rotation=45)
        ax12.legend()
        
        # 13. 耗散率统计
        ax13 = fig.add_subplot(gs[3, 0])
        dissip_train = self.results['dissipation']['training']
        dissip_pict = self.results['dissipation']['pict']
        
        dissip_names = ['Mean ε', 'Std ε', 'Max ε']
        dissip_train_vals = [dissip_train['mean'], dissip_train['std'], dissip_train['max']]
        dissip_pict_vals = [dissip_pict['mean'], dissip_pict['std'], dissip_pict['max']]
        
        x_pos = np.arange(len(dissip_names))
        ax13.bar(x_pos - width/2, dissip_train_vals, width, label='Training', alpha=0.8)
        ax13.bar(x_pos + width/2, dissip_pict_vals, width, label='PICT', alpha=0.8)
        ax13.set_xlabel('Statistic')
        ax13.set_ylabel('Dissipation Rate')
        ax13.set_title('Energy Dissipation Statistics')
        ax13.set_xticks(x_pos)
        ax13.set_xticklabels(dissip_names)
        ax13.legend()
        
        # 14. 误差演化
        ax14 = fig.add_subplot(gs[3, 1])
        u_errors = np.sqrt(np.mean((data['training_u'] - data['pict_u'])**2, axis=(-2, -1)))
        v_errors = np.sqrt(np.mean((data['training_v'] - data['pict_v'])**2, axis=(-2, -1)))
        
        ax14.plot(timesteps, u_errors, 'b-', label='RMSE U', linewidth=2)
        ax14.plot(timesteps, v_errors, 'r-', label='RMSE V', linewidth=2)
        ax14.set_xlabel('Time step')
        ax14.set_ylabel('RMSE')
        ax14.set_title('Error Evolution')
        ax14.legend()
        ax14.grid(True, alpha=0.3)
        
        # 15. 基础统计对比
        ax15 = fig.add_subplot(gs[3, 2])
        basic = self.results['basic_metrics']
        
        basic_names = ['RMSE U', 'RMSE V', 'Corr U', 'Corr V']
        basic_vals = [basic['rmse_u'], basic['rmse_v'], 
                     basic['correlation_u'], basic['correlation_v']]
        
        bars = ax15.bar(basic_names, basic_vals, color=['lightblue', 'lightcoral', 'lightgreen', 'gold'])
        ax15.set_ylabel('Value')
        ax15.set_title('Basic Metrics')
        ax15.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, basic_vals):
            ax15.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(basic_vals)*0.01,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 16. 总结评估
        ax16 = fig.add_subplot(gs[3, 3])
        
        # 计算总体评估分数
        corr_score = (basic['correlation_u'] + basic['correlation_v']) / 2
        energy_score = 1.0 / (1.0 + self.results['energy_spectrum']['log_difference_metric'])
        
        if corr_score > 0.99:
            assessment = "🎉 EXCELLENT"
            color = "green"
        elif corr_score > 0.95:
            assessment = "✅ GOOD"
            color = "blue"
        elif corr_score > 0.9:
            assessment = "⚠️ MODERATE"
            color = "orange"
        else:
            assessment = "❌ POOR"
            color = "red"
        
        summary_text = f"""
TURBULENCE PHYSICS ASSESSMENT

Field Correlation: {corr_score:.4f}
Energy Spectrum Metric: {self.results['energy_spectrum']['log_difference_metric']:.4f}

Reynolds Numbers:
  Training Re_λ: {self.results['reynolds_analysis']['training']['reynolds_lambda']:.1f}
  PICT Re_λ: {self.results['reynolds_analysis']['pict']['reynolds_lambda']:.1f}

Assessment: {assessment}
        """
        
        ax16.text(0.05, 0.95, summary_text, transform=ax16.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        ax16.set_xlim(0, 1)
        ax16.set_ylim(0, 1)
        ax16.axis('off')
        
        plt.suptitle('Comprehensive Turbulence Physics Metrics Comparison', 
                    fontsize=16, fontweight='bold')
        
        # 保存图像
        save_path = os.path.join(save_dir, 'turbulence_physics_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Physics metrics visualization saved to: {save_path}")
        return save_path

def main():
    """主函数"""
    
    # 配置
    resolution = "128x128"
    max_timesteps = 500
    
    # 文件路径
    training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
    pict_file = f"data/pict_data/decaying_turbulence_{resolution}_index_1.npz"
    
    print("🌊 TURBULENCE PHYSICS METRICS COMPARISON")
    print("="*60)
    
    # 检查文件存在
    if not os.path.exists(training_file):
        print(f"❌ Training file not found: {training_file}")
        return
    if not os.path.exists(pict_file):
        print(f"❌ PICT file not found: {pict_file}")
        return
    
    # 初始化可视化器
    visualizer = TurbulenceMetricsVisualizer()
    
    try:
        # 加载数据
        print("📁 Loading data...")
        data = visualizer.load_data(training_file, pict_file, max_timesteps)
        
        # 运行物理指标分析
        print("🔬 Running physics metrics analysis...")
        results = visualizer.analyze_all_metrics(data)
        
        # 创建可视化
        print("📊 Creating physics visualization...")
        save_dir = "turbulence_physics_results"
        plot_path = visualizer.create_physics_visualization(data, save_dir)
        
        # 保存详细结果
        results_file = os.path.join(save_dir, 'physics_metrics_results.json')
        with open(results_file, 'w') as f:
            json.dump(visualizer.convert_to_json_serializable(results), f, indent=2)
        
        print("\n" + "="*60)
        print("✅ PHYSICS ANALYSIS COMPLETE!")
        print(f"📁 Results directory: {save_dir}/")
        print(f"🖼️ Visualization: {plot_path}")
        print(f"📄 Detailed results: {results_file}")
        print("="*60)
        
        # 打印关键发现
        print("\n🔍 KEY PHYSICS FINDINGS:")
        basic = results['basic_metrics']
        re_train = results['reynolds_analysis']['training']
        re_pict = results['reynolds_analysis']['pict']
        
        print(f"  • Field correlations: U={basic['correlation_u']:.4f}, V={basic['correlation_v']:.4f}")
        print(f"  • Energy spectrum metric: {results['energy_spectrum']['log_difference_metric']:.4f}")
        print(f"  • Reynolds λ: Training={re_train['reynolds_lambda']:.1f}, PICT={re_pict['reynolds_lambda']:.1f}")
        print(f"  • Dissipation rate ratio: {results['dissipation']['pict']['mean']/results['dissipation']['training']['mean']:.4f}")
        
        avg_corr = (basic['correlation_u'] + basic['correlation_v']) / 2
        if avg_corr > 0.99:
            print("  🎉 EXCELLENT physics agreement!")
        elif avg_corr > 0.95:
            print("  ✅ GOOD physics agreement")
        elif avg_corr > 0.9:
            print("  ⚠️ MODERATE agreement - some physics discrepancies")
        else:
            print("  ❌ POOR agreement - significant physics differences")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 