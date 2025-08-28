#!/usr/bin/env python3
"""
Physics-based turbulence simulation comparison analysis.

This tool compares turbulence simulations using physical properties and statistical
measures that are robust to the chaotic nature of turbulent flows, rather than 
point-wise differences which are highly sensitive to initial conditions.

Key improvements over traditional comparison:
- Energy decay law analysis (E(t) ~ t^(-n))
- Kolmogorov scaling verification (E(k) ~ k^(-5/3))
- Higher-order velocity statistics (skewness, kurtosis)
- Integral length scale comparison
- Overall physics-based assessment

These metrics focus on whether both simulations capture the essential physics
of turbulence, avoiding the chaos sensitivity of point-wise comparisons.
"""

import os
# Force JAX to use CPU to avoid CUDA/cuDNN issues
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
import json
from pathlib import Path

# Additional JAX configuration for CPU
jax.config.update('jax_platform_name', 'cpu')

def load_and_align_data(training_file: str, pred_file: str, sampling_interval: int = 10, max_timesteps: int = 1000):
    """Load and temporally align training and prediction data."""
    print(f"Loading training data: {training_file}")
    training_data = np.load(training_file)
    
    print(f"Loading prediction data: {pred_file}")
    pred_data = np.load(pred_file)
    
    # Extract velocity fields
    training_u = training_data['u'][:max_timesteps]  # Take first max_timesteps steps
    training_v = training_data['v'][:max_timesteps]
    pred_u = pred_data['u'][:max_timesteps]
    pred_v = pred_data['v'][:max_timesteps]
    
    # Subsample training data to match prediction sampling
    training_u_subsampled = training_u[::sampling_interval]
    training_v_subsampled = training_v[::sampling_interval]
    pred_u = pred_u[:training_u_subsampled.shape[0]]
    pred_v = pred_v[:training_v_subsampled.shape[0]]
    
    print(f"Original training shape: {training_u.shape}")
    print(f"Subsampled training shape: {training_u_subsampled.shape}")
    print(f"Prediction shape: {pred_u.shape}")
    
    # Verify alignment
    assert training_u_subsampled.shape[0] == pred_u.shape[0], f"Time alignment failed: {training_u_subsampled.shape[0]} vs {pred_u.shape[0]}"
    
    return {
        'training_u': training_u_subsampled,
        'training_v': training_v_subsampled,
        'pred_u': pred_u,
        'pred_v': pred_v,
        'resolution': training_data['resolution'],
        'time_steps': training_u_subsampled.shape[0]
    }

def compute_energy_spectrum(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute energy spectrum from velocity fields."""
    # Compute 2D FFT of velocity fields
    u_fft = jnp.fft.fft2(u, axes=(-2, -1))
    v_fft = jnp.fft.fft2(v, axes=(-2, -1))
    
    # Compute energy density
    energy_density = (jnp.abs(u_fft)**2 + jnp.abs(v_fft)**2) / 2
    
    # Get spatial dimensions
    nx, ny = u.shape[-2], u.shape[-1]
    kx = jnp.fft.fftfreq(nx, 1.0) * nx  # Scale by nx for proper wavenumber
    ky = jnp.fft.fftfreq(ny, 1.0) * ny  # Scale by ny for proper wavenumber
    
    # Create 2D wavenumber grid
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
    k_magnitude = jnp.sqrt(kx_grid**2 + ky_grid**2)
    
    # FIXED: Radial averaging to get 1D spectrum
    k_max = int(jnp.max(k_magnitude)) + 1  # +1 to ensure non-empty range
    k_max = max(k_max, 10)  # Ensure at least 10 bins
    k_bins = jnp.linspace(0, k_max, k_max + 1)
    energy_spectrum = jnp.zeros((energy_density.shape[0], len(k_bins) - 1))
    
    for i in range(len(k_bins) - 1):
        mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
        energy_spectrum = energy_spectrum.at[:, i].set(
            jnp.mean(energy_density * mask[None, :, :], axis=(-2, -1))
        )
    
    return energy_spectrum

def compute_physics_based_metrics(data: Dict) -> Dict:
    """Apply physics-based evaluation metrics that are robust to chaos."""
    results = {}
    
    training_u = data['training_u']
    training_v = data['training_v'] 
    pred_u = data['pred_u']
    pred_v = data['pred_v']
    
    print("Computing physics-based metrics...")
    
    # 1. Energy Decay Law Analysis
    print("  1. Computing energy decay analysis...")
    def compute_energy_decay(u, v):
        kinetic_energy = 0.5 * jnp.mean(u**2 + v**2, axis=(-2, -1))
        time_steps = jnp.arange(1, len(kinetic_energy) + 1)
        
        # Fit power law: E(t) = A * t^(-n)
        log_t = jnp.log(time_steps)
        log_E = jnp.log(kinetic_energy)
        valid_mask = jnp.isfinite(log_E) & jnp.isfinite(log_t)
        
        if jnp.sum(valid_mask) > 5:
            coeffs = jnp.polyfit(log_t[valid_mask], log_E[valid_mask], 1)
            decay_exponent = -coeffs[0]
            fit_quality = jnp.corrcoef(log_t[valid_mask], log_E[valid_mask])[0, 1]**2
        else:
            decay_exponent = jnp.nan
            fit_quality = 0.0
            
        return {
            'kinetic_energy': kinetic_energy.tolist(),
            'decay_exponent': float(decay_exponent),
            'fit_quality': float(fit_quality)
        }
    
    training_decay = compute_energy_decay(training_u, training_v)
    pred_decay = compute_energy_decay(pred_u, pred_v)
    
    results['energy_decay_analysis'] = {
        'training': training_decay,
        'predicted': pred_decay,
        'decay_exponent_difference': abs(training_decay['decay_exponent'] - pred_decay['decay_exponent']),
        'is_physical': (1.0 <= training_decay['decay_exponent'] <= 1.5) and 
                      (1.0 <= pred_decay['decay_exponent'] <= 1.5)
    }
    
    # 2. Enhanced Energy Spectrum Analysis (Kolmogorov scaling)
    print("  2. Computing Kolmogorov scaling analysis...")
    training_spectrum = compute_energy_spectrum(training_u, training_v)
    pred_spectrum = compute_energy_spectrum(pred_u, pred_v)
    
    def analyze_kolmogorov_scaling(spectrum):
        k_values = jnp.arange(1, spectrum.shape[1])
        avg_spectrum = jnp.mean(spectrum, axis=0)[1:]  # Skip k=0
        
        # Find inertial range (k = 3 to k = k_max/3)
        k_max = len(avg_spectrum)
        inertial_start = 3
        inertial_end = max(inertial_start + 3, k_max // 3)
        
        if inertial_end > inertial_start:
            k_inertial = k_values[inertial_start:inertial_end]
            E_inertial = avg_spectrum[inertial_start:inertial_end]
            
            valid_mask = E_inertial > 0
            if jnp.sum(valid_mask) > 3:
                log_k = jnp.log(k_inertial[valid_mask])
                log_E = jnp.log(E_inertial[valid_mask])
                coeffs = jnp.polyfit(log_k, log_E, 1)
                slope = coeffs[0]
                fit_quality = jnp.corrcoef(log_k, log_E)[0, 1]**2
            else:
                slope = jnp.nan
                fit_quality = 0.0
        else:
            slope = jnp.nan
            fit_quality = 0.0
            
        return {
            'inertial_slope': float(slope),
            'deviation_from_kolmogorov': float(abs(slope + 5/3)) if not jnp.isnan(slope) else jnp.nan,
            'fit_quality': float(fit_quality)
        }
    
    training_kolm = analyze_kolmogorov_scaling(training_spectrum)
    pred_kolm = analyze_kolmogorov_scaling(pred_spectrum)
    
    results['kolmogorov_analysis'] = {
        'training': training_kolm,
        'predicted': pred_kolm,
        'slope_difference': abs(training_kolm['inertial_slope'] - pred_kolm['inertial_slope'])
    }
    
    # Traditional energy spectrum comparison (for reference)
    threshold = 0.01
    valid_mask = (training_spectrum > threshold) & (pred_spectrum > threshold)
    log_diff = jnp.abs(jnp.log(pred_spectrum) - jnp.log(training_spectrum))
    energy_metric = jnp.mean(jnp.where(valid_mask, log_diff, 0))
    results['traditional_energy_spectrum_metric'] = float(energy_metric)
    
    # 3. Velocity Statistics Analysis (Higher-order moments)
    print("  3. Computing velocity statistics...")
    def compute_velocity_moments(u, v):
        u_flat = u.flatten()
        v_flat = v.flatten()
        
        # Remove any infinite or NaN values
        u_clean = u_flat[jnp.isfinite(u_flat)]
        v_clean = v_flat[jnp.isfinite(v_flat)]
        
        u_mean, v_mean = jnp.mean(u_clean), jnp.mean(v_clean)
        u_std, v_std = jnp.std(u_clean), jnp.std(v_clean)
        
        # Higher-order moments (skewness and kurtosis indicate non-Gaussianity)
        u_skew = jnp.mean(((u_clean - u_mean) / (u_std + 1e-10))**3)
        v_skew = jnp.mean(((v_clean - v_mean) / (v_std + 1e-10))**3)
        u_kurt = jnp.mean(((u_clean - u_mean) / (u_std + 1e-10))**4) - 3  # Excess kurtosis
        v_kurt = jnp.mean(((v_clean - v_mean) / (v_std + 1e-10))**4) - 3
        
        return {
            'mean': (float(u_mean), float(v_mean)),
            'std': (float(u_std), float(v_std)),
            'skewness': (float(u_skew), float(v_skew)),
            'kurtosis': (float(u_kurt), float(v_kurt))
        }
    
    training_moments = compute_velocity_moments(training_u, training_v)
    pred_moments = compute_velocity_moments(pred_u, pred_v)
    
    results['velocity_statistics'] = {
        'training': training_moments,
        'predicted': pred_moments,
        'skewness_difference': abs(training_moments['skewness'][0] - pred_moments['skewness'][0]),
        'kurtosis_difference': abs(training_moments['kurtosis'][0] - pred_moments['kurtosis'][0])
    }
    
    # 4. Integral Scale Analysis
    print("  4. Computing integral scale analysis...")
    def compute_integral_length_scale(field_2d):
        """Compute integral length scale from spatial autocorrelation."""
        mid_row = field_2d.shape[0] // 2
        signal = field_2d[mid_row, :]
        
        # Autocorrelation
        autocorr = jnp.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
        autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize
        
        # Integral scale = ∫₀^∞ R(r) dr (until first zero crossing)
        zero_crossing = jnp.where(autocorr <= 0)[0]
        if len(zero_crossing) > 0:
            integral_scale = np.trapz(autocorr[:zero_crossing[0]])
        else:
            integral_scale = np.trapz(autocorr)
        
        return float(integral_scale)
    
    # Use time-averaged fields for spatial correlation
    training_u_avg = jnp.mean(training_u, axis=0)
    training_v_avg = jnp.mean(training_v, axis=0)
    pred_u_avg = jnp.mean(pred_u, axis=0)
    pred_v_avg = jnp.mean(pred_v, axis=0)
    
    training_L_scale = compute_integral_length_scale(training_u_avg)
    pred_L_scale = compute_integral_length_scale(pred_u_avg)
    
    results['integral_scales'] = {
        'training_length_scale': training_L_scale,
        'predicted_length_scale': pred_L_scale,
        'length_scale_ratio': pred_L_scale / (training_L_scale + 1e-10)
    }
    
    # 5. Physics-based assessment
    print("  5. Computing overall physics assessment...")
    
    # Check if both simulations follow physical laws
    energy_decay_ok = results['energy_decay_analysis']['is_physical']
    kolmogorov_ok = (results['kolmogorov_analysis']['training']['deviation_from_kolmogorov'] < 0.5 and 
                    results['kolmogorov_analysis']['predicted']['deviation_from_kolmogorov'] < 0.5)
    
    physics_score = 0.0
    if energy_decay_ok:
        physics_score += 0.4
    if kolmogorov_ok:
        physics_score += 0.4
    if results['energy_decay_analysis']['decay_exponent_difference'] < 0.2:
        physics_score += 0.2
    
    results['physics_assessment'] = {
        'overall_physics_score': physics_score,
        'energy_decay_physical': energy_decay_ok,
        'kolmogorov_scaling_physical': kolmogorov_ok,
        'recommendation': (
            "Excellent physical agreement" if physics_score > 0.8 else
            "Good physical agreement" if physics_score > 0.6 else
            "Moderate physical agreement" if physics_score > 0.4 else
            "Poor physical agreement - investigate numerical methods"
        )
    }
    
    # Traditional metrics (for comparison only)
    results['traditional_metrics'] = {
        'rmse_u': float(jnp.sqrt(jnp.mean((training_u - pred_u)**2))),
        'rmse_v': float(jnp.sqrt(jnp.mean((training_v - pred_v)**2))),
        'note': 'These are chaos-sensitive and may not reflect physical accuracy'
    }
    
    return results

def apply_selected_metrics(data: Dict) -> Dict:
    """Legacy function name - now calls physics-based metrics."""
    return compute_physics_based_metrics(data)

def create_comprehensive_plots(data: Dict, results: Dict, resolution: str, save_dir: str):
    """Create comprehensive visualization plots."""
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Energy spectrum comparison
    plt.subplot(3, 3, 1)
    training_spectrum = compute_energy_spectrum(data['training_u'], data['training_v'])
    pred_spectrum = compute_energy_spectrum(data['pred_u'], data['pred_v'])
    
    k_values = jnp.arange(training_spectrum.shape[1])
    plt.loglog(k_values[1:], jnp.mean(training_spectrum, axis=0)[1:], 'b-', label='Training', linewidth=2)
    plt.loglog(k_values[1:], jnp.mean(pred_spectrum, axis=0)[1:], 'r--', label='Predicted', linewidth=2)
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy E(k)')
    plt.title('Energy Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Energy Decay Analysis
    plt.subplot(3, 3, 2)
    if 'energy_decay_analysis' in results:
        training_energy = results['energy_decay_analysis']['training']['kinetic_energy']
        pred_energy = results['energy_decay_analysis']['predicted']['kinetic_energy']
        time_steps = jnp.arange(len(training_energy))
        
        plt.semilogy(time_steps, training_energy, 'b-', label='Training', linewidth=2)
        plt.semilogy(time_steps, pred_energy, 'r--', label='Predicted', linewidth=2)
        
        # Add theoretical decay lines
        if len(training_energy) > 5:
            t_theory = time_steps[1:]  # Avoid t=0
            E_theory_12 = training_energy[1] * (t_theory/1)**(-1.2)  # Physical expectation
            plt.semilogy(t_theory, E_theory_12, 'k:', alpha=0.7, label='t^(-1.2) theory')
        
        plt.xlabel('Time step')
        plt.ylabel('Kinetic Energy')
        plt.title('Energy Decay Law')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Physics-based Metrics Summary
    plt.subplot(3, 3, 3)
    if 'physics_assessment' in results:
        physics_score = results['physics_assessment']['overall_physics_score']
        energy_decay_diff = results['energy_decay_analysis']['decay_exponent_difference']
        kolm_slope_diff = results['kolmogorov_analysis']['slope_difference']
        
        metric_names = ['Physics\nScore', 'Energy Decay\nAgreement', 'Kolmogorov\nAgreement']
        metric_values = [
            physics_score,
            max(0, 1 - energy_decay_diff / 0.5),  # Good if diff < 0.5
            max(0, 1 - kolm_slope_diff / 0.5)
        ]
        
        colors = ['lightgreen' if v > 0.7 else 'yellow' if v > 0.4 else 'lightcoral' for v in metric_values]
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylabel('Score (0-1)')
        plt.title('Physics-Based Assessment')
        plt.ylim(0, 1.1)
        
        for bar, val in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 4-6: Velocity field snapshots (first time step)
    plt.subplot(3, 3, 4)
    plt.imshow(data['training_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar()
    plt.title('Training U-velocity (t=0)')
    
    plt.subplot(3, 3, 5)
    plt.imshow(data['pred_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar() 
    plt.title('Predicted U-velocity (t=0)')
    
    plt.subplot(3, 3, 6)
    plt.imshow(data['training_u'][0] - data['pred_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar()
    plt.title('U-velocity Difference (t=0)')
    
    # Plot 7-9: Error statistics over time
    plt.subplot(3, 3, 7)
    time_steps = jnp.arange(data['time_steps'])
    u_errors = jnp.sqrt(jnp.mean((data['training_u'] - data['pred_u'])**2, axis=(-2, -1)))
    v_errors = jnp.sqrt(jnp.mean((data['training_v'] - data['pred_v'])**2, axis=(-2, -1)))
    plt.plot(time_steps, u_errors, 'b-', label='U RMSE', linewidth=2)
    plt.plot(time_steps, v_errors, 'r-', label='V RMSE', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('RMSE')
    plt.title('Error Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Statistics comparison
    plt.subplot(3, 3, 8)
    stats_data = {
        'RMSE U': results.get('rmse_u', 0),
        'RMSE V': results.get('rmse_v', 0),
        'MAE U': results.get('mae_u', 0),
        'MAE V': results.get('mae_v', 0)
    }
    bars = plt.bar(stats_data.keys(), stats_data.values(), 
                  color=['lightblue', 'lightblue', 'lightgreen', 'lightgreen'])
    plt.ylabel('Error Value')
    plt.title('Error Statistics')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, stats_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 9: Data characteristics
    plt.subplot(3, 3, 9)
    char_data = {
        'Training U std': float(jnp.std(data['training_u'])),
        'Pred U std': float(jnp.std(data['pred_u'])),
        'Training V std': float(jnp.std(data['training_v'])),
        'Pred V std': float(jnp.std(data['pred_v']))
    }
    bars = plt.bar(char_data.keys(), char_data.values(),
                  color=['darkblue', 'blue', 'darkred', 'red'])
    plt.ylabel('Standard Deviation')
    plt.title('Data Characteristics')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, char_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'Comprehensive Data Comparison Analysis - {resolution}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f'comprehensive_analysis_{resolution}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualization saved to: {save_path}")

def main():
    """Main function to run the refined data comparison analysis."""
    # Create results directory
    results_dir = "refined_comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define data files to compare (start with 64x64 for testing)
    resolutions = ['1024x1024']  # Can expand to ['64x64', '128x128', '256x256', '512x512']
    
    all_results = {}
    
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution: {resolution}")
        print(f"{'='*60}")
        
        training_file = f"data/training_data/1024/decaying_turbulence_v2_with_warmup_init_{resolution}_index_1.npz"
        pred_file = f"data/pict_data/turbulence_1024_step1000_{resolution}_index_1.npz"
        
        if not os.path.exists(training_file):
            print(f"Training file not found: {training_file}")
            continue
        if not os.path.exists(pred_file):
            print(f"Predicted file not found: {pred_file}")
            continue
        
        try:
            print("Loading and aligning data...")
            data = load_and_align_data(training_file, pred_file, max_timesteps=1000)
            
            print("Applying evaluation metrics...")
            results = apply_selected_metrics(data)
            
            all_results[resolution] = results
            
            print("Creating visualizations...")
            create_comprehensive_plots(data, results, resolution, results_dir)
            
            results_file = os.path.join(results_dir, f'detailed_results_{resolution}.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📊 Physics-Based Results Summary for {resolution}:")
            print("="*50)
            
            # Physics assessment
            if 'physics_assessment' in results:
                assessment = results['physics_assessment']
                print(f"🔬 Overall Physics Score: {assessment['overall_physics_score']:.3f}/1.0")
                print(f"📝 Assessment: {assessment['recommendation']}")
                print()
            
            # Energy decay analysis
            if 'energy_decay_analysis' in results:
                decay = results['energy_decay_analysis']
                print(f"⚡ Energy Decay Analysis:")
                print(f"  Training decay exponent: {decay['training']['decay_exponent']:.3f}")
                print(f"  Predicted decay exponent: {decay['predicted']['decay_exponent']:.3f}")
                print(f"  Difference: {decay['decay_exponent_difference']:.3f}")
                print(f"  Physical range (1.0-1.5): {'✅' if decay['is_physical'] else '❌'}")
                print()
            
            # Kolmogorov scaling
            if 'kolmogorov_analysis' in results:
                kolm = results['kolmogorov_analysis']
                print(f"🌪️  Kolmogorov Scaling Analysis:")
                print(f"  Training inertial slope: {kolm['training']['inertial_slope']:.3f}")
                print(f"  Predicted inertial slope: {kolm['predicted']['inertial_slope']:.3f}")
                print(f"  Theoretical slope: -1.667")
                print(f"  Slope difference: {kolm['slope_difference']:.3f}")
                print()
            
            # Velocity statistics
            if 'velocity_statistics' in results:
                stats = results['velocity_statistics']
                print(f"📈 Velocity Statistics:")
                print(f"  Skewness difference: {stats['skewness_difference']:.3f}")
                print(f"  Kurtosis difference: {stats['kurtosis_difference']:.3f}")
                print()
            
            # Integral scales
            if 'integral_scales' in results:
                scales = results['integral_scales']
                print(f"📏 Integral Length Scales:")
                print(f"  Training: {scales['training_length_scale']:.3f}")
                print(f"  Predicted: {scales['predicted_length_scale']:.3f}")
                print(f"  Ratio: {scales['length_scale_ratio']:.3f}")
                print()
            
            # Traditional metrics (with warning)
            if 'traditional_metrics' in results:
                trad = results['traditional_metrics']
                print(f"⚠️  Traditional Metrics (chaos-sensitive):")
                print(f"  RMSE U: {trad['rmse_u']:.6f}")
                print(f"  RMSE V: {trad['rmse_v']:.6f}")
                print(f"  Note: {trad['note']}")
                print()
            
        except Exception as e:
            print(f"❌ Error processing {resolution}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    combined_results_file = os.path.join(results_dir, 'all_refined_results.json')
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Refined analysis complete!")
    print(f"📁 Results saved in: {results_dir}/")
    print(f"📄 Combined results: {combined_results_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 