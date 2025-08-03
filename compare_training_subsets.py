#!/usr/bin/env python3
"""Compare first 50 steps of training data with another subset for validation."""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import json

def load_training_subsets(training_file: str, subset1_start: int = 0, subset2_start: int = 50, num_steps: int = 50):
    """Load two subsets of training data for comparison."""
    print(f"Loading training data: {training_file}")
    data = np.load(training_file)
    
    # Extract velocity fields
    u_full = data['u']  # Shape: (12200, 64, 64)
    v_full = data['v']  # Shape: (12200, 64, 64)
    
    # Extract two subsets
    u_subset1 = u_full[subset1_start:subset1_start + num_steps]
    v_subset1 = v_full[subset1_start:subset1_start + num_steps]
    u_subset2 = u_full[subset2_start:subset2_start + num_steps]
    v_subset2 = v_full[subset2_start:subset2_start + num_steps]
    
    print(f"Subset 1 (steps {subset1_start}-{subset1_start + num_steps}): {u_subset1.shape}")
    print(f"Subset 2 (steps {subset2_start}-{subset2_start + num_steps}): {u_subset2.shape}")
    
    return {
        'ref_u': u_subset1,  # Reference (first 50 steps)
        'ref_v': v_subset1,
        'test_u': u_subset2, # Test (next 50 steps)
        'test_v': v_subset2,
        'resolution': data['resolution'],
        'time_steps': num_steps
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
    kx = jnp.fft.fftfreq(nx, 1.0)
    ky = jnp.fft.fftfreq(ny, 1.0)
    
    # Create 2D wavenumber grid
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
    k_magnitude = jnp.sqrt(kx_grid**2 + ky_grid**2)
    
    # Radial averaging to get 1D spectrum
    k_max = int(jnp.max(k_magnitude))
    k_bins = jnp.arange(0, k_max + 1)
    energy_spectrum = jnp.zeros((energy_density.shape[0], len(k_bins) - 1))
    
    for i in range(len(k_bins) - 1):
        mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
        energy_spectrum = energy_spectrum.at[:, i].set(
            jnp.mean(energy_density * mask[None, :, :], axis=(-2, -1))
        )
    
    return energy_spectrum

def apply_metrics(data):
    """Apply validation metrics to compare two training subsets."""
    results = {}
    
    ref_u = data['ref_u']
    ref_v = data['ref_v']
    test_u = data['test_u']
    test_v = data['test_v']
    
    print("Computing validation metrics...")
    
    # 1. Energy Spectrum Comparison
    print("  1. Energy spectrum...")
    ref_spectrum = compute_energy_spectrum(ref_u, ref_v)
    test_spectrum = compute_energy_spectrum(test_u, test_v)
    
    threshold = 0.01
    valid_mask = (ref_spectrum > threshold) & (test_spectrum > threshold)
    log_diff = jnp.abs(jnp.log(test_spectrum) - jnp.log(ref_spectrum))
    energy_metric = jnp.mean(jnp.where(valid_mask, log_diff, 0))
    results['energy_spectrum_metric'] = float(energy_metric)
    
    # 2. Spatial Statistics
    print("  2. Spatial statistics...")
    results['ref_u_mean'] = float(jnp.mean(ref_u))
    results['test_u_mean'] = float(jnp.mean(test_u))
    results['ref_u_std'] = float(jnp.std(ref_u))
    results['test_u_std'] = float(jnp.std(test_u))
    
    # 3. Basic Error Metrics
    print("  3. Error metrics...")
    results['rmse_u'] = float(jnp.sqrt(jnp.mean((ref_u - test_u)**2)))
    results['rmse_v'] = float(jnp.sqrt(jnp.mean((ref_v - test_v)**2)))
    results['mae_u'] = float(jnp.mean(jnp.abs(ref_u - test_u)))
    results['mae_v'] = float(jnp.mean(jnp.abs(ref_v - test_v)))
    
    return results

def create_comparison_plots(data, results, resolution, save_dir):
    """Create comparison plots."""
    fig = plt.figure(figsize=(15, 10))
    
    # Energy spectrum comparison
    plt.subplot(2, 3, 1)
    ref_spectrum = compute_energy_spectrum(data['ref_u'], data['ref_v'])
    test_spectrum = compute_energy_spectrum(data['test_u'], data['test_v'])
    
    k_values = jnp.arange(ref_spectrum.shape[1])
    plt.loglog(k_values[1:], jnp.mean(ref_spectrum, axis=0)[1:], 'b-', label='Steps 0-50', linewidth=2)
    plt.loglog(k_values[1:], jnp.mean(test_spectrum, axis=0)[1:], 'r--', label='Steps 50-100', linewidth=2)
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy E(k)')
    plt.title('Energy Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Field snapshots
    plt.subplot(2, 3, 2)
    plt.imshow(data['ref_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar()
    plt.title('Reference U-field (t=0)')
    
    plt.subplot(2, 3, 3)
    plt.imshow(data['test_u'][0], cmap='RdBu_r', origin='lower')
    plt.colorbar()
    plt.title('Test U-field (t=50)')
    
    # Time evolution comparison
    plt.subplot(2, 3, 4)
    time_steps = jnp.arange(data['time_steps'])
    ref_energy = jnp.mean(data['ref_u']**2 + data['ref_v']**2, axis=(-2, -1))
    test_energy = jnp.mean(data['test_u']**2 + data['test_v']**2, axis=(-2, -1))
    plt.plot(time_steps, ref_energy, 'b-', label='Steps 0-50', linewidth=2)
    plt.plot(time_steps, test_energy, 'r-', label='Steps 50-100', linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Kinetic Energy')
    plt.title('Energy Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics bar chart
    plt.subplot(2, 3, 5)
    metrics = ['RMSE U', 'RMSE V', 'MAE U', 'MAE V']
    values = [results['rmse_u'], results['rmse_v'], results['mae_u'], results['mae_v']]
    plt.bar(metrics, values, color=['lightblue', 'lightblue', 'lightgreen', 'lightgreen'])
    plt.ylabel('Error Value')
    plt.title('Error Metrics')
    plt.xticks(rotation=45)
    
    # Statistics comparison
    plt.subplot(2, 3, 6)
    stats = ['Ref U std', 'Test U std']
    values = [results['ref_u_std'], results['test_u_std']]
    plt.bar(stats, values, color=['blue', 'red'])
    plt.ylabel('Standard Deviation')
    plt.title('Statistical Comparison')
    
    plt.suptitle(f'Training Data Subset Comparison - {resolution}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'training_subset_comparison_{resolution}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {save_path}")

def main():
    """Main function."""
    results_dir = "training_subset_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    resolutions = ['64x64', '128x128']
    
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution: {resolution}")
        print(f"{'='*60}")
        
        training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
        
        if not os.path.exists(training_file):
            print(f"Training file not found: {training_file}")
            continue
        
        try:
            # Load two subsets of training data
            data = load_training_subsets(training_file, subset1_start=0, subset2_start=50, num_steps=50)
            
            # Apply metrics
            results = apply_metrics(data)
            
            # Create plots
            create_comparison_plots(data, results, resolution, results_dir)
            
            # Save results
            results_file = os.path.join(results_dir, f'results_{resolution}.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print summary
            print(f"\n📊 Validation Results for {resolution}:")
            print(f"  Energy Spectrum Metric: {results['energy_spectrum_metric']:.6f}")
            print(f"  RMSE U: {results['rmse_u']:.6f}")
            print(f"  RMSE V: {results['rmse_v']:.6f}")
            print(f"  Reference U std: {results['ref_u_std']:.6f}")
            print(f"  Test U std: {results['test_u_std']:.6f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()