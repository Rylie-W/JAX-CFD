#!/usr/bin/env python3
"""Data comparison analysis between training_data and pict_data using selected evaluation metrics."""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
import os
import json
from pathlib import Path

# Import the evaluation functions from the JAX-CFD library
import sys
sys.path.append('jax-cfd')
from jax_cfd.data.evaluation import (
    energy_spectrum_metric,
    u_x_correlation_metric, 
    temporal_autocorrelation,
    u_t_correlation_metric
)

def load_data_file(filepath: str) -> Dict:
    """Load and examine a .npz data file."""
    print(f"Loading {filepath}...")
    data = np.load(filepath)
    print(f"Keys in file: {list(data.keys())}")
    
    # Print shapes of all arrays
    for key in data.keys():
        print(f"  {key}: shape {data[key].shape}, dtype {data[key].dtype}")
    
    return dict(data)

def convert_to_xarray(data: Dict, data_type: str) -> xr.Dataset:
    """Convert numpy arrays to xarray Dataset with proper dimensions."""
    # This function needs to be adapted based on the actual data structure
    # For now, let's create a basic structure assuming common CFD data format
    
    # Assume the data has velocity components and is organized as (time, space_dims...)
    dataset_vars = {}
    
    for key, array in data.items():
        if len(array.shape) >= 2:  # Assuming time series data
            if len(array.shape) == 3:  # (time, x, y) or similar
                dims = ['time', 'x', 'y'] if array.shape[2] > 1 else ['time', 'x']
            elif len(array.shape) == 2:  # (time, x)
                dims = ['time', 'x']
            else:
                dims = [f'dim_{i}' for i in range(len(array.shape))]
            
            # Create coordinates
            coords = {}
            for i, dim in enumerate(dims):
                coords[dim] = np.arange(array.shape[i])
            
            dataset_vars[key] = xr.DataArray(
                array, 
                dims=dims,
                coords=coords
            )
    
    ds = xr.Dataset(dataset_vars)
    ds.attrs['data_type'] = data_type
    return ds

def create_combined_dataset(training_ds: xr.Dataset, pict_ds: xr.Dataset) -> xr.Dataset:
    """Combine training and predicted datasets for comparison."""
    # Add model dimension to distinguish between datasets
    training_ds_expanded = training_ds.expand_dims('model')
    pict_ds_expanded = pict_ds.expand_dims('model')
    
    # Concatenate along model dimension
    combined = xr.concat([training_ds_expanded, pict_ds_expanded], dim='model')
    combined.coords['model'] = ['ground_truth', 'predicted']
    
    return combined

def compute_energy_spectrum(velocity_data: xr.DataArray) -> xr.DataArray:
    """Compute energy spectrum from velocity data."""
    # This is a simplified energy spectrum computation
    # In practice, you might need to use the proper FFT-based method
    fft_data = jnp.fft.fft(velocity_data, axis=-1)
    energy_spectrum = jnp.abs(fft_data)**2
    
    # Create wavenumber coordinate
    n_points = velocity_data.shape[-1]
    kx = jnp.arange(n_points//2 + 1)
    
    return xr.DataArray(
        energy_spectrum[..., :len(kx)],
        dims=list(velocity_data.dims[:-1]) + ['kx'],
        coords={**velocity_data.coords, 'kx': kx}
    )

def apply_evaluation_metrics(combined_ds: xr.Dataset) -> Dict:
    """Apply the four selected evaluation metrics."""
    results = {}
    
    # Assume 'u' is the main velocity component
    if 'u' in combined_ds.data_vars:
        velocity_data = combined_ds['u']
        
        # 1. Energy spectrum metric
        print("Computing energy spectrum metric...")
        energy_spec = compute_energy_spectrum(velocity_data)
        energy_metric_fn = energy_spectrum_metric(threshold=0.01)
        energy_result = energy_metric_fn(
            energy_spec.sel(model='predicted'),
            energy_spec.sel(model='ground_truth')
        )
        results['energy_spectrum_metric'] = float(energy_result)
        
        # 2. Spatial correlation metric (u_x_correlation_metric)
        print("Computing spatial correlation metric...")
        if 'x' in velocity_data.dims:
            ux_metric_fn = u_x_correlation_metric(threshold=0.5)
            # This requires spatial correlation data - simplified for now
            spatial_corr_result = ux_metric_fn(
                velocity_data.sel(model='predicted'),
                velocity_data.sel(model='ground_truth')
            )
            results['u_x_correlation_metric'] = float(spatial_corr_result)
        
        # 3. Temporal autocorrelation
        print("Computing temporal autocorrelation...")
        if 'time' in velocity_data.dims:
            temp_autocorr = temporal_autocorrelation(combined_ds)
            results['temporal_autocorrelation'] = temp_autocorr
        
        # 4. Temporal correlation metric
        print("Computing temporal correlation metric...")
        if 'time' in velocity_data.dims:
            ut_metric_fn = u_t_correlation_metric(threshold=0.5)
            temporal_result = ut_metric_fn(
                jnp.array(velocity_data.sel(model='predicted')),
                jnp.array(velocity_data.sel(model='ground_truth'))
            )
            results['u_t_correlation_metric'] = float(temporal_result)
    
    return results

def visualize_results(results: Dict, resolution: str, save_dir: str):
    """Create visualizations of the comparison results."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Data Comparison Analysis - Resolution: {resolution}', fontsize=16)
    
    # Plot 1: Energy spectrum metric
    if 'energy_spectrum_metric' in results:
        axes[0, 0].bar(['Energy Spectrum Metric'], [results['energy_spectrum_metric']])
        axes[0, 0].set_title('Energy Spectrum Metric')
        axes[0, 0].set_ylabel('Metric Value')
    
    # Plot 2: Spatial correlation metric
    if 'u_x_correlation_metric' in results:
        axes[0, 1].bar(['Spatial Correlation Metric'], [results['u_x_correlation_metric']])
        axes[0, 1].set_title('U-X Correlation Metric')
        axes[0, 1].set_ylabel('Metric Value')
    
    # Plot 3: Temporal correlation metric
    if 'u_t_correlation_metric' in results:
        axes[1, 0].bar(['Temporal Correlation Metric'], [results['u_t_correlation_metric']])
        axes[1, 0].set_title('U-T Correlation Metric')
        axes[1, 0].set_ylabel('Metric Value')
    
    # Plot 4: Summary metrics
    metric_names = []
    metric_values = []
    for key, value in results.items():
        if isinstance(value, (int, float)):
            metric_names.append(key.replace('_', ' ').title())
            metric_values.append(value)
    
    if metric_names:
        axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_title('All Metrics Summary')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f'comparison_analysis_{resolution}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

def main():
    """Main function to run the data comparison analysis."""
    # Create results directory
    results_dir = "comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define data files to compare
    resolutions = ['64x64', '128x128', '256x256', '512x512']
    
    all_results = {}
    
    for resolution in resolutions:
        print(f"\n{'='*50}")
        print(f"Processing resolution: {resolution}")
        print(f"{'='*50}")
        
        # Define file paths
        training_file = f"data/training_data/decaying_turbulence_v2_{resolution}_index_1.npz"
        pict_file = f"data/pict_data/decaying_turbulence_{resolution}_index_1.npz"
        
        # Check if files exist
        if not os.path.exists(training_file):
            print(f"Training file not found: {training_file}")
            continue
        if not os.path.exists(pict_file):
            print(f"Predicted file not found: {pict_file}")
            continue
        
        try:
            # Load data files
            print("Loading training data...")
            training_data = load_data_file(training_file)
            print("Loading predicted data...")
            pict_data = load_data_file(pict_file)
            
            # Convert to xarray datasets
            print("Converting to xarray datasets...")
            training_ds = convert_to_xarray(training_data, 'training')
            pict_ds = convert_to_xarray(pict_data, 'predicted')
            
            # Create combined dataset
            print("Creating combined dataset...")
            combined_ds = create_combined_dataset(training_ds, pict_ds)
            
            # Apply evaluation metrics
            print("Applying evaluation metrics...")
            results = apply_evaluation_metrics(combined_ds)
            
            # Store results
            all_results[resolution] = results
            
            # Save individual results
            results_file = os.path.join(results_dir, f'results_{resolution}.json')
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        json_results[key] = value
                    else:
                        json_results[key] = str(type(value))  # Just store type for complex objects
                json.dump(json_results, f, indent=2)
            
            # Create visualizations
            print("Creating visualizations...")
            visualize_results(results, resolution, results_dir)
            
            print(f"Results for {resolution}:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.6f}")
                else:
                    print(f"  {metric}: {type(value)}")
            
        except Exception as e:
            print(f"Error processing {resolution}: {str(e)}")
            continue
    
    # Save combined results
    combined_results_file = os.path.join(results_dir, 'all_results.json')
    with open(combined_results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_all_results = {}
        for resolution, results in all_results.items():
            json_all_results[resolution] = {}
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    json_all_results[resolution][key] = value
                else:
                    json_all_results[resolution][key] = str(type(value))
        json.dump(json_all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"Results saved in: {results_dir}/")
    print(f"Combined results: {combined_results_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 