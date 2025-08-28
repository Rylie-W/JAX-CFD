import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.ml as cfd_ml
import numpy as np
import seaborn
import xarray
import matplotlib.pyplot as plt
import logging
import argparse
import os


def get_dt(args, size, cl):
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                              (0, 2 * jnp.pi * cl * args.domain_scale)))
  # Choose a time step.
  dt = cfd.equations.stable_time_step(
      args.max_velocity, args.cfl_safety_factor, args.viscosity / cl, grid)
  return dt


def get_trajectory(args, size, rng=None, outer_steps=50, v0=None):
  cl = args.characteristic_length
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                              (0, 2 * jnp.pi * cl * args.domain_scale)))

  # Choose a time step.
  dt = get_dt(args, size, cl)
  delta_t = get_dt(args, args.low_res, cl)
  inner_steps = round(delta_t / dt)
  logging.info("inner step %d" % inner_steps)

  # Define the physical dimensions of the simulation.

  if args.decay:
    forcing = None
  else:
    forcing = cfd_ml.forcings.kolmogorov_forcing(grid,
                                                 args.forcing_scale / cl,
                                                 args.peak_wavenumber / cl,
                                                 -0.1 / cl)

  # Construct a random initial velocity. The `filtered_velocity_field` function
  # ensures that the initial velocity is divergence free and it filters out
  # high frequency fluctuations.
  if v0 is None:
    v0 = cfd.initial_conditions.filtered_velocity_field(rng, grid, args.max_velocity,
                                                        args.peak_wavenumber / cl)
  elif size < args.high_res:
    large_grid = cfd.grids.Grid((args.high_res, args.high_res), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                                                        (0, 2 * jnp.pi * cl * args.domain_scale)))
    v0 = cfd.resize.downsample_staggered_velocity(large_grid, grid, v0)

  # Define a step function and use it to compute a trajectory.
  step_fn = cfd.funcutils.repeated(
      cfd.equations.semi_implicit_navier_stokes(
          density=args.density, viscosity=args.viscosity / cl,
          dt=dt, grid=grid, forcing=forcing),
      steps=inner_steps)

  # trajectory_fn = cfd.funcutils.trajectory(step_fn, outer_steps)
  # _, trajectory = trajectory_fn(v0)

  rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
  _, trajectory = jax.device_get(rollout_fn(v0))
  return trajectory


def load_initial_velocity_from_file(filepath: str, target_size: int, args) -> tuple:
  """
  Load initial velocity field from a .npz file and prepare it for simulation.
  The output structure exactly matches what filtered_velocity_field produces.
  
  Args:
    filepath: Path to the .npz file containing velocity data
    target_size: Target resolution for the simulation
    args: Simulation arguments
    
  Returns:
    List of GridVariable objects compatible with filtered_velocity_field output
  """
  import os
  
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Initial velocity file not found: {filepath}")
  
  # Load the data
  data = np.load(filepath)
  u_data = data['u']
  v_data = data['v']
  
  logging.info(f"Loaded initial velocity from: {filepath}")
  logging.info(f"Original data shape: u={u_data.shape}, v={v_data.shape}")
  
  # Use the first timestep as initial condition
  if len(u_data.shape) == 3:  # (time, x, y)
    u_initial = u_data[0] 
    v_initial = v_data[0]
  else:  # (x, y)
    u_initial = u_data
    v_initial = v_data
  
  original_size = u_initial.shape[0]
  logging.info(f"Using initial condition shape: {u_initial.shape}")
  
  # Create the target grid (this is what we want)
  cl = args.characteristic_length
  target_grid = cfd.grids.Grid((target_size, target_size), 
                              domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                     (0, 2 * jnp.pi * cl * args.domain_scale)))
  
  # If sizes don't match, we need to resize the data first
  if target_size != original_size:
    logging.info(f"Resizing from {original_size}x{original_size} to {target_size}x{target_size}")
    
    # Create temporary grid for original data
    original_grid = cfd.grids.Grid((original_size, original_size), 
                                  domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                         (0, 2 * jnp.pi * cl * args.domain_scale)))
    
    # Create temporary GridArrays
    u_temp = cfd.grids.GridArray(jnp.array(u_initial), offset=(0.5, 0.0), grid=original_grid)
    v_temp = cfd.grids.GridArray(jnp.array(v_initial), offset=(0.0, 0.5), grid=original_grid)
    
    # Resize
    if target_size < original_size:
      u_array, v_array = cfd.resize.downsample_staggered_velocity(original_grid, target_grid, [u_temp, v_temp])
    else:
      u_array, v_array = cfd.resize.upsample_staggered_velocity(original_grid, target_grid, [u_temp, v_temp])
  else:
    # Same size - create GridArrays directly
    u_array = cfd.grids.GridArray(jnp.array(u_initial), offset=(0.5, 0.0), grid=target_grid)
    v_array = cfd.grids.GridArray(jnp.array(v_initial), offset=(0.0, 0.5), grid=target_grid)
  
  # Create a structure that exactly matches filtered_velocity_field output
  # Use the same approach as filtered_velocity_field but with our data
  u_final = cfd.grids.GridVariable(
    u_array, 
    bc=cfd.boundaries.periodic_boundary_conditions(target_grid.ndim)
  )
  v_final = cfd.grids.GridVariable(
    v_array,
    bc=cfd.boundaries.periodic_boundary_conditions(target_grid.ndim)
  )
  
  # Wrap in tuple exactly like filtered_velocity_field does
  result = (u_final, v_final)
  
  logging.info(f"Final initial velocity shape: u={u_final.shape}, v={v_final.shape}")
  logging.info(f"Final structure matches filtered_velocity_field: ✅")
  
  return result


def create_initial_velocity_verification_plot(original_file: str, warmup_first_step: tuple, args, save_path: str = None):
  """
  Create a comparison plot between original PICT data and warmup first step to verify initialization.
  
  Args:
    original_file: Path to the original .npz file  
    warmup_first_step: Tuple of (u, v) GridArrays from first warmup step
    args: Simulation arguments
    save_path: Where to save the plot (optional)
  """
  
  # Load original data
  orig_data = np.load(original_file)
  orig_u = orig_data['u']
  orig_v = orig_data['v']
  
  # Use first timestep as specified by user's modification
  if len(orig_u.shape) == 3:
    orig_u_2d = orig_u[0]  # First timestep
    orig_v_2d = orig_v[0]
  else:
    orig_u_2d = orig_u
    orig_v_2d = orig_v
    
  # Get warmup data (now always GridVariable structure)
  warmup_u_2d = np.array(warmup_first_step[0].array.data)
  warmup_v_2d = np.array(warmup_first_step[1].array.data)
  
  # Create comparison figure
  fig, axes = plt.subplots(2, 3, figsize=(18, 12))
  
  # Compute vorticity for better visualization
  def compute_vorticity(u, v):
    # Simple finite difference for vorticity
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    return dv_dx - du_dy
  
  orig_vorticity = compute_vorticity(orig_u_2d, orig_v_2d)
  warmup_vorticity = compute_vorticity(warmup_u_2d, warmup_v_2d)
  
  # Original data plots
  im1 = axes[0, 0].imshow(orig_u_2d, cmap='RdBu_r', origin='lower')
  axes[0, 0].set_title('Original PICT: U velocity')
  axes[0, 0].set_xlabel('x')
  axes[0, 0].set_ylabel('y')
  plt.colorbar(im1, ax=axes[0, 0])
  
  im2 = axes[0, 1].imshow(orig_v_2d, cmap='RdBu_r', origin='lower')
  axes[0, 1].set_title('Original PICT: V velocity')
  axes[0, 1].set_xlabel('x')
  axes[0, 1].set_ylabel('y')
  plt.colorbar(im2, ax=axes[0, 1])
  
  im3 = axes[0, 2].imshow(orig_vorticity, cmap='RdBu_r', origin='lower')
  axes[0, 2].set_title('Original PICT: Vorticity')
  axes[0, 2].set_xlabel('x')
  axes[0, 2].set_ylabel('y')
  plt.colorbar(im3, ax=axes[0, 2])
  
  # Warmup first step plots
  im4 = axes[1, 0].imshow(warmup_u_2d, cmap='RdBu_r', origin='lower')
  axes[1, 0].set_title('Warmup First Step: U velocity')
  axes[1, 0].set_xlabel('x')
  axes[1, 0].set_ylabel('y')
  plt.colorbar(im4, ax=axes[1, 0])
  
  im5 = axes[1, 1].imshow(warmup_v_2d, cmap='RdBu_r', origin='lower')
  axes[1, 1].set_title('Warmup First Step: V velocity')
  axes[1, 1].set_xlabel('x')
  axes[1, 1].set_ylabel('y')
  plt.colorbar(im5, ax=axes[1, 1])
  
  im6 = axes[1, 2].imshow(warmup_vorticity, cmap='RdBu_r', origin='lower')
  axes[1, 2].set_title('Warmup First Step: Vorticity')
  axes[1, 2].set_xlabel('x')
  axes[1, 2].set_ylabel('y')
  plt.colorbar(im6, ax=axes[1, 2])
  
  # Add statistical comparison
  u_diff = np.mean((orig_u_2d - warmup_u_2d)**2)**0.5
  v_diff = np.mean((orig_v_2d - warmup_v_2d)**2)**0.5
  vort_diff = np.mean((orig_vorticity - warmup_vorticity)**2)**0.5
  
  # Add text with statistics
  stats_text = f"""
Initial Velocity Verification:
• U velocity RMSE: {u_diff:.6f}
• V velocity RMSE: {v_diff:.6f}
• Vorticity RMSE: {vort_diff:.6f}
• Original shape: {orig_u_2d.shape}
• Warmup shape: {warmup_u_2d.shape}
• Source file: {os.path.basename(original_file)}
"""
  
  plt.figtext(0.02, 0.02, stats_text, fontsize=10, family='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
  
  plt.suptitle('Initial Velocity Verification: PICT vs Warmup First Step', fontsize=16, weight='bold')
  plt.tight_layout()
  
  # Save the plot
  if save_path is None:
    save_path = f'../verification_plots/initial_velocity_verification_{args.save_index}.png'
  
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.close()
  
  logging.info(f"✅ Verification plot saved: {save_path}")
  logging.info(f"📊 Verification statistics:")
  logging.info(f"   U velocity RMSE: {u_diff:.6f}")
  logging.info(f"   V velocity RMSE: {v_diff:.6f}")
  logging.info(f"   Vorticity RMSE: {vort_diff:.6f}")
  
  return {
    'u_rmse': u_diff,
    'v_rmse': v_diff, 
    'vorticity_rmse': vort_diff,
    'verification_plot': save_path
  }


def get_trajectory_subsampled(args, size, rng=None, total_steps=50, save_interval=1000, subsample_interval=1, v0=None, save_dir=None, base_filename=None):
  """
  Generate trajectory data in segments with subsampling to avoid memory issues.
  Saves data every save_interval steps, but only stores every subsample_interval timestep.
  """
  cl = args.characteristic_length
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                              (0, 2 * jnp.pi * cl * args.domain_scale)))

  # Choose a time step.
  dt = get_dt(args, size, cl)
  delta_t = get_dt(args, args.low_res, cl)
  inner_steps = round(delta_t / dt)
  logging.info("inner step %d" % inner_steps)

  # Define the physical dimensions of the simulation.
  if args.decay:
    forcing = None
  else:
    forcing = cfd_ml.forcings.kolmogorov_forcing(grid,
                                                 args.forcing_scale / cl,
                                                 args.peak_wavenumber / cl,
                                                 -0.1 / cl)

  # Construct a random initial velocity.
  if v0 is None:
    v0 = cfd.initial_conditions.filtered_velocity_field(rng, grid, args.max_velocity,
                                                        args.peak_wavenumber / cl)
  elif size < args.high_res:
    large_grid = cfd.grids.Grid((args.high_res, args.high_res), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                                                        (0, 2 * jnp.pi * cl * args.domain_scale)))
    v0 = cfd.resize.downsample_staggered_velocity(large_grid, grid, v0)

  # Define a step function
  step_fn = cfd.funcutils.repeated(
      cfd.equations.semi_implicit_navier_stokes(
          density=args.density, viscosity=args.viscosity / cl,
          dt=dt, grid=grid, forcing=forcing),
      steps=inner_steps)

  # Initialize current velocity state
  current_v = v0
  segment_count = 0
  saved_segments = []
  all_subsampled_u = []
  all_subsampled_v = []
  global_step = 0
  
  # Process in segments
  remaining_steps = total_steps
  while remaining_steps > 0:
    # Determine how many steps to compute in this segment
    segment_steps = min(save_interval, remaining_steps)
    
    logging.info(f"Computing segment {segment_count}: {segment_steps} steps (remaining: {remaining_steps})")
    
    # Compute trajectory for this segment
    rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, segment_steps))
    _, trajectory_segment = jax.device_get(rollout_fn(current_v))
    
    # Subsample the trajectory segment
    # Determine which timesteps to keep in this segment
    segment_indices = []
    for i in range(segment_steps):
      if (global_step + i) % subsample_interval == 0:
        segment_indices.append(i)
    
    if segment_indices:
      # Extract subsampled data
      subsampled_u = trajectory_segment[0].data[segment_indices]
      subsampled_v = trajectory_segment[1].data[segment_indices]
      all_subsampled_u.append(subsampled_u)
      all_subsampled_v.append(subsampled_v)
      
      logging.info(f"Segment {segment_count}: kept {len(segment_indices)} out of {segment_steps} timesteps")
    
    # Update state for next segment (use last timestep as initial condition)
    current_v = [trajectory_segment[0][-1:], trajectory_segment[1][-1:]]
    current_v[0].array.data = current_v[0].array.data[0]  # Remove time dimension
    current_v[1].array.data = current_v[1].array.data[0]  # Remove time dimension
    
    global_step += segment_steps
    segment_count += 1
    remaining_steps -= segment_steps
  
  # Combine all subsampled data
  if all_subsampled_u and save_dir and base_filename:
    final_u = np.concatenate(all_subsampled_u, axis=0)
    final_v = np.concatenate(all_subsampled_v, axis=0)
    
    # Save the complete subsampled trajectory
    data_file = f'{save_dir}/{base_filename}_{size}x{size}_index_{args.save_index}.npz'
    logging.info(f"Saving subsampled trajectory to: {data_file}")
    logging.info(f"Original steps: {total_steps}, Subsampled steps: {final_u.shape[0]} (every {subsample_interval} steps)")
    
    # Save with metadata
    np.savez_compressed(data_file,
                       u=final_u,
                       v=final_v,
                       resolution=size,
                       total_steps=total_steps,
                       saved_steps=final_u.shape[0],
                       subsample_interval=subsample_interval,
                       warmup_time=args.warmup_time,
                       max_velocity=args.max_velocity,
                       viscosity=args.viscosity,
                       decay=args.decay,
                       seed=args.seed,
                       characteristic_length=args.characteristic_length,
                       domain_scale=args.domain_scale,
                       low_res=args.low_res,
                       cfl_safety_factor=args.cfl_safety_factor,
                       density=args.density,
                       forcing_scale=args.forcing_scale,
                       peak_wavenumber=args.peak_wavenumber,
                       timestep=get_dt(args, args.low_res, args.characteristic_length))
    
    saved_segments.append(data_file)
  
  logging.info(f"Completed subsampled generation: {len(all_subsampled_u)} segments processed")
  return saved_segments


def plot_trajectory(args, size, trajectory, file_name):
  cl = args.characteristic_length
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                              (0, 2 * jnp.pi * cl * args.domain_scale)))

  x_len = grid.axes()[0].shape[0]
  x = 2 * np.double(grid.axes()[0]).mean() / x_len * np.arange(x_len)

  y_len = grid.axes()[1].shape[0]
  y = 2 * np.double(grid.axes()[1]).mean() / y_len * np.arange(y_len)

  # load into xarray for visualization and analysis
  delta_t = get_dt(args, args.low_res, cl)
  ds = xarray.Dataset(
      {
          'u': (('time', 'x', 'y'), trajectory[0].data),
          'v': (('time', 'x', 'y'), trajectory[1].data),
      },
      coords={
          'x': x,
          'y': y,
          'time': (delta_t * np.arange(trajectory[0].shape[0]))
      }
  )

  def vorticity(ds):
    x = (ds.v.differentiate('x') - ds.u.differentiate('y'))
    x = x.rename('vorticity')
    return x

  (ds.pipe(vorticity).thin(time=args.demo_steps // 5).transpose()
   .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))

  plt.savefig(file_name)


def main(args):
  logger = logging.getLogger()
  logger.setLevel("INFO")
  seed = args.seed
  warmup_time = args.warmup_time
  outer_steps = args.outer_steps
  rng = jax.random.PRNGKey(seed)

  for iter in range(args.iters):
    rng, subrng = jax.random.split(rng)

    delta_t = get_dt(args, args.low_res, args.characteristic_length)
    warm_up_step = round(warmup_time / delta_t)

    count = 0
    warmup_result = None
    
    # Check if initial velocity file is provided
    if args.initial_velocity_file:
      logger.info(f"Loading initial velocity from: {args.initial_velocity_file}")
      warmup_result = load_initial_velocity_from_file(args.initial_velocity_file, args.high_res, args)
      logger.info("Using loaded initial velocity as warmup starting point")
      
      # Generate verification plot if requested
      if args.verify_initial_velocity:
        logger.info("Generating initial velocity verification plot...")
        verification_result = create_initial_velocity_verification_plot(
          args.initial_velocity_file, warmup_result, args)
        logger.info(f"Verification plot created: {verification_result['verification_plot']}")
    
    # Run warmup simulation (compute only, no saving)
    while count + outer_steps <= warm_up_step:
      logger.info(f"Warmup step {count} of {warm_up_step}")
      if warmup_result is not None:
        # Check if this is trajectory output (has time dimension) and extract last timestep
        if len(warmup_result[0].array.data.shape) > 2:  # Has time dimension
          warmup_result[0].array.data = warmup_result[0].array.data[-1]
          warmup_result[1].array.data = warmup_result[1].array.data[-1]
      warmup_result = get_trajectory(args, size=args.high_res, rng=subrng,
                                     outer_steps=outer_steps, v0=warmup_result)
      count += outer_steps

    if warm_up_step > count:
      if warmup_result is not None:
        # Check if this is trajectory output (has time dimension) and extract last timestep
        if len(warmup_result[0].array.data.shape) > 2:  # Has time dimension
          warmup_result[0].array.data = warmup_result[0].array.data[-1]
          warmup_result[1].array.data = warmup_result[1].array.data[-1]
      final_warmup = get_trajectory(args, size=args.high_res, rng=subrng,
                                   outer_steps=warm_up_step - count, v0=warmup_result)
      warmup_result = final_warmup
    
    logger.info("Warmup completed - no warmup data saved")

    # Extract final state for subsequent generation
    if warmup_result is not None:
      # Always extract last timestep as warmup_result is trajectory output here
      warmup_result[0].array.data = warmup_result[0].array.data[-1]
      warmup_result[1].array.data = warmup_result[1].array.data[-1]

    resolution_list = []
    res = args.low_res
    while res <= args.high_res:
      resolution_list.append(res)
      res *= 2

    if args.demo:
      # Demo mode: create visualizations with demo_steps
      for resolution in resolution_list:
        trajectory = get_trajectory(args, size=resolution, outer_steps=args.demo_steps, v0=warmup_result)
        file_name = f'../figs/%s_demo_{resolution}x{resolution}.png' % args.save_file
        logger.info(file_name)
        plot_trajectory(args, resolution, trajectory, file_name)
    else:
      # Training data mode: generate full datasets with generate_steps
      for resolution in resolution_list:
        trajectory = get_trajectory(args, size=resolution, outer_steps=args.generate_steps, v0=warmup_result)
        
        save_dir = f'../data/training_data/{resolution}'
        os.makedirs(save_dir, exist_ok=True)
        
        data_file = f'{save_dir}/{args.save_file}_{resolution}x{resolution}_index_{args.save_index}.npz'
        logger.info(f"Saving training data to: {data_file}")
        
        # Save velocity components and ALL metadata needed for plot_trajectory
        np.savez_compressed(data_file,
                           u=trajectory[0].data,  # x-velocity component
                           v=trajectory[1].data,  # y-velocity component
                           resolution=resolution,
                           outer_steps=args.generate_steps,
                           warmup_time=args.warmup_time,
                           max_velocity=args.max_velocity,
                           viscosity=args.viscosity,
                           decay=args.decay,
                           seed=args.seed,
                           # Additional args needed for plot_trajectory
                           characteristic_length=args.characteristic_length,
                           domain_scale=args.domain_scale,
                           low_res=args.low_res,
                           cfl_safety_factor=args.cfl_safety_factor,
                           density=args.density,
                           forcing_scale=args.forcing_scale,
                           peak_wavenumber=args.peak_wavenumber,
                           timestep = get_dt(args, args.low_res, args.characteristic_length))
        
        logger.info(f"Saved trajectory shape: {trajectory[0].data.shape}")
        logger.info(f"Resolution: {resolution}x{resolution}, Steps: {args.generate_steps}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--iters', type=int, default=1)
  parser.add_argument('--outer_steps', type=int, default=50)
  parser.add_argument('--demo_steps', type=int, default=50)
  parser.add_argument('--generate_steps', type=int, default=50)
  parser.add_argument('--warmup_time', type=float, default=40)
  parser.add_argument('--max_velocity', type=float, default=7.0)
  parser.add_argument('--cfl_safety_factor', type=float, default=0.5)
  parser.add_argument('--viscosity', type=float, default=1e-3)
  parser.add_argument('--density', type=float, default=1.0)
  parser.add_argument('--forcing_scale', type=float, default=1.0)
  parser.add_argument('--simulation_time', type=float, default=30.0)
  parser.add_argument('--peak_wavenumber', type=int, default=4)
  parser.add_argument('--low_res', type=int, default=64)
  parser.add_argument('--high_res', type=int, default=2048)
  parser.add_argument('--demo_file', type=str, default="re1000")
  parser.add_argument('--characteristic_length', type=int, default=1)
  parser.add_argument('--domain_scale', type=int, default=1)
  parser.add_argument('--decay', default=False, action='store_true')
  parser.add_argument('--demo', default=False, action='store_true')
  # For generating data
  parser.add_argument('--save_file', type=str, default="re1000")
  parser.add_argument('--save_index', type=int, default=1)
  parser.add_argument('--save_interval', type=int, default=1000, help='Save data every N steps to avoid memory issues')
  parser.add_argument('--subsample_interval', type=int, default=1, help='Save every Nth timestep (e.g., 10 means save every 10th timestep)')
  parser.add_argument('--initial_velocity_file', type=str, default=None, help='Path to .npz file containing initial velocity field (optional)')
  parser.add_argument('--verify_initial_velocity', default=False, action='store_true', help='Generate verification plot comparing original and loaded initial velocity')
  main(parser.parse_args()) 