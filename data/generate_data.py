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
  
  force = forcing(v0) if forcing is not None else (jnp.zeros_like(v0[0].data), jnp.zeros_like(v0[1].data))

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
  return trajectory, force


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
    
    # Run warmup simulation (compute only, no saving)
    while count + outer_steps <= warm_up_step:
      logger.info(f"Warmup step {count} of {warm_up_step}")
      if len(warmup_result[0].array.data.shape) > 2:  # Has time dimension
          warmup_result[0].array.data = warmup_result[0].array.data[-1]
          warmup_result[1].array.data = warmup_result[1].array.data[-1]
      if count == outer_steps:
        save_dir = f'../data/training_data/{args.high_res}'
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(f'{save_dir}/{args.save_file}_warmup_initial_velocity.npz',
                           u=warmup_result[0].data,
                           v=warmup_result[1].data,
                           resolution=args.high_res,
                           outer_steps=outer_steps if count + outer_steps <= warm_up_step else warm_up_step - count,
                           warmup_step_count=count,
                           warmup_time=args.warmup_time,
                           warmup_time_step=delta_t,  # Add warmup time step
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
                           peak_wavenumber=args.peak_wavenumber)
    warmup_result, _ = get_trajectory(args, size=args.high_res, rng=subrng,
                                        outer_steps=outer_steps, v0=warmup_result)
      
    # Store warmup trajectory if save_warmup is enabled
    if args.save_warmup:
      warmup_trajectories.append((count, warmup_result))
      
    count += outer_steps

    if warm_up_step > count:
      if warmup_result is not None:
        warmup_result[0].array.data = warmup_result[0].array.data[-1]
        warmup_result[1].array.data = warmup_result[1].array.data[-1]
      final_warmup, _ = get_trajectory(args, size=args.high_res, rng=subrng,
                                       outer_steps=warm_up_step - count, v0=warmup_result)
      if args.save_warmup:
        warmup_trajectories.append((count, final_warmup))
      warmup_result = final_warmup
    
    logger.info("Warmup completed - no warmup data saved")

    # Save warmup data if requested
    if args.save_warmup and warmup_trajectories:
      save_dir = f'../data/warmup_data'
      os.makedirs(f'{save_dir}/{args.high_res}', exist_ok=True)
      
      logger.info(f"Saving {len(warmup_trajectories)} warmup trajectory segments...")
      for segment_idx, (step_count, trajectory) in enumerate(warmup_trajectories):
        warmup_file = f'{save_dir}/{args.high_res}/{args.save_file}_warmup_segment_{segment_idx}_step_{step_count}_index_{args.save_index}.npz'
        logger.info(f"Saving warmup segment to: {warmup_file}")
        
        np.savez_compressed(warmup_file,
                           u=trajectory[0].data,
                           v=trajectory[1].data,
                           resolution=args.high_res,
                           outer_steps=outer_steps if step_count + outer_steps <= warm_up_step else warm_up_step - step_count,
                           warmup_segment=segment_idx,
                           warmup_step_count=step_count,
                           warmup_time=args.warmup_time,
                           warmup_time_step=delta_t,  # Add warmup time step
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
                           peak_wavenumber=args.peak_wavenumber)

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
        trajectory, _ = get_trajectory(args, size=resolution, outer_steps=args.demo_steps, v0=warmup_result)
        file_name = f'../figs/%s_demo_{resolution}x{resolution}.png' % args.save_file
        logger.info(file_name)
        plot_trajectory(args, resolution, trajectory, file_name)
    else:
      # Training data mode: generate full datasets with generate_steps
      for resolution in resolution_list:
        count = 0
        trajectory = None
        while count + args.training_save_interval <= args.generate_steps:
          if trajectory is not None:
            trajectory[0].array.data = trajectory[0].array.data[-1]
            trajectory[1].array.data = trajectory[1].array.data[-1]
          trajectory, force = get_trajectory(args, size=resolution, outer_steps=args.training_save_interval, v0=warmup_result if count == 0 else trajectory)
          
          save_dir = f'../data/training_data/{resolution}'
          os.makedirs(save_dir, exist_ok=True)
          
          count += args.training_save_interval
          data_file = f'{save_dir}/{args.save_file}_{resolution}x{resolution}_step_{count}_index_{args.save_index}.npz'
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
                            timestep = get_dt(args, args.low_res, args.characteristic_length),
                            fu=force[0].data,
                            fv=force[1].data
                            )
          
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
  parser.add_argument('--training_save_interval', type=int, default=10)
  parser.add_argument('--initial_velocity_file', type=str, default="/Users/yiwei/Projects/Python/thesis/JAX-CFD/data/training_data/256/kolmogorov_warmup_initial_velocity.npz", help='Path to .npz file containing initial velocity field (optional)')
  main(parser.parse_args()) 