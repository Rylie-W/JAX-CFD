#!/bin/bash
set -e
set -x

# Decaying turbulence training data generation
# Based on Appendix B parameters:
# - Max velocity: 4.2 for decaying turbulence  
# - 12200 time slices for decaying turbulence dataset
# - Burn-in time and other parameters from paper

python generate_data.py \
    --high_res 2048 \
    --low_res 1024 \
    --outer_steps 50 \
    --generate_steps 12200 \
    --warmup_time 4.0 \
    --max_velocity 4.2 \
    --peak_wavenumber 4 \
    --decay \
    --save_file "decaying_turbulence_v2" \
    --save_index 1 \
    --iters 1 \
    --demo_file "decaying_turbulence_v2" \
    --seed 42 