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
    --low_res  2048\
    --outer_steps 1 \
    --generate_steps 34770 \
    --warmup_time 4.0 \
    --max_velocity 7.0 \
    --peak_wavenumber 4 \
    --forcing_scale 1.0 \
    --decay \
    --save_file "kolmogorov" \
    --save_index 1 \
    --iters 1 \
    --demo_file "Kolmogorov" \
    --seed 42 \
    --training_save_interval 10