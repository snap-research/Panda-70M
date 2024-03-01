#!/bin/bash

python -m torch.distributed.run --standalone --nproc_per_node=1 -m utils.noise_correlation.scripts.compute_dataset_statistics --correlation_config configs/dataset_statistics/kinetics-64x36/correlation_statistics_debug.yaml
python -m torch.distributed.run --standalone --nproc_per_node=1 -m utils.noise_correlation.scripts.compute_covariance_statistics --correlation_config configs/dataset_statistics/kinetics-64x36/correlation_statistics_debug.yaml
