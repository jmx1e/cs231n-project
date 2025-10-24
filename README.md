# Posterior Sampling Using Diffusion Models for HDR Reconstruction

Project code for CS 231N Spring 2025
Adapted from DAPS: https://github.com/zhangbingliang2019/DAPS

Author: Jamin Xie

## Unique contributions:
- All files 'configs/task/hdr*.yaml` are configuration files for new forward operators
- `forward_operator/__init__.py` modified to include the new forward operators
- `baseline_reconstruction.py` contains logic for running baseline
- `baseline.sh` runs the baseline in command line
- `compress_figures.py` compresses figures to pdf
- `make_figures.py` makes figures
- `plot_ablation.py` makes the ablation study graph
- Updated version of `requirements.txt` with fixed dependencies is in `environment.yml`
- Full run history to make figures in paper and poster are in `samples.sh`
