# Neural Hamilton

[![arXiv](https://img.shields.io/badge/arXiv-2410.20951-b31b1b.svg)](https://arxiv.org/abs/2410.20951)

This repository contains the official implementation of the paper "Neural Hamilton: Can A.I. Understand Hamiltonian Mechanics?"

## Overview

Neural Hamilton reformulates Hamilton's equations as an operator learning problem, exploring whether artificial intelligence can grasp the principles of Hamiltonian mechanics without explicitly solving differential equations. The project introduces new neural network architectures specifically designed for operator learning in Hamiltonian systems.

Key features:
- Novel algorithm for generating physically plausible potential functions using Gaussian Random Fields and cubic B-splines
- Multiple neural network architectures (DeepONet, TraONet, VaRONet, MambONet) for solving Hamilton's equations
- Comparison with traditional numerical methods (RK4)
- Performance evaluation on various physical potentials (harmonic oscillators, double-well potentials, Morse potentials)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU support)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Axect/Neural_Hamilton
   cd Neural_Hamilton
   ```

2. Install the required packages:
   ```bash
   # Use pip
   pip install -r requirements.txt

   # Or Use uv with sync requirements.txt (recommended)
   uv pip sync requirements.txt

   # Or Use uv (fresh install)
   uv pip install -U torch wandb survey polars numpy optuna matplotlib scienceplots beaupy rich
   ```

## Usage

### Training Models

The main training script can be run with different dataset sizes:
```bash
python main.py --data normal --run_config configs/run_config.yaml  # 10,000 potentials
python main.py --data more --run_config configs/run_config.yaml    # 100,000 potentials
python main.py --data much --run_config configs/run_config.yaml    # 1,000,000 potentials
```

For hyperparameter optimization:
```bash
python main.py --data normal --run_config configs/run_config.yaml --optimize_config configs/optimize_config.yaml
```

### Analyzing Results

To analyze trained models:
```bash
python analyze.py
```

The script provides options to:
- Evaluate model performance on test datasets
- Generate visualizations of potential functions and trajectories
- Compare performance with RK4 numerical solutions

### Model Architectures

1. **DeepONet**: Baseline neural operator model
   ```python
   config = {
       "nodes": 128,
       "layers": 3,
       "branches": 10
   }
   ```

2. **VaRONet**: Variational Recurrent Operator Network
   ```python
   config = {
       "hidden_size": 512,
       "num_layers": 4,
       "latent_size": 30
   }
   ```

3. **TraONet**: Transformer Operator Network
   ```python
   config = {
       "d_model": 64,
       "nhead": 8,
       "num_layers": 3,
       "dim_feedforward": 512
   }
   ```

4. **MambONet**: Mamba Operator Network
   ```python
   config = {
       "d_model": 128,
       "num_layers1": 4,
       "n_head": 4,
       "num_layers2": 4,
       "d_ff": 1024
   }
   ```

## Key Results

1. Performance Comparison:
   - MambONet consistently outperforms other architectures and RK4
   - Models show improved performance with larger training datasets
   - Neural approaches maintain accuracy over longer time periods compared to RK4

2. Computation Time:
   - TraONet demonstrates fastest computation time
   - MambONet and DeepONet show comparable speeds to RK4
   - VaRONet requires more computational resources

3. Physical Potential Tests:
   - Superior performance on Simple Harmonic Oscillator, Double Well, and Morse potentials
   - Successful extrapolation to non-differentiable potentials (Mirrored Free Fall)
   - Improved accuracy on smoothed variants (Softened Mirrored Free Fall)

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{kim2024neuralhamiltonaiunderstand,
      title={Neural Hamilton: Can A.I. Understand Hamiltonian Mechanics?}, 
      author={Tae-Geun Kim and Seong Chan Park},
      year={2024},
      eprint={2410.20951},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20951}, 
}
```

## License

[MIT License](LICENSE)
