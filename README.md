# PyTorch Template Project

This project provides a flexible template for PyTorch-based machine learning experiments.
It includes configuration management, logging with Weights & Biases (wandb), hyperparameter optimization with Optuna, and a modular structure for easy customization and experimentation.

## Project Structure

- `config.py`: Defines the `RunConfig` and `OptimizeConfig` classes for managing experiment configurations and optimization settings.
- `main.py`: The entry point of the project, handling command-line arguments and experiment execution.
- `model.py`: Contains the model architecture (currently an MLP).
- `util.py`: Utility functions for data loading, device selection, training, and analysis.
- `run_template.yaml`: Template for run configuration.
- `optimize_template.yaml`: Template for optimization configuration.
- `analyze.py`: Script for analyzing completed runs and optimizations, utilizing functions from `util.py`.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/pytorch_template.git
   cd pytorch_template
   ```

2. Install the required packages:
   ```sh
   # Use pip
   pip install torch wandb survey polars numpy optuna matplotlib scienceplots

   # Or Use uv with sync requirements.txt (recommended)
   uv pip sync requirements.txt

   # Or Use uv (fresh install)
   uv pip install -U torch wandb survey polars numpy optuna matplotlib scienceplots
   ```

3. (Optional) Set up a Weights & Biases account for experiment tracking.

## Usage

1. Configure your experiment by modifying `run_template.yaml` or creating a new YAML file based on it.

2. (Optional) Configure hyperparameter optimization by modifying `optimize_template.yaml` or creating a new YAML file based on it.

3. Run the experiment:
   ```sh
   python main.py --run_config path/to/run_config.yaml [--optimize_config path/to/optimize_config.yaml]
   ```

   If `--optimize_config` is provided, the script will perform hyperparameter optimization using Optuna.

4. Analyze the results:
   ```sh
   python analyze.py
   ```

## Configuration

### Run Configuration (`run_template.yaml`)

- `project`: Project name for wandb logging
- `device`: Device to run on (e.g., 'cpu', 'cuda:0')
- `net`: Model class to use
- `optimizer`: Optimizer class
- `scheduler`: Learning rate scheduler class
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `seeds`: List of random seeds for multiple runs
- `net_config`: Model-specific configuration
- `optimizer_config`: Optimizer-specific configuration
- `scheduler_config`: Scheduler-specific configuration

### Optimization Configuration (`optimize_template.yaml`)

- `study_name`: Name of the optimization study
- `trials`: Number of optimization trials
- `seed`: Random seed for optimization
- `metric`: Metric to optimize
- `direction`: Direction of optimization ('minimize' or 'maximize')
- `sampler`: Optuna sampler configuration
- `pruner`: (Optional) Optuna pruner configuration
- `search_space`: Definition of the hyperparameter search space

## Customization

- Custom model: Modify or add models in `model.py`. Models should accept a `hparams` argument as a dictionary, with keys matching the `net_config` parameters in the run configuration YAML file.

- Custom data: Modify the `load_data` function in `util.py`. The current example uses Cosine regression. The `load_data` function should return train and validation datasets compatible with PyTorch's DataLoader.

- Custom training: Customize the `Trainer` class in `util.py` by modifying `step`, `train_epoch`, `val_epoch`, and `train` methods to suit your task. Ensure that `train` returns `val_loss` or a custom metric for proper hyperparameter optimization.

## Features

- Configurable experiments using YAML files
- Integration with Weights & Biases for experiment tracking
- Hyperparameter optimization using Optuna
- Support for multiple random seeds
- Flexible model architecture (currently MLP)
- Device selection (CPU/CUDA)
- Learning rate scheduling
- Analysis tools for completed runs and optimizations

## Analysis

The `analyze.py` script utilizes functions from `util.py` to analyze completed runs and optimizations. Key functions include:

- `select_group`: Select a run group for analysis
- `select_seed`: Select a specific seed from a run group
- `select_device`: Choose a device for analysis
- `load_model`: Load a trained model and its configuration
- `load_study`: Load an Optuna study
- `load_best_model`: Load the best model from an optimization study

These functions are defined in `util.py` and used within `analyze.py`.

To use the analysis tools:

1. Run the `analyze.py` script:
   ```
   python analyze.py
   ```

2. Follow the prompts to select the project, run group, and seed (if applicable).

3. The script will load the selected model and perform basic analysis, such as calculating the validation loss.

4. You can extend the `main()` function in `analyze.py` to add custom analysis as needed, utilizing the utility functions from `util.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as a template and is intended to be freely used, modified, and distributed. Users of this template are encouraged to choose a license that best suits their specific project needs. 

For the template itself:
- You are free to use, modify, and distribute this template.
- No attribution is required, although it is appreciated.
- The template is provided "as is", without warranty of any kind.

When using this template for your own project, please remember to:
1. Remove this license section or replace it with your chosen license.
2. Ensure all dependencies and libraries used in your project comply with their respective licenses.

For more information on choosing a license, visit [choosealicense.com](https://choosealicense.com/).
