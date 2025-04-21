# American Options Pricing with Poseidon

This project demonstrates how to finetune the Poseidon foundation model to solve American options pricing problems. Poseidon is a PDE (Partial Differential Equation) foundation model that can be adapted to various PDE-based problems with minimal training data.

## Overview

American options pricing is governed by a free-boundary partial differential equation (PDE) problem, making it an excellent candidate for Poseidon's capabilities. By finetuning Poseidon on American option data, we create a model that can quickly generate option price surfaces for any combination of market parameters.

## Project Structure

```
american_options_poseidon/
├── data/
│   ├── raw/                              # Raw data from QuantLib
│   │   └── american_option_dataset.npz   # Raw generated dataset
│   │
│   └── poseidon_data/                    # Formatted data for Poseidon
│       ├── splits.json                   # Train/val/test splits
│       ├── sample_0000/
│       │   ├── input.npy                 # Input grid (4 channels)
│       │   ├── output.npy                # Output grid (1 channel)
│       │   └── metadata.json             # Option parameters
│       └── ...
│
├── scripts/
│   ├── generate_data.py                  # Script to generate option data
│   ├── prepare_data.py                   # Script to format data for Poseidon
│   ├── dataset.py                        # Dataset class for loading data
│   ├── model_config.py                   # ScOTConfig for American options
│   ├── finetune.py                       # Script to fine-tune Poseidon
│   └── inference.py                      # Script to run inference
│
├── configs/
│   └── training_config.yaml              # Training hyperparameters
│
├── models/
│   └── american_options/                 # Fine-tuned model output
│
├── viz/
│   └── results/                          # Inference results
│
└── requirements.txt                      # Project dependencies
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Clone and install Poseidon:

```bash
git clone https://github.com/camlab-ethz/poseidon.git
cd poseidon
pip install -e .
cd ..
```

## Usage

### 1. Generate Dataset

Generate American options data using QuantLib:

```bash
mkdir -p data/raw
python scripts/generate_data.py
```

### 2. Prepare Data for Poseidon

Format the data for Poseidon finetuning:

```bash
python scripts/prepare_data.py
```

### 3. Finetune Poseidon

Finetune the Poseidon model on American options data:

```bash
python scripts/finetune.py \
    --data_dir data/poseidon_data \
    --output_dir models/american_options \
    --model_size B \
    --batch_size 16 \
    --epochs 50
```

Or use the Poseidon command-line approach:

```bash
accelerate launch poseidon/scOT/train.py \
    --config configs/training_config.yaml \
    --wandb_run_name "american_options_finetuning" \
    --checkpoint_path models \
    --data_path data/poseidon_data \
    --finetune_from "camlab-ethz/Poseidon-B" \
    --replace_embedding_recovery
```

### 4. Run Inference

Visualize the results on test samples:

```bash
mkdir -p viz/results
python scripts/inference.py \
    --model_dir models/american_options/final \
    --data_dir poseidon_data \
    --output_dir viz/results \
    --num_samples 10
```

## Model Configuration

The Poseidon model is configured with:

- 4 input channels (spot/strike ratio, risk-free rate, dividend yield, volatility)
- 1 output channel (option price)
- Grid size of 64x64 (representing a space-time grid for the stock price and time to expiry)

## Performance

After finetuning, the model can generate full American option price surfaces in a single forward pass. This is significantly faster than traditional PDE solvers which require iterative methods.

Typical relative L1 errors on test samples:
- Mean: ~2-5%
- Median: ~1-3%

## Acknowledgements

This project uses:
- [Poseidon](https://github.com/camlab-ethz/poseidon) - Foundation model for PDEs
- [QuantLib-Python](https://github.com/lballabio/QuantLib-SWIG) - Quantitative finance library
- [PyTorch](https://pytorch.org/) - Deep learning framework

## License

This project is licensed under the MIT License - see the LICENSE file for details.
