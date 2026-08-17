# HoomanLearning

Human learning-oriented from-scratch neural network builds.

A small from-scratch neural-network playground focused first on learning XOR with a single hidden layer. The current trainer implements the forward pass, backpropagation, gradient descent updates, saving/loading weights, loss plotting, and a reproducible Markdown training-results report using NumPy.

## Current Model

The main working example is a single-hidden-layer feedforward neural network for XOR:

- Input layer: 2 units
- Hidden layer: 4 sigmoid units
- Output layer: 1 sigmoid unit
- Task: binary XOR classification
- Loss: `0.5 * sum squared error` over the batch
- Optimizer: full-batch gradient descent
- Dataset: `train/data/train_XOR.json`

This project is intentionally explicit rather than highly vectorized. Many operations are written with loops so the chain-rule/backpropagation mechanics stay visible.

## Repository Layout

```text
.
|-- train/
|   |-- train_SingleHiddenLayer_XOR.py
|   `-- data/
|       `-- train_XOR.json
|-- runtimes/
|   |-- xor_runtime.py
|   `-- results/
|       `-- <network-name>_results.md
|-- utils/
|   |-- nn_helpers.py
|   |-- cite_sources.py
|   `-- results.py
|-- networks/
|   `-- *.json
|-- references.py
`-- README.md
```

## Requirements

- Python 3.14 or compatible modern Python 3
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
```

## Train XOR

From the repository root:

```bash
python train/train_SingleHiddenLayer_XOR.py
```

The trainer will:

1. Load `train/data/train_XOR.json`.
2. Initialize the 2-4-1 network.
3. Train until the loss cutoff is reached or the max epoch count is exhausted.
4. Prompt for a network name and save weights under `networks/<name>.json`.
5. Plot the training loss.
6. Write a Markdown results report to `runtimes/results/<network-name>_results.md` with architecture, dataset, hyperparameters, final predictions, parameter shapes/counts, reproducibility notes, and citations.

## Run a Saved XOR Network

From the repository root:

```bash
python runtimes/xor_runtime.py
```

The runtime lists JSON weight files in `networks/`, prompts for a selection, accepts a binary input pair, and prints the network output plus rounded class prediction.

## Results Utility

`utils/results.py` is the reporting layer. It is designed to be reused by future trainers. A trainer should pass plain dictionaries and arrays into:

```python
results.build_training_results(...)
results.print_training_results(training_results, refs.source_list)
```

The Markdown output is meant for a reader who already knows neural networks and wants enough information to reproduce the run: network kind, architecture, dataset, optimizer, loss, stopping criterion, hyperparameters, learned-parameter summary, final predictions, saved-weight location, implementation notes, and source citations.

## Extending

Good next steps for this project:

- Vectorize the dense-layer math with NumPy matrix operations.
- Generalize helpers to support multiple hidden layers.
- Add deterministic seeds for exact reproducibility.
- Add binary cross-entropy as an alternative loss.
- Add gradient checking with finite differences.
- Add a small convolution-forward-pass experiment.

When adding a new network, keep the trainer responsible for training and pass a standard result dictionary into `utils/results.py` for reporting.
