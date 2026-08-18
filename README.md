# HoomanLearning

Human learning-oriented from-scratch neural network builds.

A small (from-scratch) neural-network sandbox.
First example: XOR classification problem w/ single net inspired by Syntax/CJ's "I Built an LLM from Scratch" YouTube video: https://www.youtube.com/watch?v=YmLp8qe87A0 .

## Usage (XOR Example)

From the top-level directory:

# Train:

```bash
python .\train\train_SingleHiddenLayer_XOR.py

# Results:
- pyplot pops up showing training loss over epochs -- Code is currently programmed to terminate when a threshold loss is achieved (later iterations will use dropout for more-sophisticated control of training)
- Results Markdown with hyperparameters and outcomes appears in \runtimes\results\<your_model_name_here>_results.md

# Run:

```python .\runtimes\xor_runtime.py

## XOR Architecture

The main working example is a single-hidden-layer feedforward neural network for XOR:

- Input layer: 2 units
- Hidden layer: 4 sigmoid units
- Output layer: 1 sigmoid unit
- Task: binary XOR classification
- Loss: `0.5 * sum squared error` over the batch
- Optimizer: full-batch gradient descent
- Dataset: `train/data/train_XOR.json`

I used element-wise operations rather than vector math for accounting/auditing during debug.  Future iterations will (should) be vector-based.

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
  - pathlib
  - json
  - sys
  - datetime
  - re
- NumPy
- Matplotlib


- Custom library files in /utils

Install dependencies with:

```bash
pip install numpy matplotlib
```

## Training Notes
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

Next steps:

- Input scaling/normalization/regularization.
- Smarter weights initialization (automated exploration of loss surfaces prior to training).
- Vector/matrix operations in place of element-wise ops.
- Further helper generalization to support multiple hidden layers.
- Seeding for reproducibility.
- Cross-entropy loss (opens gate for MNIST?)
- Finite difference checks on gradient
- Dropout on biases/weights
- Convolution?  Variable-width network?  

Note: When adding a new network, keep the trainer responsible for training and pass a standard result dictionary into `utils/results.py` for reporting.
