# xor_test_02 Results

## Summary
- **Network:** `xor_test_02`
- **Kind:** `single-hidden-layer feedforward neural network`
- **Task:** `XOR binary classification`

## Architecture
- **Input Units:** `2`
- **Hidden Layers:** `[4]`
- **Output Units:** `1`
- **Hidden Activation:** `sigmoid`
- **Output Activation:** `sigmoid`

## Dataset
- **Name:** `train_XOR`
- **Path:** `train/data/train_XOR.json`
- **Samples:** `4`
- **Input Shape:** `[4, 2]`
- **Target Shape:** `[4, 1]`

## Training
- **Optimizer:** `full-batch gradient descent`
- **Loss Function:** `0.5 * sum squared error over the batch`
- **Epochs Run:** `8234`
- **Max Epochs:** `50000`
- **Final Loss:** `0.00999726298023`
- **Loss Cutoff:** `0.01`
- **Converged:** `True`
- **Loss History Length:** `8234`

## Hyperparameters
- **Activation:** `sigmoid`
- **Gamma R:** `0.1`
- **Epochs:** `50000`
- **Batch Size:** `4`
- **Num Batches:** `1`
- **Steps:** `1`
- **Num Layers:** `1`
- **Hidden Units:** `4`
- **Input Dim:** `2`
- **Output Dim:** `1`
- **Pop Size:** `4`
- **D:** `[4]`
- **Loss Cutoff:** `0.01`

## Parameters
- **Alpha:**
  - **Shape:** `(4, 2)`
  - **Count:** `8`
  - **Dtype:** `float64`
- **Hidden Biases:**
  - **Shape:** `(4,)`
  - **Count:** `4`
  - **Dtype:** `float64`
- **Beta:**
  - **Shape:** `(4, 1)`
  - **Count:** `4`
  - **Dtype:** `float64`
- **Output Biases:**
  - **Shape:** `(1,)`
  - **Count:** `1`
  - **Dtype:** `float64`
- **Total Parameters:** `17`

## Final Predictions

| Input | Target | Prediction | Rounded |
| --- | --- | --- | --- |
| `[0, 0]` | `[0]` | `[0.054677168675147346]` | `[0]` |
| `[1, 0]` | `[1]` | `[0.9344951637934569]` | `[1]` |
| `[0, 1]` | `[1]` | `[0.9247967946411485]` | `[1]` |
| `[1, 1]` | `[0]` | `[0.08401504343532971]` | `[0]` |

## Reproducibility
- **Initialization:** `Random uniform weights with selected sign flips and 0.1-scaled biases; see nn_helpers.initialize_features().`
- **Saved Weights:** `insert saved weights filepath here`
- **Project Root:** `insert root here`

## Notes
- Backpropagation is implemented directly with NumPy arrays and explicit loops.
- Training stops when epoch loss falls below hyperparam_dict['loss_cutoff'] or max epochs is reached.

## IEEE Reference List
- [1]	CJ (Syntax), "I Built an LLM from Scratch," in *YouTube*, Jul. 10, 2026, [Online]. Available: https://www.youtube.com/watch?v=YmLp8qe87A0 [Accessed: Aug. 16, 2026].
- [2]	T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning (2nd Edition)*. New York, NY: Springer Science+Business Media, LLC, 2017.
- [3]	G. James et al., *An Introduction to Statistical Learning with Applications in R (2nd Edition)*. New York, NY: Springer Science+Business Media, LLC, 2021.
- [4]	I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA: MIT Press, 2016.
