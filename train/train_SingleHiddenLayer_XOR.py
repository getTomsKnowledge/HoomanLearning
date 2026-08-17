"""
Filename: train_SingleHiddenLayer_XOR.py
Author: Tom West
Date: 08/10/2026
Description: Training a single-hidden-layer neural network
to learn the XOR function using backpropagation and gradient descent.

Architecture: 2->4->1
    Input Layer:  2 neurons for the two input features (x1, x2)
    Hidden Layer: 4 neurons with sigmoid activation function
    Output Layer: 1 neuron with sigmoid activation function (y)

Sources:
    These should print to console following execution.
    See cite_sources.py for citation generation functions.  IEEE styling.
    Primary inspirations:
        CJ (Syntax) "I Built an LLM from Scratch" https://www.youtube.com/watch?v=YmLp8qe87A0 (July, A2026)
        Ch. 10 -- Tibshirani, Hastie, & Friedman - "The Elements of Statistical Learning (2nd Edition)" (2017)
        Ch. 10 -- Tibshirani, Hastie, Friedman, & James - "An Introduction to Statistical Learning with Applications in R (2nd Edition)" (2021)
        pp. 210-212 -- Goodfellow, Bengio, & Courville - "Deep Learning" (2016)
"""

### IMPORT STATEMENTS: ###

# Standard Libraries:
import sys
from pathlib import Path
import json
import numpy as np


# Custom Libraries:
# Change path to match local directory structure:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import nn_helpers as nn
from utils import results
import references as refs

### GRADIENT OPERATIONS: ###
# Hidden layer gradients:
def delta(K, error_i, y_pred_i):
    delta_i = np.zeros(K)
    for k in range(K):
        delta_i[k] = error_i[k] * (-1 * nn.sigmoid_prime(y_pred_i[k]))

    return delta_i

# Output layer gradients:
def ess(M, K, delta_i, beta, z_i):
    ess_i = np.zeros(M)
    z_prime = 0.0
    for m in range(M):
        z_prime = 0.0
        z_prime = nn.sigmoid_prime(z_i[m])
        for k in range(K):
            ess_i[m] += delta_i[k]*beta[m][k]
        ess_i[m] *= z_prime

    return ess_i

# Main:
def train():

    ## Get input (X) and target (Y) data from JSON file: ##
    X, Y = nn.load_data("xor")

    ## Set hyperparameters for training: ##
    # activation, gamma_r, epochs, batch_size, num_batches, steps, hidden_units, num_layers, depth:
    hyperparam_dict = nn.set_hyperparameters("auto", X, Y, "sigmoid", 0.1, 50000, X.shape[0], 1, 1, 4, 1, [4], 1e-2)

    
    ## Initialize weights and biases: ##
    feature_dict = nn.initialize_features(hyperparam_dict)
        
    # Set loss parameters:
    epoch_loss = 0.0
    training_loss = np.zeros([hyperparam_dict["epochs"]])
    step_loss = 0.0

    # Run training epochs:
    final_epoch = hyperparam_dict["epochs"]
    for e in range(hyperparam_dict["epochs"]):

        epoch_loss = 0.0

        # steps:
        for n in range(hyperparam_dict["steps"]):
            # Initialize step batches:
            X_batch, Y_batch, Z_batch, Y_pred, error_batch, delta_NK, ess_NM = nn.generate_batches(X, Y, hyperparam_dict)
            for i in range(hyperparam_dict["batch_size"]):
                ## FORWARD PASS ##

                # HIDDEN:
                # (Single hidden layer for XOR...)
                Z_batch[i] = nn.get_hidden(X_batch[i], hyperparam_dict, feature_dict)

                # OUTPUT:
                Y_pred[i] = nn.get_output(Z_batch[i], hyperparam_dict, feature_dict)

                # ERROR:
                error_batch[i] = nn.get_error("simple", Y_batch[i], Y_pred[i])
                                   
                # Calculate gradients:
                delta_NK[i] = delta(hyperparam_dict["output_dim"], error_batch[i], Y_pred[i])
                ess_NM[i] = ess(hyperparam_dict["hidden_units"], hyperparam_dict["output_dim"], delta_NK[i], feature_dict["beta"], Z_batch[i])

            ## Backpropagation: ##
            feature_dict["alpha"], feature_dict["hidden_biases"] = nn.update_weights("hidden", X_batch, ess_NM, hyperparam_dict["batch_size"], hyperparam_dict["hidden_units"], hyperparam_dict["input_dim"], hyperparam_dict["gamma_r"], feature_dict["alpha"], feature_dict["hidden_biases"])
            feature_dict["beta"], feature_dict["output_biases"] = nn.update_weights("output", Z_batch, delta_NK, hyperparam_dict["batch_size"], hyperparam_dict["output_dim"], hyperparam_dict["hidden_units"], hyperparam_dict["gamma_r"], feature_dict["beta"], feature_dict["output_biases"])

            ## Step loss: ##
            step_loss = 0.0
            for i in range(hyperparam_dict["batch_size"]):
                Z_batch[i] = nn.get_hidden(X_batch[i], hyperparam_dict, feature_dict)
                Y_pred[i] = nn.get_output(Z_batch[i], hyperparam_dict, feature_dict)
            step_loss = nn.get_error("mse", Y_batch, Y_pred)
            epoch_loss += step_loss
        ## Epoch loss: ##
        training_loss[e] = epoch_loss
        if epoch_loss < hyperparam_dict["loss_cutoff"]:
            final_epoch = e + 1
            break
    # Truncate loss vector to actual number of epochs run:
    training_loss = training_loss[:final_epoch]

    # Save to JSON file:
    saved_weight_path = nn.save_network(feature_dict)

    # Collect final predictions for reproducibility:
    final_predictions = []
    for i in range(X.shape[0]):
        final_hidden = nn.get_hidden(X[i], hyperparam_dict, feature_dict)
        final_predictions.append(nn.get_output(final_hidden, hyperparam_dict, feature_dict))
    prediction_records = results.make_prediction_records(X, Y, final_predictions)

    # Visualize loss over training epochs:
    nn.plot_loss(training_loss, f"Multi-layer Perceptron: Learning XOR\n(Training Loss over {final_epoch} Epochs)", hyperparam_dict)

    training_results = results.build_training_results(
        network_name=Path(saved_weight_path).stem,
        network_kind="single-hidden-layer feedforward neural network",
        task="XOR binary classification",
        dataset={
            "name": "train_XOR",
            "path": "train/data/train_XOR.json",
            "samples": int(X.shape[0]),
            "input_shape": tuple(int(dim) for dim in X.shape),
            "target_shape": tuple(int(dim) for dim in Y.shape),
        },
        architecture={
            "input_units": hyperparam_dict["input_dim"],
            "hidden_layers": [hyperparam_dict["hidden_units"]],
            "output_units": hyperparam_dict["output_dim"],
            "hidden_activation": hyperparam_dict["activation"],
            "output_activation": hyperparam_dict["activation"],
        },
        hyperparameters=hyperparam_dict,
        feature_dict=feature_dict,
        training_loss=training_loss,
        epochs_run=final_epoch,
        predictions=prediction_records,
        optimizer="full-batch gradient descent",
        loss_function="0.5 * sum squared error over the batch",
        initialization="Random uniform weights with selected sign flips and 0.1-scaled biases; see nn_helpers.initialize_features().",
        saved_weights=saved_weight_path,
        notes=[
            "Backpropagation is implemented directly with NumPy arrays and explicit loops.",
            "Training stops when epoch loss falls below hyperparam_dict['loss_cutoff'] or max epochs is reached.",
        ],
    )
    return training_results

if __name__ == "__main__":
    # Main code:
    training_results = train()

    # Results and citations:
    results.print_training_results(training_results, refs.source_list)