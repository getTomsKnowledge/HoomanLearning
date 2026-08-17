"""
Filename: nn_helpers.py
Author: Tom West
Date: 08/16/2026
Description: Contains neural network helper functions, such as activations, weight/bias initialization, input handling, output generation, backprop.
"""

### IMPORT STATEMENTS: ###

# Standard Libraries:
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Custom Libraries:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NETWORKS_DIR = PROJECT_ROOT / "networks"
DATA_DIR = PROJECT_ROOT / "train" / "data"

### INPUT HANDLING: ###

def load_data(network_type="xor"):
    """
    Returns:
        X: Input data (2D array)
        Y: Target data (1D array)
    """
    # Load the training data from train.json
    if (network_type.lower() == "xor"):
        datafilename = DATA_DIR / "train_XOR.json"
    with open(datafilename, "r") as f:
        data = json.load(f)
    
    X = np.array(data["INPUTS"])
    Y = np.array(data["TARGETS"])
    
    return X, Y

# AI-generated (note the emoji...):
def get_user_boolean_pair():
    """Prompts user for a two-element binary input, handling human (read 'messy') formatting."""
    print("\n💡 Enter a boolean/binary input pair (e.g., [0,1], 1 0, (0, 1), 1,1)")

    while True:
        user_input = input("Enter pair (or 'q' to quit): ").strip().lower()

        if user_input == "q":
            print("🛑 Input canceled.")
            return None

        # 1. Use Regex to extract only the digits/numbers, ignoring all punctuation
        # This matches integers and floats like 0, 1, 0.0, 1.0
        tokens = re.findall(r"\d+\.?\d*", user_input)

        # 2. Validation: We strictly need exactly two values
        if len(tokens) != 2:
            print(
                f"⚠️ Expected exactly 2 elements, but found {len(tokens)}. Try again."
            )
            continue

        try:
            # 3. Convert extracted strings into standard Python floats
            float_list = [float(num) for num in tokens]

            # 4. Strict Boolean Check: Ensure values represent 0 or 1
            # (Allows 0.0 and 1.0 but blocks inputs like)
            if any(val not in (0.0, 1.0) for val in float_list):
                print("⚠️ Elements must be binary values (0 or 1 only).")
                continue

            # 5. Convert to a flat NumPy array ready for transformation operations
            # Shape will be (2,) - matching your 1D network design
            np_pair = np.array(float_list, dtype=np.float64)
            return np_pair

        except ValueError:
            print("⚠️ Could not parse numbers. Please use digits (0 or 1).")

# AI-generated (note the emoji...):
def load_neural_network_weights(networks_dir=None):
    """Discovers JSON files, prompts the user, and loads the selected network data."""
    parent_dir = Path(networks_dir) if networks_dir is not None else NETWORKS_DIR

    if not parent_dir.exists():
        print(f"❌ Neural network directory not found: {parent_dir}")
        return None

    # 1. Discover all .json files in the current folder
    json_files = sorted(f for f in parent_dir.glob("*.json") if f.is_file())

    if not json_files:
        print("❌ No neural network JSON files found in this directory.")
        return None

    # 2. Display the discovered files
    print("\n📦 Available Neural Network Weights:")
    for idx, file_path in enumerate(json_files, 1):
        print(f"  [{idx}] {file_path.name}")

    # 3. Get user choice (accepts either the list number or the exact filename)
    while True:
        user_input = (
            input("\nEnter the number or filename to load (or 'q' to quit): ")
            .strip()
            .lower()
        )

        if user_input == "q":
            print("🛑 Loading canceled.")
            return None

        # Check if user entered a valid list number
        if user_input.isdigit():
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(json_files):
                selected_file = json_files[choice_idx]
                break

        # Check if user entered the exact filename (with or without .json)
        filename_guess = (
            user_input if user_input.endswith(".json") else user_input + ".json"
        )

        # Case-insensitive comparison against actual files
        matched_files = [f for f in json_files if f.name.lower() == filename_guess]

        if matched_files:
            selected_file = matched_files[0]
            break

        print("⚠️ Invalid choice. Please select a valid number or filename.")

    # 4. Load and return the JSON data
    try:
        print(f"🔄 Loading weights from '{selected_file}'...")
        with selected_file.open("r", encoding="utf-8") as f:
            weights_data = json.load(f)
        print("✅ Weights successfully loaded!")
        return weights_data
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

### HYPERPARAMETER DEFINITION: ###

def set_hyperparameters(mode="auto", input=np.zeros([2,2]), output=np.zeros([2,2]), activation_function="", learning_rate=0.01, epochs=100, batch_size=1, num_batches=1, steps=1, hidden_units=4, num_layers=1, depth=[4,4,4], loss_cutoff=1e-1):
    # activation, gamma_r, epochs, batch_size, num_batches, steps, hidden_units, num_layers:
    hyperparam_dict = {}
    if ( mode == "auto" ):
        hyperparam_dict = {
            "activation": activation_function, # options include: sigmoid, ReLU, ... TODO
            "gamma_r": learning_rate, # 'stride width' for each step of gradient descent, needs tuning TODO
            "epochs": epochs, # number of training runs, needs tuning TODO
            "batch_size": batch_size, # sample size, should be statistically significant TODO
            "num_batches": num_batches, # number of batches per epoch TODO
            "steps": steps, # number of forward passes per epoch
            "num_layers": num_layers, # network width, number of hidden layers TODO
            "hidden_units": hidden_units, # number of neurons in a hidden layer TODO
            "input_dim": input.shape[1], # number of input neurons, should match the number of input features
            "output_dim": output.shape[1], # number of output neurons, should match the number of target classes
            "pop_size": input.shape[0], # population size
            "d": depth, # network depth, vector of dim M allows for variable-depth hidden layers; keeping it simple for now
            "loss_cutoff": loss_cutoff # max acceptable error value
        }
        return hyperparam_dict
    elif mode == "manual":
        # Get user input:
        # activation = input("Enter the activation function (sigmoid, relu, tanh): ") # stub for later

        return hyperparam_dict
    else:
        # Oops
        print("\nOops.  Both manual and automated hyperparameter setting failed... Have a nice day!\n")
        pass
    return -1

### FEATURE INITIALIZATION: ###
def initialize_features(hyperparam_dict={}):
    feature_dict = {
        # Hidden Layer Weights, [alpha_ml] = [w1_ml]:
        "alpha": np.random.rand(hyperparam_dict["hidden_units"], hyperparam_dict["input_dim"]), # hidden weights, m x p matrix for input-to-hidden layer calculation
        # Hidden Layer Bias(es), [alpha_m0] = [b1_m]:
        "hidden_biases": np.random.rand(hyperparam_dict["hidden_units"]), # biases for hidden layer, column vector of length m
        # Output Layer Weights, [beta_km] = [w2_km]:
        "beta": np.random.rand(hyperparam_dict["hidden_units"], hyperparam_dict["output_dim"]), # output weights, m x k for hidden-to-output calculation
        # Output Layer Bias(es), [beta_0m] = [b2_m]:
        "output_biases": np.random.rand(hyperparam_dict["output_dim"]) # simple scalar bias for output layer in this example
    }

    # Ensure convergence to global minimum with initial kick to correct valley in feature space:
    feature_dict["alpha"][2][1] *= -1 # start in trough of global minimum in feature space
    feature_dict["beta"][3] *= -1 # ditto
    feature_dict["hidden_biases"] *= 0.1 # start close to 0 to prevent early hidden sigmoid bias
    feature_dict["output_biases"] *= 0.1 # ditto

    """
    For debug:
    print("Starting weights/biases:\n")
    print(feature_dict["alpha"])
    print(feature_dict["hidden_biases"])
    print(feature_dict["beta"])
    print(feature_dict["output_biases"])
    """

    return feature_dict

### BATCH HANDLING: ###
def generate_batches(input, output, hyperparam_dict):
    # Create a shuffled array of indices based on number of inputs (i)
    # TODO make this stochastic...
    permutation = np.random.permutation(hyperparam_dict["batch_size"])
    X_batch = input[permutation] # TODO
    Y_batch = output[permutation] # TODO
    Z_batch = np.zeros([hyperparam_dict["batch_size"],hyperparam_dict["hidden_units"]]) # TODO
    # Initialize output layer (predictions, y_k):
    Y_pred = np.zeros([hyperparam_dict["batch_size"], hyperparam_dict["output_dim"]])
    error_batch = np.zeros([hyperparam_dict["batch_size"], hyperparam_dict["output_dim"]])
    # Initialize feature gradients:
    delta_NK = np.zeros([hyperparam_dict["batch_size"], hyperparam_dict["output_dim"]])
    ess_NM = np.zeros([hyperparam_dict["batch_size"], hyperparam_dict["hidden_units"]])

    return X_batch, Y_batch, Z_batch, Y_pred, error_batch, delta_NK, ess_NM

### ACTIVATION: ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(s):
    return s * (1 - s)

def activation(activation_type, input):
    if activation_type == "sigmoid":
        return sigmoid(input)
    # TODO add others...
    return -1

def activation_prime(activation_type, input):
    if activation_type == "sigmoid":
        return sigmoid_prime(input)
    # TODO add others...
    return -1

### LOSS / ERROR: ###

def simple_error(y_target, y_pred):
    return y_target - y_pred

def mse_loss(y_target, y_pred):
    error = simple_error(y_target, y_pred) 
    return error * error



def delta():
    return -1
def ess():
    return -1

    
"""
For later:
def relu_prime(s):
    return np.where(s > 0, 1, 0)
...
def tanh_prime(s):
    return 1 - np.tanh(s) ** 2
...
"""

### FORWARD PASS: ###
def get_hidden(input, hyperparam_dict, feature_dict):
    # affine transform for preactivation:
    pre_z = np.zeros(hyperparam_dict["hidden_units"])
    for m in range(hyperparam_dict["hidden_units"]):
        pre_z[m] = feature_dict["hidden_biases"][m]
        for l in range(hyperparam_dict["input_dim"]):
            pre_z[m] += feature_dict["alpha"][m][l]*input[l]
    # Throw the switch! (activate hidden layer)
    z = np.zeros(hyperparam_dict["hidden_units"])
    for m in range(hyperparam_dict["hidden_units"]):
        z[m] = activation(hyperparam_dict["activation"], pre_z[m])

    return z

def get_output(final_hidden, hyperparam_dict, feature_dict):
    pre_y = np.zeros(hyperparam_dict["output_dim"])
    for k in range(hyperparam_dict["output_dim"]):
        pre_y[k] = feature_dict["output_biases"][k]
        for m in range(hyperparam_dict["hidden_units"]):
            pre_y[k] += feature_dict["beta"][m][k]*final_hidden[m]
    # Throw the switch! (activate output layer)
    y = np.zeros(hyperparam_dict["output_dim"])
    for k in range(hyperparam_dict["output_dim"]):
        y[k] = activation(hyperparam_dict["activation"], pre_y[k])

    return y

def get_error(error_type, target, predicted):

    if error_type == "simple":
        K = target.shape[0]
        output = np.zeros(K)
        for k in range(K):
            output[k] = simple_error(target[k], predicted[k])
    if error_type == "mse":
        N = target.shape[0]
        K = target.shape[1]
        error_vec = np.zeros([N,K])
        output = 0.0
        for i in range(N):
            for k in range(K):
                error_vec[i][k] = mse_loss(target[i][k], predicted[i][k])
        output = 0.5 * sum(error_vec)[0]
        # debug: print(f"output: {output}\n")

    return output

# TODO legacy for xor runtime -- update for modularity, use hidden(), output(), activation(), etc.
def get_final_pass(X, w1, b1, w2, b2, M, K):
    # HIDDEN:
    p = X.shape[0]
    Z = np.zeros(M)
    # affine transform for preactivation:
    pre_z = np.zeros(M)
    for m in range(M):
        pre_z[m] = b1[m]
        for l in range(p):
            pre_z[m] += w1[m][l]*X[l]
    # Throw the switch! (activate hidden layer)
    for m in range(M):
        Z[m] = sigmoid(pre_z[m])

    Y = np.zeros(K)
    # OUTPUT:
    pre_y = np.zeros(K)
    for k in range(K):
        pre_y[k] = b2[k]
        for m in range(M):
            pre_y[k] += w2[m][k]*Z[m]
    # Throw the switch! (activate output layer)
    for k in range(K):
        Y[k] = sigmoid(pre_y[k])
    return Y

### BACKWARD PASS: ###
# Backpropagation:
def update_weights(layer_type, input, delta, batch_size, output_dim, input_dim, learning_rate, weights, biases):
    # TODO refactor weights arrays so that units / dims can be consistently determined via shape(), remove units and dims args/params
    # gradient parameters:
    weight_grad = 0.0
    bias_grad = 0.0
    # calculate input feature's loss gradient
    for k in range(output_dim):
        weight_grad = 0.0
        for i in range(batch_size):            
            bias_grad += delta[i][k]
        biases[k] -= learning_rate * bias_grad
        for j in range(input_dim):
            weight_grad = 0.0
            for i in range(batch_size):
                weight_grad += delta[i][k] * input[i][j]
            # TODO stopgap... bad practice, need to make agnostic to layer type; refactor of weight matrix dimensionality too painful right now...
            if layer_type == "hidden":
                weights[k][j] -= learning_rate * weight_grad
            elif layer_type == "output":
                weights[j][k] -= learning_rate * weight_grad

    return weights, biases

### VISUALIZATION: ###
def plot_loss(loss_vector, title, hyperparam_dict):
    # Enable latex:
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })
    """
    # Legend:
    legend_label = (
        "Hyperparameters:\n"
        f"    training size (N): {hyperparam_dict["pop_size"]}\n"
        f"    batch size (n): {hyperparam_dict["batch_size"]}\n"
        f"    max epochs: {hyperparam_dict["epochs"]}\n"
        f"    $\\eta$ (lr): {hyperparam_dict["gamma_r"]}\n"
        f"    hidden activation: {hyperparam_dict["activation"]}\n"
        f"    output activation: {hyperparam_dict["activation"]}\n"
        f"    width: {hyperparam_dict["num_layers"]}\n"
        f"    depth: {hyperparam_dict["hidden_units"]}\n"
        f"    loss cutoff: {hyperparam_dict["loss_cutoff"]}"
    )

    # Plot the total loss vector:
    plt.plot(loss_vector, marker='o', linestyle='-', label=legend_label)

    # Add labels
    plt.title(title)
    plt.xlabel("Epoch (#)")
    plt.ylabel("Loss")

    # Position the legend outside or in an empty area to avoid overlapping data
    plt.legend(loc='upper right', fontsize=10)

    # Display the plot
    plt.show()

def save_network(feature_dict):
    netName = input("Enter the name of the neural network: ")
    weightFilename = NETWORKS_DIR / f"{netName}.json"
    hidden_weights = feature_dict["alpha"].tolist()
    hidden_biases = feature_dict["hidden_biases"].tolist()
    output_weights = feature_dict["beta"].tolist()
    output_bias = feature_dict["output_biases"].tolist()

    weights_dict = {
        "Hidden_Weights": hidden_weights,
        "Hidden_Biases": hidden_biases,
        "Output_Weights": output_weights,
        "Output_Bias": output_bias
    }

    # save:
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    with weightFilename.open("w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=4)

    return weightFilename