"""
Name: xor_runtime.py
Author: Tom West
Date: 08/16/2026
Description: The XOR binary classification task runtime.  Be sure to remember the name of your trained network!
"""

### IMPORT STATEMENTS ###

# Standard Libraries:
import sys
from pathlib import Path
import numpy as np

# Custom Libraries:
# Change path to match local directory structure:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Get custom:
from utils import nn_helpers

if __name__ == "__main__":

    # 1. Load weights/biases dictionary:
    weights = nn_helpers.load_neural_network_weights()

    """
        # print(weights)
        alpha = weights("Hidden_Weights")
        hidden_bias = weights("Hidden_Biases")
        beta = weights("Output_Weights")
        output_bias = weights("Output_Bias")
        user_input = nn_helpers.get_user_boolean_pair()
        hidden_units = hidden_bias.shape[0]
        output_dim = output_bias.shape[0]
        output = nn_helpers.get_output(alpha, hidden_bias, beta, output_bias, hidden_units, output_dim)
        print(output)
    """

    if weights is not None:
        # 2. Extract keys using SQUARE BRACKETS and convert to NumPy arrays:
        alpha = np.array(weights["Hidden_Weights"])
        hidden_bias = np.array(weights["Hidden_Biases"])
        beta = np.array(weights["Output_Weights"])
        output_bias = np.array(weights["Output_Bias"])
        flag = True
        while(flag):
            # 3. Get parsed clean boolean input from user:
            user_input = nn_helpers.get_user_boolean_pair()

            if user_input is not None:
                # 4. .shape[0] to call vectors:
                hidden_units = hidden_bias.shape[0]
                output_dim = output_bias.shape[0]

                # 5. Run forward pass:
                output = nn_helpers.get_final_pass(
                    user_input,
                    alpha,
                    hidden_bias,
                    beta,
                    output_bias,
                    hidden_units,
                    output_dim
                )
                print(f"\n❓ Network Input:\n    {user_input}\n\n")
                print(f"\n🎯 Network Output:\n    {output} --> {int(np.round(output)[0])}")
            flag_string = input("Continue? (True or False)")
            flag = flag_string.lower() in ('true', '1', 'yes', 'yup', 'sure', 'okay', 'y')
    else:
        print("❌ Script aborted: No weights were loaded.")
