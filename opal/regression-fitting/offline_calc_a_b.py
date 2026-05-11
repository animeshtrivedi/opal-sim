# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses

# https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
import logging
import os
import json
import random
import time

logger = logging.getLogger(__name__)
# FORMAT = "[%(levelname)s: %(asctime)s %(filename)s:%(lineno)s - %(funcName)10s()] %(message)s"
FORMAT = "[%(levelname)s: %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)

logging_level = os.environ.get("LOGGING_LEVEL", "INFO")
logger.setLevel(logging._nameToLevel.get(logging_level))
allowed_levels = logging._nameToLevel.keys()

GPU_TFLOPS_FP16 = {
    "H100": 1979 / 2,
    "L40S": 733 / 2,
}


def calculate_series_from_file(model_dim: int, layers: int, tflops: float, filename: str, tp: int = 1):
    with open(filename, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    data = data1["gpu"]
    n = data.keys()
    y = data.values()
    D = model_dim
    L = layers

    y = []
    x1 = []
    x2 = []
    divider_to_seconds = int(data1["params"]["divider_to_seconds"])
    for nn in n:
        nni = int(nn)
        term1 = L * ((nni**2 * D) / (tp * tflops * 10**12))
        term2 = L * ((nni * D**2) / (tp * tflops * 10**12))
        print(f" {data[nn]} = a . {term1} + b . {term2}")
        # because runtime is in milliseconds
        y.append(float(data[nn]) / divider_to_seconds)
        x1.append(term1)
        x2.append(term2)
        # print(f"\t term1={term1} , term2= {terms2}")
    return y, x1, x2


def regression(y, x1, x2):
    # Preparing and solving the linear least squares for the provided data points.
    import numpy as np
    import json

    # Given data (y = a * x1 + b * x2)
    y = np.array(y, dtype=float)

    x1 = np.array(x1, dtype=float)

    x2 = np.array(x2, dtype=float)

    # Design matrix (N x 2)
    X = np.column_stack((x1, x2))

    # Solve with least squares (more stable than manual inv)
    theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a_hat, b_hat = theta

    # Predictions and residuals
    y_pred = X @ theta
    res = y - y_pred
    N = len(y)
    p = 2  # parameters

    # Residual variance estimate
    sigma2 = (res**2).sum() / (N - p) if N > p else float("nan")

    # Covariance of theta: sigma2 * (X^T X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_theta = sigma2 * XtX_inv
    std_err = np.sqrt(np.diag(cov_theta))

    # R^2
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (res**2).sum()
    R2 = 1 - ss_res / ss_tot

    # Print results
    results = {
        "a_hat": float(a_hat),
        "b_hat": float(b_hat),
        "residuals_sum_of_squares": float(ss_res),
        "sigma2 (residual variance)": float(sigma2),
        "std_err_a": float(std_err[0]),
        "std_err_b": float(std_err[1]),
        "R2": float(R2),
        "rank": int(rank),
        "singular_values": [float(v) for v in s],
    }

    print("Least squares solution:")
    print(f"  a = {a_hat:.8f}")
    print(f"  b = {b_hat:.8f}\n")

    """
    atr notes: 
     1. RSS : two order of magnitude gap between y and errors are good. 
     2. Sigma^2 : How much noise is left after fitting RSS / (N - P) P = params = 2, N = data points 
     3. R2 = cofficient of determiniatio, 1 = perfect fit, 0 = no fit. 
    """
    print("Uncertainty and fit diagnostics:")
    print(f"  Residual sum of squares = {ss_res:.6e}")
    print(f"  Residual variance (sigma^2) = {sigma2:.6e}")
    print(f"  Std error a = {std_err[0]:.6e}")
    print(f"  Std error b = {std_err[1]:.6e}")
    print(f"  R^2 = {R2:.6f}")
    print(f"  Design matrix rank = {rank}")
    print(f"  Singular values = {s}\n")

    # 95% confidence intervals (approx Normal: theta +/- 1.96*std_err)
    ci_lower = theta - 1.96 * std_err
    ci_upper = theta + 1.96 * std_err

    print("Approx. 95% confidence intervals:")
    print(f"  a: [{ci_lower[0]:.8f}, {ci_upper[0]:.8f}]")
    print(f"  b: [{ci_lower[1]:.8f}, {ci_upper[1]:.8f}]")

    # Also show the data, predictions and residuals
    table = np.column_stack((y, y_pred, res))
    print("\nData | Predicted | Residuals (first 10 rows):")
    for i, row in enumerate(table):
        print(f"{i+1:2d}: y={row[0]:.8f}, y_hat={row[1]:.8f}, res={row[2]:.8e}")

    # Return results as JSON for programmatic use (in notebook output)
    results_json = json.dumps(results)
    return results_json, a_hat, b_hat


def generate_projected_data(args, filename, gpu_name: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model)
    # Print important details
    print("Model name:", args.model)
    model_dim = getattr(config, "hidden_size", None)
    model_layers = getattr(config, "num_hidden_layers", None)
    print("Hidden size / model dimension:", model_dim)
    print("Number of hidden layers:", getattr(config, "n_layer", model_layers))
    gpu_key = None
    for k in GPU_TFLOPS_FP16.keys():
        if k.lower() in gpu_name.lower():
            gpu_key = k
            break
        print(f" matching {k} with {gpu_name} ")

    tflops = GPU_TFLOPS_FP16[gpu_key]
    print(f"GPU key is {gpu_key}, TFLOPS = {tflops}")
    y, x1, x2 = calculate_series_from_file(
        model_dim, model_layers, tflops, os.path.join(args.output_dir, filename), args.tensor_parallel
    )
    json, a, b = regression(y, x1, x2)
    return a, b


def get_GPU_names():
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], stdout=subprocess.PIPE, text=True
    )
    gpu_names = result.stdout.strip().split("\n")
    print(gpu_names)
    return gpu_names


def get_model_name(args):
    full_model_name = args.model
    return "-".join(full_model_name.split("/"))


def main():
    # create parser
    parser = argparse.ArgumentParser(description="Regression calculation.")
    parser.add_argument(
        "-m",
        "--model",
        help="HuggingFace model name (e.g., ibm-granite/granite-3.1-8b-instruct, mistralai/Mistral-7B-Instruct-v0.3)",
        default="ibm-granite/granite-3.1-8b-instruct",
        required=True,
    )
    parser.add_argument("-gpu", "--gpu", type=str, help="GPU type", default="h100", required=True)
    parser.add_argument("-measurements", "--measurements", type=str, help="which measurement file?", required=True)
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=1, help="Tensor parallel mode", required=True)
    parser.add_argument("-o", "--output-dir", type=str, help="directory to put files in", default="./", required=False)
    args = parser.parse_args()

    a, b = generate_projected_data(args, args.measurements, args.gpu)
    code = f"MoonshotAIModel_A = {a}\nMoonshotAIModel_B = {b}"
    print(code)


if __name__ == "__main__":
    main()
