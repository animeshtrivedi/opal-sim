# SPDX-License-Identifier: Apache-2.0
import dataclasses

# https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
import logging
import os
import json
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)
# FORMAT = "[%(levelname)s: %(asctime)s %(filename)s:%(lineno)s - %(funcName)10s()] %(message)s"
FORMAT = "[%(levelname)s: %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)

logging_level = os.environ.get("LOGGING_LEVEL", "INFO")
logger.setLevel(logging._nameToLevel.get(logging_level))
allowed_levels = logging._nameToLevel.keys()

from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import EngineArgs  # or whichever vLLM module defines default args

from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

GPU_TFLOPS_FP16 = {
    "H100": 1671 / 2,
    "L40S": 733 / 2,
}

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer


def init_vllm(args):
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    return llm


def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)

    # Remove the special tokens.
    return random.choices(
        [v for k, v in vocab.items() if k not in all_special_ids],
        k=length,
    )


def generate_random_prompt(size: int, tokenizer: PreTrainedTokenizerBase):
    prompt_token_ids = sample_tokens(tokenizer, size)
    random.shuffle(prompt_token_ids)
    prompt = tokenizer.decode(prompt_token_ids)
    return prompt


def get_runtime(llm=None, sampling_params=None, prompts=None, size=None):
    start_time = time.time()
    llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    print(f"[size: {size} tokens] prefill time : {end_time - start_time} seconds")
    return end_time - start_time


def generate_runtime_data(args, tokenizer: PreTrainedTokenizerBase):
    sampling_params = SamplingParams(temperature=0, max_tokens=1, detokenize=False)

    vllm = init_vllm(args)
    start = args.start_prompt_length
    end = args.end_prompt_length
    prompts = list(range(start, end, args.prompt_stepping))
    runtime = [get_runtime(vllm, sampling_params, generate_random_prompt(x, tokenizer), x) for x in prompts]
    return zip(prompts, runtime)


def calculate_series_from_file(model_dim: int, layers: int, tflops: float, filename: str, tp: int = 1):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = data.keys()
    y = data.values()
    D = model_dim
    L = layers

    y = []
    x1 = []
    x2 = []
    for nn in n:
        nni = int(nn)
        term1 = L * ((nni**2 * D) / (tflops * 10**12))
        term2 = L * ((nni * D**2) / (tflops * 10**12))
        print(f" {data[nn]} = a . {term1} + b . {term2}")
        y.append(float(data[nn]))
        x1.append(term1)
        x2.append(term2)
        # print(f"\t term1={term1} , term2= {terms2}")
    return y, x1, x2


def regression(y, x1, x2):
    # Preparing and solving the linear least squares for the provided data points.
    import numpy as np
    from math import sqrt
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
        if k in gpu_name:
            gpu_key = k
            break
    tflops = GPU_TFLOPS_FP16[gpu_key]
    print(f"GPU key is {gpu_key}, TFLOPS = {tflops}")
    y, x1, x2 = calculate_series_from_file(model_dim, model_layers, tflops, os.path.join(args.output_dir, filename))
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


def get_active_gpu_name():
    gpu_names = get_GPU_names()
    gpu_index = os.getenv("CUDA_VISIBLE_DEVICES", 0)
    gpu_index = int(gpu_index)
    gpu_name = "-".join(gpu_names[gpu_index].split())
    return gpu_name


def get_model_name(args):
    full_model_name = args.model
    return "-".join(full_model_name.split("/"))


def write_results(zipped2T, filename: str, args):
    dict = {}
    for x, y in zipped2T:
        dict[x] = y
    full_fname = os.path.join(args.output_dir, filename)
    with open(full_fname, "w", encoding="utf-8") as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)


def main():
    # create parser
    print("meta-llama/Llama-3.1-8B")
    parser = FlexibleArgumentParser(description="a_b simulation")
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument("--start-prompt-length", type=int, default=5000, help="Start prompt length")
    parser.add_argument("--end-prompt-length", type=int, default=125000, help="End prompt length")
    parser.add_argument("--prompt-stepping", type=int, default=5000, help="prompt stepping")
    parser.add_argument(
        "-o", "--output-dir", type=str, help="directory to put files in", default="./ab-results/", required=False
    )
    args = parser.parse_args()

    # make the directory
    hostname = os.uname().nodename.split(".")[0]
    tmp_dir = f"{args.output_dir}/{hostname}-{get_model_name(args)}-{get_active_gpu_name()}/"
    cwd = os.getcwd()
    from pathlib import Path

    output_dir = Path(os.path.join(cwd, os.path.abspath(tmp_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    # print all args results
    # print("Parsed arguments:")
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    measured_results = generate_runtime_data(args, tokenizer)
    write_results(measured_results, "measured.json", args)
    a, b = generate_projected_data(args, "measured.json", get_active_gpu_name())
    code = f"MoonshotAIModel_A = {a}\nMoonshotAIModel_A = {b}"
    ab_full_fname = os.path.join(args.output_dir, "a_b.py")
    with open(ab_full_fname, "w") as text_file:
        text_file.write(code)
    print(f"SUCCESS: final result written to {ab_full_fname}")


if __name__ == "__main__":
    main()
