import argparse
import multiprocessing as mp
import os
import sys
import urllib.request

import yaml
from cutpoints import cutpoints_ABC
from optimizer import optimizer
from read_data import maf_parser

# URL of the example MAF file on Zenodo
EXAMPLE_MAF_URL = "https://zenodo.org/records/14930374/files/example_alignment.maf"


def download_example_maf(output_dir):
    """
    Downloads the example MAF file from Zenodo and saves it inside the specified output directory.
    """
    maf_path = os.path.join(output_dir, "example_alignment.maf")

    if not os.path.exists(maf_path):
        print(f"Downloading example MAF file from {EXAMPLE_MAF_URL}...")
        os.makedirs(output_dir, exist_ok=True)
        urllib.request.urlretrieve(EXAMPLE_MAF_URL, maf_path)
        print(f"Download complete! File saved at: {maf_path}")
    else:
        print(f"Example MAF file already exists at: {maf_path}")

    return maf_path


def load_config(config_file):
    """Load the YAML configuration file."""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Command-line entry point for running the optimizer."""
    parser = argparse.ArgumentParser(
        description="Optimize workflow using TRAILS",
        usage="itrails-optimize <config.yaml> --output OUTPUT_PATH | itrails-optimize example --output OUTPUT_PATH",
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML config file or 'example' to use built-in example (downloads an example MAF file into the output directory).",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the MAF alignment file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path for output files to be stored, ending with run name. Format: 'directory/run_name'.",
    )

    args = parser.parse_args()

    config_path = args.config_file
    config = load_config(config_path)

    input_config = config["input"]["maf_alignment"]
    output_config = config["output"]["output_name"]
    input_cmd = args.input
    output_cmd = args.output

    if input_cmd and input_config:
        print(
            f"Warning: MAF alignment file specified in both config file ({input_config}) and command-line ({input_cmd}). Using command-line input."
        )
        maf_path = input_cmd
    elif input_cmd:
        maf_path = input_cmd
        print(f"Using MAF alignment file: {maf_path}")
    elif input_config:
        maf_path = input_config
        print(f"Using MAF alignment file: {maf_path}")
    elif not (input_cmd and input_config):
        print(
            "Error: MAF alignment file not specified in config file or command-line.",
            file=sys.stderr,
        )
    else:
        raise ValueError(
            "Error: MAF alignment file not specified in config file or command-line."
        )

    if output_cmd and output_config:
        print(
            f"Warning: Output file specified in both config file ({output_config}) and command-line ({output_cmd}). Using command-line output."
        )
        output_file = output_cmd
    elif output_cmd:
        output_file = output_cmd
    elif output_config:
        output_file = output_config
    elif not (output_cmd and output_config):
        print(
            "Error: Output file not specified in config file or command-line.",
            file=sys.stderr,
        )
    else:
        raise ValueError("Error: Output file not specified in config file or command")

    output_path = os.path.dirname(output_file)
    os.makedirs(output_path, exist_ok=True)
    print(f"Results will be saved to: {output_path}")

    # Retrieve the total number of available CPU cores
    available_cores = mp.cpu_count()

    # Get requested number of cores from config; default to available cores
    requested_cores = config["fixed_parameters"].get("num_cpu", available_cores)

    # Ensure num_cpu does not exceed available cores
    num_cpu = min(requested_cores, available_cores)

    # Set environment variables for various libraries
    os.environ["OMP_NUM_THREADS"] = str(num_cpu)  # Set OpenMP threads
    os.environ["MKL_NUM_THREADS"] = str(num_cpu)  # Set MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpu)  # Set NumExpr threads
    os.environ["RAYON_NUM_THREADS"] = str(num_cpu)  # Set Rayon threads

    print(f"Using {num_cpu} CPU cores (out of {available_cores} available).")

    # Extract fixed parameters
    fixed_params = config["fixed_parameters"]
    optimized_params = config["optimized_parameters"]
    species_list = config["species_list"]
    mu = fixed_params["mu"]

    if not (isinstance(fixed_params["n_int_AB"], int) and fixed_params["n_int_AB"] > 0):
        raise ValueError("n_int_AB must be a positive integer")

    if not (
        isinstance(fixed_params["n_int_ABC"], int) and fixed_params["n_int_ABC"] > 0
    ):
        raise ValueError("n_int_ABC must be a positive integer")

    # Validate fixed_params["mu"]
    if not isinstance(fixed_params["mu"], (int, float)) or fixed_params["mu"] <= 0:
        raise ValueError("mu must be a positive float or int.")

    # Validate fixed_params["method"]
    allowed_methods = [
        "nelder-mead",
        "powell",
        "cg",
        "bfgs",
        "newton-cg",
        "l-bfgs-b",
        "tnc",
        "cobyla",
        "slsqp",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]
    if fixed_params["method"] not in allowed_methods:
        raise ValueError(f"Method must be one of {allowed_methods}.")

    # Validate optimized_parameters
    for param, values in optimized_params.items():
        # Check that each parameter is a list (or tuple) of exactly three values.
        if not (isinstance(values, (list, tuple)) and len(values) == 3):
            raise ValueError(
                f"optimized_parameters['{param}'] must be a list of three numbers [starting, min, max]."
            )

        starting, lower, upper = values

        # Check that all values are positive numbers.
        for value in values:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(
                    f"All values for optimized_parameters['{param}'] must be positive numbers."
                )

        # Check that the starting value is between min and max.
        if not (lower <= starting <= upper):
            raise ValueError(
                f"For optimized_parameters['{param}'], the starting value ({starting}) must be between "
                f"the minimum ({lower}) and maximum ({upper})."
            )

    if (
        "t_2" not in optimized_params
        or "N_ABC" not in optimized_params
        or "N_AB" not in optimized_params
        or "r" not in optimized_params
    ):
        raise ValueError(
            "Parameters 't_2', 'N_ABC', 'N_AB' and 'r' must be present in optimized parameters."
        )

    if "t_1" in optimized_params:
        print(
            "'t_1' found in optimized parameters. Assuming ultrametric tree and setting t_A = t_B = t_1."
        )
        optimized_params["t_A"] = optimized_params["t_1"]
        optimized_params["t_B"] = optimized_params["t_1"]
        optimized_params.pop("t_1")
    elif "t_A" not in optimized_params or "t_B" not in optimized_params:
        raise ValueError(
            "Either 't_1' or 't_A' and 't_B' must be present in optimized parameters."
        )
    else:
        print("Using 't_A' and 't_B' from optimized parameters.")

    if "t_C" in optimized_params:
        print("Using 't_C' from optimized parameters.")
    elif "t_C" not in optimized_params and "t_1" in optimized_params:
        print(
            "Computing 't_C' from 't_1' plus 't_2'. Adding standard minumum and maximum (10)"
        )
        t_C = optimized_params["t_1"][0] + optimized_params["t_2"][0]
        optimized_params["t_C"] = [t_C, t_C / 10, t_C * 10]
    else:
        raise ValueError(
            "Either 't_C' must be present in optimized parameters or 't_1' and 't_2' must be present."
        )

    # If no error is raised, all validations passed.
    print("All parameter validations passed successfully!")

    for param, values in optimized_params.items():
        if param == "r":
            for i, value in enumerate(values):
                value[i] = value[i] / mu
        else:
            for i, value in enumerate(values):
                value[i] = value[i] * mu

    # Compute dependent parameters using cutpoints_ABC()
    t_upper = (
        optimized_params["t_3"][0]
        - cutpoints_ABC(fixed_params["n_int_ABC"], 1 / optimized_params["N_ABC"][0])[-2]
    )
    t_out = (
        optimized_params["t_1"][0]
        + optimized_params["t_2"][0]
        + optimized_params["t_3"][0]
        + 2 * optimized_params["N_ABC"][0]
    )

    optimized_params["t_upper"] = [t_upper, t_upper / 10, t_upper * 10]
    optimized_params["t_out"] = [t_out, t_out / 10, t_out * 10]

    # Read MAF alignment
    maf_alignment = maf_parser(maf_path, species_list)

    # Run optimization
    optimizer(
        optim_params=optimized_params,
        fixed_params=fixed_params,
        V_lst=maf_alignment,
        res_name=output_file,
        method=fixed_params["method"],
        header=True,
    )

    print(f"Optimization complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
