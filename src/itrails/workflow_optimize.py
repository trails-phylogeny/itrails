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


def determine_output_path(output_arg, is_example=False):
    """
    Determines the output directory and file path.

    - If using "example":
      - If `output_arg` is a directory, use it.
      - If `output_arg` is a file (e.g., "dir/result.csv"), extract the directory part.
      - If `output_arg` is just a filename (e.g., "result.csv"), use the current working directory.
    - Otherwise:
      - If `output_arg` is a directory, save results as 'optimization_results.csv' inside it.
      - If `output_arg` is a file, use the given filename.
    """
    output_path = os.path.abspath(output_arg)

    if is_example:
        if os.path.isdir(output_path):
            return os.path.join(output_path, "optimization_results.csv"), output_path
        else:
            # Extract directory if a file is given (e.g., "dir/result.csv")
            output_dir = (
                os.path.dirname(output_path)
                if os.path.dirname(output_path)
                else os.getcwd()
            )
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            return os.path.join(output_dir, "optimization_results.csv"), output_dir

    # If not using "example", allow either a file or directory
    if os.path.isdir(output_path):
        return os.path.join(output_path, "optimization_results.csv"), output_path
    elif output_path.endswith(".csv"):
        return output_path, os.path.dirname(output_path)
    else:
        print(
            f"Error: Output path '{output_arg}' must be either a directory or a .csv file.",
            file=sys.stderr,
        )
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
        "--output",
        type=str,
        required=True,
        help="Path to the output directory or output CSV file. If a directory is provided, saves as 'optimization_results.csv' inside it. "
        "When using 'example', the example MAF file will be downloaded here.",
    )

    args = parser.parse_args()

    # Determine the final output file path and ensure output directory exists
    is_example = args.config_file.lower() == "example"
    output_file, output_dir = determine_output_path(args.output, is_example=is_example)

    # If "example" is passed, use example config and download the MAF file
    if is_example:
        print("Using example configuration.")

        # Get the script's directory (where example_config.yaml is located)
        base_dir = os.path.dirname(__file__)
        example_dir = os.path.join(base_dir, "examples")
        example_config = os.path.join(example_dir, "example_config.yaml")

        if not os.path.exists(example_config):
            print(
                "Error: Example config file not found in the installed package.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Download the example MAF file to the output directory
        maf_path = download_example_maf(output_dir)

        # Load example config
        config = load_config(example_config)

        # Update the config to point to the downloaded MAF file
        config["input_files"]["maf_alignment"] = maf_path

    else:
        config_path = args.config_file
        config = load_config(config_path)

        # Extract the MAF file path from the config
        maf_path = config["input_files"]["maf_alignment"]
        if not os.path.exists(maf_path):
            print(
                f"Error: Specified MAF alignment file '{maf_path}' not found.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Retrieve number of CPU cores
    num_cores = config["fixed_parameters"].get(
        "num_cores", mp.cpu_count()
    )  # Default to all cores
    os.environ["OMP_NUM_THREADS"] = str(num_cores)  # Set OpenMP threads
    os.environ["MKL_NUM_THREADS"] = str(num_cores)  # Set MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)  # Set NumExpr threads
    os.environ["RAYON_NUM_THREADS"] = str(num_cores)  # Set Rayon threads

    print(f"Using {num_cores} CPU cores.")
    print(f"Using MAF alignment file: {maf_path}")
    print(f"Results will be saved to: {output_file}")

    # Extract fixed parameters
    fixed_params = config["fixed_parameters"]
    optimized_params = config["optimized_parameters"]
    species_list = config["species_list"]
    mu = fixed_params["mu"]

    # Compute dependent parameters dynamically, just like the standalone script
    t_1 = 240000 * mu
    t_A = t_1
    t_B = t_1
    t_2 = 40000 * mu
    t_C = t_1 + t_2
    t_3 = 800000 * mu

    # Compute dependent parameters using cutpoints_ABC()
    t_upper = (
        t_3
        - cutpoints_ABC(fixed_params["n_int_ABC"], 1 / optimized_params["N_ABC"][0])[-2]
    )
    t_out = t_1 + t_2 + t_3 + 2 * optimized_params["N_ABC"][0]

    # Define initial parameter values
    optimized_params["t_A"][0] = t_A
    optimized_params["t_B"][0] = t_B
    optimized_params["t_C"][0] = t_C
    optimized_params["t_2"][0] = t_2
    optimized_params["t_upper"][0] = t_upper

    # Update the dictionary dynamically with computed values
    optimized_params["N_AB"][0] = optimized_params["N_AB"][0]  # Keep initial
    optimized_params["N_ABC"][0] = optimized_params["N_ABC"][0]  # Keep initial
    optimized_params["r"][0] = 1e-8 / mu  # Compute r from mu

    # Read MAF alignment
    maf_alignment = maf_parser(maf_path, species_list)

    # Run optimization
    res = optimizer(
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
