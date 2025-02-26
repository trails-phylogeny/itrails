import argparse
import os
import sys

import yaml
from cutpoints import cutpoints_ABC
from optimizer import optimizer
from read_data import maf_parser


def get_example_paths():
    """Get the paths to the example config and MAF file inside the installed package."""
    base_dir = os.path.dirname(
        __file__
    )  # Get the directory where this script is located
    example_dir = os.path.join(base_dir, "examples")  # Locate examples folder

    example_config = os.path.join(example_dir, "example_config.yaml")
    example_maf = os.path.join(example_dir, "example_alignment.maf")

    if not os.path.exists(example_config) or not os.path.exists(example_maf):
        print(
            "Error: Example files not found in the installed package.", file=sys.stderr
        )
        sys.exit(1)

    return example_config, example_maf


def load_config(config_file):
    """Load the YAML configuration file."""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)


def determine_output_path(output_arg):
    """
    Determines if the --output argument is a directory or a file.
    - If it's a directory, save as `optimization_results.csv` inside it.
    - If it's a file, use the given filename.
    """
    output_path = os.path.abspath(output_arg)

    if os.path.isdir(output_path):
        return os.path.join(output_path, "optimization_results.csv")
    elif output_path.endswith(".csv"):
        return output_path
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
        help="Path to the YAML config file or 'example' to use built-in example",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory or output CSV file. If a directory is provided, saves as 'optimization_results.csv' inside it.",
    )

    args = parser.parse_args()

    # Determine the final output file path
    output_file = determine_output_path(args.output)

    # If "example" is passed, use example config and MAF
    if args.config_file.lower() == "example":
        config_path, maf_path = get_example_paths()
        print(f"Using example configuration: {config_path}")

        # Load example config
        config = load_config(config_path)

        # Dynamically set the correct path for maf_alignment inside the config
        config["input_files"]["maf_alignment"] = maf_path

    else:
        config_path = args.config_file
        config = load_config(config_path)

    # Extract parameters
    fixed_params = config["fixed_parameters"]
    optimized_params = config["optimized_parameters"]
    species_list = config["species_list"]
    input_files = config["input_files"]

    # Get MAF file path
    maf_path = input_files["maf_alignment"]
    if not os.path.exists(maf_path):
        print(
            f"Error: Specified MAF alignment file '{maf_path}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Using MAF alignment file: {maf_path}")
    print(f"Results will be saved to: {output_file}")

    mu = fixed_params["mu"]

    # Compute dependent parameters
    t_1 = 240000 * mu
    t_2 = 40000 * mu
    t_3 = 800000 * mu
    t_C = t_1 + t_2
    t_upper = (
        t_3
        - cutpoints_ABC(fixed_params["n_int_ABC"], 1 / optimized_params["N_ABC"][0])[-2]
    )
    t_out = t_1 + t_2 + t_3 + 2 * optimized_params["N_ABC"][0]

    # Update optimized parameters with computed values
    optimized_params["t_C"] = [t_C, t_C / 10, t_C * 10]
    optimized_params["t_upper"] = [t_upper, t_upper / 10, t_upper * 10]

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
