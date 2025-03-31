import argparse
import os
from math import inf

import yaml

from itrails.cutpoints import cutpoints_ABC
from itrails.ncpu import N_CPU, update_n_cpu
from itrails.optimizer import optimizer
from itrails.read_data import maf_parser
from itrails.yaml_helpers import FlowSeq, load_config

## URL of the example MAF file on Zenodo
# EXAMPLE_MAF_URL = "https://zenodo.org/records/14930374/files/example_alignment.maf"


def main():
    """Command-line entry point for running the optimizer."""
    parser = argparse.ArgumentParser(
        description="Optimize workflow using TRAILS",
        usage="itrails-optimize <config.yaml> --output OUTPUT_PATH | itrails-optimize example --output OUTPUT_PATH",
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML config file.",
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
        help="Path and prefix for output files to be stored. Format: 'directory/prefix'.",
    )

    args = parser.parse_args()

    config_path = args.config_file
    config = load_config(config_path)

    input_config = config["settings"]["input_maf"]
    output_config = config["settings"]["output_prefix"]
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
        raise ValueError(
            "Error: MAF alignment file not specified in config file or command-line."
        )
    else:
        raise ValueError(
            "Error: MAF alignment file not specified in config file or command-line."
        )

    if output_cmd and output_config:
        print(
            f"Warning: Output file specified in both config file ({output_config}) and command-line ({output_cmd}). Using command-line output."
        )
        user_output = output_cmd
    elif output_cmd:
        user_output = output_cmd
    elif output_config:
        user_output = output_config
    elif not (output_cmd and output_config):
        raise ValueError(
            "Error: Output file not specified in config file or command-line."
        )
    else:
        raise ValueError(
            "Error: Output file not specified in config file or command-line."
        )
    output_dir, output_prefix = os.path.split(user_output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results will be saved to: {output_dir}.")

    # Get user-requested CPU count from the configuration, if present.
    requested_cores = config["settings"].get("n_cpu")
    if requested_cores is not None:
        update_n_cpu(requested_cores)
    else:
        # If not specified, we leave N_CPU as the default
        print(f"No CPU count specified in config; using default {N_CPU} cores.")

    # Extract fixed parameters
    fixed_params = config["fixed_parameters"]
    optimized_params = config["optimized_parameters"]
    settings = config["settings"]
    settings["output_prefix"] = user_output
    settings["input_maf"] = maf_path
    settings["n_cpu"] = N_CPU
    species_list = settings["species_list"]
    mu = float(fixed_params["mu"])
    n_int_AB = settings["n_int_AB"]
    n_int_ABC = settings["n_int_ABC"]
    fixed_dict = {}

    # Fixed parameters validation, sets n_int_AB and n_int_ABC in fixed_dict
    if not (isinstance(n_int_AB, int) and n_int_AB > 0):
        raise ValueError("n_int_AB must be a positive integer")
    fixed_dict["n_int_AB"] = n_int_AB
    if not (isinstance(n_int_ABC, int) and n_int_ABC > 0):
        raise ValueError("n_int_ABC must be a positive integer")
    fixed_dict["n_int_ABC"] = n_int_ABC
    if not isinstance(mu, (int, float)) or mu <= 0:
        raise ValueError("mu must be a positive float or int.")

    # Method validation
    method_input = settings["method"].lower()
    allowed_methods = [
        "nelder-mead",
        "l-bfgs-b",
    ]
    if method_input not in allowed_methods:
        raise ValueError(f"Method must be one of {allowed_methods}.")
    else:
        print(f"Using optimization method: {method_input}")
        method = method_input

    # Validate t1, ta, tb, tc
    # Found values tracking
    found_values = set()
    optim_variables = []
    optim_list = []
    bounds_list = []

    # Function to process parameters
    def process_parameter(param):
        if param in fixed_params and param in optimized_params:
            raise ValueError(f"Parameter '{param}' cannot be both fixed and optimized.")
        if param in fixed_params:
            found_values.add(param)
            return fixed_params[param], None, None, True  # Value, min, max, fixed=True
        elif param in optimized_params:
            found_values.add(param)
            return (
                optimized_params[param][0],
                optimized_params[param][1],
                optimized_params[param][2],
                False,
            )  # Value, min, max, fixed=False
        return None, None, None, None  # Not found

    # Process each parameter
    t_1, t_1_min, t_1_max, t_1_fixed = process_parameter("t_1")
    t_A, t_A_min, t_A_max, t_A_fixed = process_parameter("t_A")
    t_B, t_B_min, t_B_max, t_B_fixed = process_parameter("t_B")
    t_C, t_C_min, t_C_max, t_C_fixed = process_parameter("t_C")
    # Define the set of allowed combinations
    allowed_combinations = {
        frozenset(["t_A", "t_B", "t_C"]),
        frozenset(["t_1", "t_A"]),
        frozenset(["t_1", "t_B"]),
        frozenset(["t_1", "t_C"]),
        frozenset(["t_A", "t_B"]),
        frozenset(["t_A", "t_C"]),
        frozenset(["t_B", "t_C"]),
        frozenset(["t_1"]),
    }

    # Check if found_values is an allowed combination
    if frozenset(found_values) not in allowed_combinations:
        raise ValueError(
            f"Invalid combination of time values: {found_values}, check possible combinations in the documentation."
        )

    # Assign values based on presence and whether they are fixed or optimized
    if "t_1" in found_values:
        if t_1_fixed:
            fixed_dict["t_1"] = t_1
        else:
            optim_variables.append("t_1")
            optim_list.append(t_1)
            bounds_list.append((t_1_min, t_1_max))

    if "t_A" in found_values:
        if t_A_fixed:
            fixed_dict["t_A"] = t_A
        else:
            optim_variables.append("t_A")
            optim_list.append(t_A)
            bounds_list.append((t_A_min, t_A_max))

    if "t_B" in found_values:
        if t_B_fixed:
            fixed_dict["t_B"] = t_B
        else:
            optim_variables.append("t_B")
            optim_list.append(t_B)
            bounds_list.append((t_B_min, t_B_max))

    if "t_C" in found_values:
        if t_C_fixed:
            fixed_dict["t_C"] = t_C
        else:
            optim_variables.append("t_C")
            optim_list.append(t_C)
            bounds_list.append((t_C_min, t_C_max))

    case = frozenset(found_values)

    # Sets t_2, N_ABC, N_AB and r
    params = ["t_2", "N_ABC", "N_AB", "r"]
    for param in params:
        if param in fixed_params and param in optimized_params:
            raise ValueError(f"Parameter '{param}' cannot be both fixed and optimized.")
        if param in fixed_params:
            fixed_dict[param] = fixed_params[param]
        elif param in optimized_params:
            optim_variables.append(param)
            optim_list.append(optimized_params[param][0])
            bounds_list.append((optimized_params[param][1], optimized_params[param][2]))
        else:
            raise ValueError(
                "Parameters 't_2', 'N_ABC', 'N_AB' and 'r' must be present in optimized or fixed parameters."
            )

    # Sets t_upper
    if "t_upper" not in optimized_params:
        print(
            "Warning: 't_upper' not found in parameter definition. Calculating from 't_3' and 'N_ABC'."
        )
        if "N_ABC" in optimized_params:
            N_ABC_starting = optimized_params["N_ABC"][0]
            lower_N_ABC = optimized_params["N_ABC"][1]
            upper_N_ABC = optimized_params["N_ABC"][2]
            if (
                "t_3" in optimized_params
            ):  ## USE t_3 BOUNDS (INVERSE, LOWER t_3, lower t_upper, higher N_ABC. Check if t_upper bounds are  still around starting, user definition of t_upper, calculate t_3 from t_upper for t_out)
                t_3_starting = optimized_params["t_3"][0]
                lower_t_3 = optimized_params["t_3"][1]
                upper_t_3 = optimized_params["t_3"][2]
                t_upper_starting = (
                    t_3_starting
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                lower_t_upper = (
                    lower_t_3
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / upper_N_ABC)[-2]
                )
                upper_t_upper = (
                    upper_t_3
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / lower_N_ABC)[-2]
                )
                if not (lower_t_upper <= t_upper_starting <= upper_t_upper):
                    raise ValueError(
                        f"When calculating t_upper from t_3 and N_ABC, the starting value ({t_upper_starting}) was not between "
                        f"the minimum ({lower_t_upper}) and maximum ({upper_t_upper})."
                    )
                t_upper = [t_upper_starting, lower_t_upper, upper_t_upper]
                optim_variables.append("t_upper")
                optim_list.append(t_upper_starting)
                bounds_list.append((lower_t_upper, upper_t_upper))

            elif "t_3" in fixed_params:
                t_3 = fixed_params["t_3"]
                t_upper_starting = (
                    t_3 - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                lower_t_upper = (
                    t_3 - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / upper_N_ABC)[-2]
                )
                upper_t_upper = (
                    t_3 - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / lower_N_ABC)[-2]
                )
                if not (lower_t_upper <= t_upper_starting <= upper_t_upper):
                    raise ValueError(
                        f"When calculating t_upper from t_3 and N_ABC, the starting value ({t_upper_starting}) was not between "
                        f"the minimum ({lower_t_upper}) and maximum ({upper_t_upper})."
                    )
                t_upper = [t_upper_starting, lower_t_upper, upper_t_upper]
                optim_variables.append("t_upper")
                optim_list.append(t_upper_starting)
                bounds_list.append((lower_t_upper, upper_t_upper))
            else:
                raise ValueError("'t_3' not found in parameter definition.")
        elif "N_ABC" in fixed_params:
            N_ABC_starting = fixed_params["N_ABC"]
            if "t_3" in optimized_params:
                t_3_starting = optimized_params["t_3"][0]
                lower_t_3 = optimized_params["t_3"][1]
                upper_t_3 = optimized_params["t_3"][2]
                t_upper_starting = (
                    t_3_starting
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                lower_t_upper = (
                    lower_t_3
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                upper_t_upper = (
                    upper_t_3
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                if not (lower_t_upper <= t_upper_starting <= upper_t_upper):
                    raise ValueError(
                        f"When calculating t_upper from t_3 and N_ABC, the starting value ({t_upper_starting}) was not between "
                        f"the minimum ({lower_t_upper}) and maximum ({upper_t_upper})."
                    )
                t_upper = [t_upper_starting, lower_t_upper, upper_t_upper]
                optim_variables.append("t_upper")
                optim_list.append(t_upper)
                bounds_list.append((lower_t_upper, upper_t_upper))
            elif "t_3" in fixed_params:
                raise ValueError(
                    "At least one, 't_3' or 'N_ABC' must be present in optimized parameters."
                )
            else:
                raise ValueError("'t_3' not found in parameter definition.")
        else:
            raise ValueError("'N_ABC' not found in parameter definition.")
    elif "t_upper" in optimized_params:
        optim_variables.append("t_upper")
        optim_list.append(optimized_params["t_upper"][0])
        bounds_list.append(
            (optimized_params["t_upper"][1], optimized_params["t_upper"][2])
        )
    elif "t_upper" in fixed_params:
        fixed_dict["t_upper"] = fixed_params["t_upper"]

    # Sets t_out
    if "t_out" in fixed_params:
        fixed_dict["t_out"] = fixed_params["t_out"]
    elif "t_out" in optimized_params:
        raise ValueError("Parameter 't_out' has to be fixed.")

    # Optimized/fixed parameters validation
    for i, param in enumerate(optim_variables):
        if param in fixed_params:
            raise ValueError(
                f"Parameter '{param}' cannot be present in both fixed and optimized parameters."
            )
        starting_value = float(optim_list[i])
        lower_bound = float(bounds_list[i][0])
        upper_bound = float(bounds_list[i][1])

        if not (lower_bound <= starting_value <= upper_bound):
            raise ValueError(
                f"Starting value for '{param}' ({starting_value}) must be between the minimum ({lower_bound}) and maximum ({upper_bound})."
            )
        if not isinstance(starting_value, (int, float)) or starting_value <= 0:
            raise ValueError(f"Starting value for '{param}' must be a positive number.")
        if not isinstance(lower_bound, (int, float)) or lower_bound <= 0:
            raise ValueError(f"Minimum value for '{param}' must be a positive number.")
        # Special handling for 'r'
        if param == "r":
            optim_list[i] = starting_value / float(mu)
            bounds_list[i] = (
                lower_bound / float(mu),
                upper_bound / float(mu),
            )
        else:
            optim_list[i] = starting_value * float(mu)
            bounds_list[i] = (
                lower_bound * float(mu),
                upper_bound * float(mu),
            )

    for param, values in fixed_dict.items():
        if param != "n_int_AB" and param != "n_int_ABC":
            if param == "r":
                fixed_dict[param] = float(values) / float(mu)
            else:
                fixed_dict[param] = float(values) * float(mu)

    filtered_fixed_dict = {
        k: v for k, v in fixed_dict.items() if k not in ["n_int_AB", "n_int_ABC"]
    }

    for param, value in filtered_fixed_dict.items():
        if param == "r":
            filtered_fixed_dict[param] = float(value) * mu
        else:
            filtered_fixed_dict[param] = float(value) / mu

    filtered_fixed_dict["mu"] = mu

    starting_params_yaml = os.path.join(
        output_dir, f"{output_prefix}_starting_params.yaml"
    )

    def adjust_value(value, param, mu):
        """Adjusts the parameter value based on the parameter name."""
        value = float(value)
        return value * mu if param == "r" else value / mu

    def adjust_bounds(bounds, param, mu):
        """Adjusts the bounds for the parameter based on the parameter name."""
        lower, upper = float(bounds[0]), float(bounds[1])
        return [lower * mu, upper * mu] if param == "r" else [lower / mu, upper / mu]

    optim_dict = {
        param: adjust_value(val, param, mu)
        for param, val in zip(optim_variables, optim_list)
    }
    bound_dict = {
        param: adjust_bounds(bounds, param, mu)
        for param, bounds in zip(optim_variables, bounds_list)
    }

    starting_bounds = {
        param: FlowSeq([optim_dict[param], *bound_dict[param]])
        for param in optim_variables
    }
    starting_params = {
        "fixed_parameters": filtered_fixed_dict,
        "optimized_parameters": starting_bounds,
        "settings": settings,
    }
    if "species_list" in starting_params["settings"]:
        starting_params["settings"]["species_list"] = FlowSeq(
            starting_params["settings"]["species_list"]
        )
    with open(starting_params_yaml, "w") as f:
        yaml.dump(starting_params, f, default_flow_style=False)

    best_model_yaml = os.path.join(output_dir, f"{output_prefix}_best_model.yaml")
    starting_best_model = {
        "fixed_parameters": filtered_fixed_dict,
        "optimized_parameters": {},
        "results": {"log_likelihood": -inf, "iteration": None},
        "settings": settings,
    }
    with open(best_model_yaml, "w") as f:
        yaml.dump(starting_best_model, f)

    maf_alignment = maf_parser(maf_path, species_list)
    if maf_alignment is None:
        raise ValueError("Error reading MAF alignment file.")

    print("Running optimization...")
    optimizer(
        optim_variables=optim_variables,
        optim_list=optim_list,
        bounds=bounds_list,
        fixed_params=fixed_dict,
        V_lst=maf_alignment,
        res_name=user_output,
        case=case,
        method=method,
        header=True,
    )

    print(
        f"Optimization complete. Results saved to {os.path.join(
        output_dir, f"{output_prefix}_optimization_history.csv"
    )}.\n Best model saved to {best_model_yaml}."
    )


if __name__ == "__main__":
    main()
