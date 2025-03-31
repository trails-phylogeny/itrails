import argparse
import csv
import os

from itrails.cutpoints import cutpoints_AB, cutpoints_ABC
from itrails.get_trans_emiss import trans_emiss_calc
from itrails.ncpu import N_CPU, update_n_cpu
from itrails.optimizer import viterbi_wrapper
from itrails.read_data import maf_parser
from itrails.yaml_helpers import load_config

## URL of the example MAF file on Zenodo
# EXAMPLE_MAF_URL = "https://zenodo.org/records/14930374/files/example_alignment.maf"


def main():
    """Command-line entry point for running viterbi decoding."""
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

    print(f"Results will be saved to: {output_dir} as '{output_prefix}_viterbi.csv'.")

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

    # Validate t1, ta, tb, tc
    # Found values tracking
    found_values = set()
    optim_variables = []
    optim_list = []

    # Function to process parameters
    def process_parameter(param):
        if param in fixed_params and param in optimized_params:
            raise ValueError(f"Parameter '{param}' cannot be both fixed and optimized.")
        if param in fixed_params:
            found_values.add(param)
            return fixed_params[param], True  # Value, min, max, fixed=True
        elif param in optimized_params:
            found_values.add(param)
            return optimized_params[param], False
            # Value, min, max, fixed=False
        return None, None  # Not found

    # Process each parameter
    t_1, t_1_fixed = process_parameter("t_1")
    t_A, t_A_fixed = process_parameter("t_A")
    t_B, t_B_fixed = process_parameter("t_B")
    t_C, t_C_fixed = process_parameter("t_C")
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

    if "t_A" in found_values:
        if t_A_fixed:
            fixed_dict["t_A"] = t_A
        else:
            optim_variables.append("t_A")
            optim_list.append(t_A)

    if "t_B" in found_values:
        if t_B_fixed:
            fixed_dict["t_B"] = t_B
        else:
            optim_variables.append("t_B")
            optim_list.append(t_B)

    if "t_C" in found_values:
        if t_C_fixed:
            fixed_dict["t_C"] = t_C
        else:
            optim_variables.append("t_C")
            optim_list.append(t_C)

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
            optim_list.append(optimized_params[param])
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
            N_ABC_starting = optimized_params["N_ABC"]
            if "t_3" in optimized_params:
                t_3_starting = optimized_params["t_3"]
                t_upper_starting = (
                    t_3_starting
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                t_upper = t_upper_starting
                optim_variables.append("t_upper")
                optim_list.append(t_upper_starting)

            elif "t_3" in fixed_params:
                t_3 = fixed_params["t_3"]
                t_upper_starting = (
                    t_3 - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                t_upper = t_upper_starting
                optim_variables.append("t_upper")
                optim_list.append(t_upper_starting)
            else:
                raise ValueError("'t_3' not found in parameter definition.")
        elif "N_ABC" in fixed_params:
            N_ABC_starting = fixed_params["N_ABC"]
            if "t_3" in optimized_params:
                t_3_starting = optimized_params["t_3"]
                t_upper_starting = (
                    t_3_starting
                    - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                t_upper = t_upper_starting
                optim_variables.append("t_upper")
                optim_list.append(t_upper)
            elif "t_3" in fixed_params:
                t_3 = fixed_params["t_3"]
                t_upper_starting = (
                    t_3 - cutpoints_ABC(fixed_dict["n_int_ABC"], 1 / N_ABC_starting)[-2]
                )
                t_upper = t_upper_starting
                optim_variables.append("t_upper")
                optim_list.append(t_upper)
            else:
                raise ValueError("'t_3' not found in parameter definition.")
        else:
            raise ValueError("'N_ABC' not found in parameter definition.")
    elif "t_upper" in optimized_params:
        optim_variables.append("t_upper")
        optim_list.append(optimized_params["t_upper"])
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
        if not isinstance(starting_value, (int, float)) or starting_value <= 0:
            raise ValueError(f"Value for '{param}' must be a positive number.")
        if param == "r":
            optim_list[i] = starting_value / float(mu)
        else:
            optim_list[i] = starting_value * float(mu)

    for param, values in fixed_dict.items():
        if param != "n_int_AB" and param != "n_int_ABC":
            if param == "r":
                fixed_dict[param] = float(values) / float(mu)
            else:
                fixed_dict[param] = float(values) * float(mu)

    for i, param in enumerate(optimized_params):
        fixed_dict[param] = optim_list[i]

    cut_ABC = cutpoints_ABC(fixed_dict["n_int_ABC"], 1)
    if case == frozenset(["t_A", "t_B", "t_C"]):

        t_out = (
            (
                (
                    ((fixed_dict["t_A"] + fixed_dict["t_B"]) / 2 + fixed_dict["t_2"])
                    + fixed_dict["t_C"]
                )
                / 2
                + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_out"] = t_out

    elif case == frozenset(["t_1", "t_A"]):
        t_B = fixed_dict["t_1"]
        t_C = fixed_dict["t_1"] + fixed_dict["t_2"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
            + fixed_dict["t_upper"]
            + 2 * fixed_dict["N_ABC"]
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_B"] = t_B
        fixed_dict["t_C"] = t_C
        fixed_dict["t_out"] = t_out
        fixed_dict.pop("t_1")
    elif case == frozenset(["t_1", "t_B"]):
        t_A = fixed_dict["t_1"]
        t_C = fixed_dict["t_1"] + fixed_dict["t_2"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
            + fixed_dict["t_upper"]
            + 2 * fixed_dict["N_ABC"]
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_A"] = t_A
        fixed_dict["t_C"] = t_C
        fixed_dict["t_out"] = t_out
        fixed_dict.pop("t_1")
    elif case == frozenset(["t_1", "t_C"]):
        t_A = fixed_dict["t_1"]
        t_B = fixed_dict["t_1"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
            + fixed_dict["t_upper"]
            + 2 * fixed_dict["N_ABC"]
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_A"] = t_A
        fixed_dict["t_B"] = t_B
        fixed_dict["t_out"] = t_out
        fixed_dict.pop("t_1")
    elif case == frozenset(["t_A", "t_B"]):
        t_C = (fixed_dict["t_A"] + fixed_dict["t_B"]) / 2 + fixed_dict["t_2"]
        t_out = (
            (
                (
                    ((fixed_dict["t_A"] + fixed_dict["t_B"]) / 2 + fixed_dict["t_2"])
                    + t_C
                )
                / 2
                + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_C"] = t_C
        fixed_dict["t_out"] = t_out
    elif case == frozenset(["t_A", "t_C"]):
        t_B = (fixed_dict["t_A"] + fixed_dict["t_C"] - fixed_dict["t_2"]) / 2
        t_out = (
            (
                (
                    ((fixed_dict["t_A"] + t_B) / 2 + fixed_dict["t_2"])
                    + fixed_dict["t_C"]
                )
                / 2
                + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_B"] = t_B
        fixed_dict["t_out"] = t_out
    elif case == frozenset(["t_B", "t_C"]):
        t_A = (fixed_dict["t_B"] + fixed_dict["t_C"] - fixed_dict["t_2"]) / 2
        t_out = (
            (
                (
                    ((t_A + fixed_dict["t_B"]) / 2 + fixed_dict["t_2"])
                    + fixed_dict["t_C"]
                )
                / 2
                + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_A"] = t_A
        fixed_dict["t_out"] = t_out
    elif case == frozenset(["t_1"]):
        t_A = t_B = fixed_dict["t_1"]
        t_C = fixed_dict["t_1"] + fixed_dict["t_2"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + cut_ABC[fixed_dict["n_int_ABC"] - 1] * fixed_dict["N_ABC"]
            + fixed_dict["t_upper"]
            + 2 * fixed_dict["N_ABC"]
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_A"] = t_A
        fixed_dict["t_B"] = t_B
        fixed_dict["t_C"] = t_C
        fixed_dict["t_out"] = t_out
        fixed_dict.pop("t_1")

    print("Parameters validated, reading alignment.")
    for key, value in fixed_dict.items():
        print(f"{key}: {value}")
    maf_alignment = maf_parser(maf_path, species_list)
    if maf_alignment is None:
        raise ValueError("Error reading MAF alignment file.")

    print("Calculating transition and emission probability matrices.")
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        fixed_dict["t_A"],
        fixed_dict["t_B"],
        fixed_dict["t_C"],
        fixed_dict["t_2"],
        fixed_dict["t_upper"],
        fixed_dict["t_out"],
        fixed_dict["N_AB"],
        fixed_dict["N_ABC"],
        fixed_dict["r"],
        fixed_dict["n_int_AB"],
        fixed_dict["n_int_ABC"],
        "standard",
        "standard",
    )

    hidden_file = os.path.join(output_dir, f"{output_prefix}_hidden_states.csv")
    if os.path.exists(hidden_file):
        print(f"Warning: File '{hidden_file}' already exists.")
        hidden_file = os.path.join(output_dir, f"{output_prefix}_hidden_states_2.csv")
        print("Using an alternative file name: {hidden_file}")
    starting_AB = (fixed_dict["t_A"] + fixed_dict["t_B"]) / 2
    t_AB = fixed_dict["t_2"] / fixed_dict["N_ABC"]
    coal_AB = fixed_dict["N_ABC"] * fixed_dict["N_AB"]
    cut_AB = cutpoints_AB(fixed_dict["n_int_AB"], t_AB, coal_AB)
    cut_AB = [((starting_AB + x) / mu) for x in cut_AB]

    starting_ABC = fixed_dict["t_C"]
    coal_ABC = fixed_dict["N_ABC"] / fixed_dict["N_ABC"]
    cut_ABC = cutpoints_ABC(fixed_dict["n_int_ABC"], coal_ABC)
    cut_ABC = [((starting_ABC + x) / mu) for x in cut_ABC]

    topology_map = {
        0: "({sp1,sp2},sp3)",
        1: "((sp1,sp2),sp3)",
        2: "((sp1,sp3),sp2)",
        3: "((sp2,sp3),sp1)",
    }
    with open(hidden_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "state_idx",
                "topology",
                "interval_1st_coalescent",
                "interval_2nd_coalescent",
                "shorthand_name",
            ]
        )
        for state_idx, shorthand in hidden_names.items():
            key_val = shorthand[0]

            topology = topology_map.get(key_val, "Unknown")
            interval_1_0 = (
                cut_AB[shorthand[1]] if key_val == 0 else cut_ABC[shorthand[1]]
            )
            interval_1_1 = (
                cut_AB[shorthand[1] + 1] if key_val == 0 else cut_ABC[shorthand[1] + 1]
            )
            interval_1_text = f"{interval_1_0:.2f}-{interval_1_1:.2f}"
            interval_2_0 = cut_ABC[shorthand[2]]
            interval_2_1 = cut_ABC[shorthand[2] + 1]
            interval_2_text = f"{interval_2_0:.2f}-{interval_2_1:.2f}"

            writer.writerow(
                [
                    state_idx,
                    topology,
                    interval_1_text,
                    interval_2_text,
                    shorthand,
                ]
            )
    print(f"Hidden states written to file {hidden_file}.")

    print("Running viterbi.")

    viterbi_result = viterbi_wrapper(a=a, b=b, pi=pi, V_lst=maf_alignment)
    print("Writing results to file.")
    output_file = os.path.join(output_dir, f"{output_prefix}_viterbi.csv")

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Block_idx", "position_start", "position_end", "most_likely_state"]
        )
        for block_idx, res in enumerate(viterbi_result):
            if len(res) == 0:
                continue
            segment_start = 0
            current_state = res[0]
            for pos in range(1, len(res)):
                if res[pos] != current_state:
                    writer.writerow(
                        [
                            block_idx,
                            segment_start,
                            pos - 1,
                            current_state,
                        ]
                    )
                    segment_start = pos
                    current_state = res[pos]
            # Write the final segment for this block
            writer.writerow([block_idx, segment_start, len(res) - 1, current_state])

    print(f"Viterbi decoding complete. Results saved to {output_file}.")


if __name__ == "__main__":
    main()
