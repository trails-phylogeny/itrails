import argparse
import csv
import math
import os

import pandas as pd

from itrails.cutpoints import cutpoints_AB, cutpoints_ABC
from itrails.int_get_trans_emiss import trans_emiss_calc_introgression
from itrails.ncpu import N_CPU, update_n_cpu
from itrails.optimizer import post_prob_wrapper
from itrails.read_data import maf_parser
from itrails.yaml_helpers import load_config

## URL of the example MAF file on Zenodo
# EXAMPLE_MAF_URL = "https://zenodo.org/records/14930374/files/example_alignment.maf"


def main():
    """Command-line entry point for running viterbi decoding."""
    parser = argparse.ArgumentParser(
        description="Run Posterior decoding using iTRAILS",
        usage="itrails-posterior <config.yaml> --input PATH_MAF --output OUTPUT_PATH",
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

    print(f"Results will be saved to: {output_dir} as '{output_prefix}.posterior.csv'.")

    requested_cores = config["settings"].get("n_cpu")
    if requested_cores is not None:
        update_n_cpu(requested_cores)
    else:
        print(f"No CPU count specified in config; using default {N_CPU} cores.")

    cut_AB = config["settings"].get("cutpoints_AB")
    cut_ABC = config["settings"].get("cutpoints_ABC")
    n_int_AB = config["settings"].get("n_int_AB")
    n_int_ABC = config["settings"].get("n_int_ABC")
    proportional_tm = config["settings"].get("proportional")

    if not n_int_AB and not cut_AB:
        raise ValueError(
            "Error: n_int_AB must be specified in the config file for automatic cutpoints, n_int_AB and cutpoints_AB must be specified in the config file for manual cutpoints."
        )
    if not n_int_ABC and not cut_ABC:
        raise ValueError(
            "Error: n_int_ABC must be specified in the config file for automatic cutpoints, n_int_ABC and cutpoints_ABC must be specified in the config file for manual cutpoints."
        )
    if (cut_AB and n_int_AB) and len(cut_AB) != n_int_AB + 1:
        raise ValueError(
            "Error: cutpoints_AB must have n_int_AB + 1 values, check the config file."
        )
    if (cut_ABC and n_int_ABC) and len(cut_ABC) != n_int_ABC:
        raise ValueError(
            "Error: cutpoints_ABC must have n_int_ABC values, check the config file."
        )

    fixed_params = config["fixed_parameters"]
    optimized_params = config["optimized_parameters"]
    settings = config["settings"]
    species_list = settings["species_list"]
    mu = float(fixed_params["mu"])
    fixed_dict = {}

    if not (isinstance(n_int_AB, int) and n_int_AB > 0):
        raise ValueError("n_int_AB must be a positive integer")
    fixed_dict["n_int_AB"] = n_int_AB
    if not (isinstance(n_int_ABC, int) and n_int_ABC > 0):
        raise ValueError("n_int_ABC must be a positive integer")
    fixed_dict["n_int_ABC"] = n_int_ABC
    if not isinstance(mu, (int, float)) or mu <= 0:
        raise ValueError("mu must be a positive float or int.")

    found_values = set()
    optim_variables = []
    optim_list = []

    params = ["t_2", "N_ABC", "N_AB", "N_BC", "r", "t_m", "m"]
    for param in params:
        if param in fixed_params and param in optimized_params:
            raise ValueError(f"Parameter '{param}' cannot be both fixed and optimized.")
        if param in fixed_params:
            if param == "t_2":
                pre_t_2 = fixed_params[param]
            if param == "N_ABC":
                pre_N_ABC = fixed_params[param]
            if param == "N_AB":
                pre_N_AB = fixed_params[param]
            fixed_dict[param] = fixed_params[param]
        elif param in optimized_params:
            if param == "t_2":
                pre_t_2 = optimized_params[param]
            if param == "N_ABC":
                pre_N_ABC = optimized_params[param]
            if param == "N_AB":
                pre_N_AB = optimized_params[param]
            optim_variables.append(param)
            optim_list.append(optimized_params[param])
        else:
            raise ValueError(
                "Parameters 't_2', 'N_ABC', 'N_AB', 'N_BC, 't_m', 'm' and 'r' must be present in optimized or fixed parameters."
            )

    def process_parameter(param):
        if param in fixed_params and param in optimized_params:
            raise ValueError(f"Parameter '{param}' cannot be both fixed and optimized.")
        if param in fixed_params:
            found_values.add(param)
            return fixed_params[param], True
        elif param in optimized_params:
            found_values.add(param)
            return optimized_params[param], False
        return None, None

    t_1, t_1_fixed = process_parameter("t_1")
    t_A, t_A_fixed = process_parameter("t_A")
    t_B, t_B_fixed = process_parameter("t_B")
    t_C, t_C_fixed = process_parameter("t_C")
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

    if frozenset(found_values) not in allowed_combinations:
        raise ValueError(
            f"Invalid combination of time values: {found_values}, check possible combinations in the documentation."
        )

    if "t_1" in found_values:
        if t_1_fixed:
            fixed_dict["t_1"] = t_1
        else:
            optim_variables.append("t_1")
            optim_list.append(t_1)

    if "t_A" in found_values:
        pre_t_A = t_A
        if t_A_fixed:
            fixed_dict["t_A"] = t_A
        else:
            optim_variables.append("t_A")
            optim_list.append(t_A)
    elif "t_A" not in found_values and "t_1" in found_values:
        pre_t_A = t_1

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

    if "t_out" in fixed_params:
        fixed_dict["t_out"] = fixed_params["t_out"]
    elif "t_out" in optimized_params:
        raise ValueError("Parameter 't_out' has to be fixed.")

    if cut_AB is None:
        abs_cut_AB = pre_t_A + cutpoints_AB(n_int_AB, pre_t_2, 1 / pre_N_AB)
        norm_cut_AB = [(x - pre_t_A) / pre_N_ABC for x in abs_cut_AB]
    else:
        abs_cut_AB = [float(x) for x in cut_AB]
        norm_cut_AB = [(float(x) - pre_t_A) / pre_N_ABC for x in cut_AB]

    if cut_ABC is None:
        norm_cut_ABC = cutpoints_ABC(n_int_ABC, 1)
        abs_cut_ABC = [(float(x) * pre_N_ABC) + pre_t_A + pre_t_2 for x in norm_cut_ABC]
    else:
        abs_cut_ABC = [float(x) for x in cut_ABC]
        norm_cut_ABC = [(float(x) - pre_t_A - pre_t_2) / pre_N_ABC for x in abs_cut_ABC]
        norm_cut_ABC.append(float("inf"))

    # Sets t_upper
    if "t_upper" not in optimized_params:
        print(
            "Warning: 't_upper' not found in parameter definition. Calculating from 't_3' and 'N_ABC'."
        )
        if "N_ABC" in optimized_params:
            N_ABC_starting = optimized_params["N_ABC"]
            if "t_3" in optimized_params:
                t_3_starting = optimized_params["t_3"]
                t_upper_starting = t_3_starting - (norm_cut_ABC[-2] * N_ABC_starting)
                t_upper = t_upper_starting
                optim_variables.append("t_upper")
                optim_list.append(t_upper_starting)

            elif "t_3" in fixed_params:
                t_3 = fixed_params["t_3"]
                t_upper_starting = t_3 - (norm_cut_ABC[-2] * N_ABC_starting)
                t_upper = t_upper_starting
                optim_variables.append("t_upper")
                optim_list.append(t_upper_starting)
            else:
                raise ValueError("'t_3' not found in parameter definition.")
        elif "N_ABC" in fixed_params:
            N_ABC_starting = fixed_params["N_ABC"]
            if "t_3" in optimized_params:
                t_3_starting = optimized_params["t_3"]
                t_upper_starting = t_3_starting - (norm_cut_ABC[-2] * N_ABC_starting)
                t_upper = t_upper_starting
                optim_variables.append("t_upper")
                optim_list.append(t_upper)
            elif "t_3" in fixed_params:
                t_3 = fixed_params["t_3"]
                t_upper_starting = t_3 - (norm_cut_ABC[-2] * N_ABC_starting)
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
    # Optimized/fixed parameters validation
    for i, param in enumerate(optim_variables):
        fixed_dict[param] = optim_list[i]

    if proportional_tm:
        if case == frozenset(["t_1"]):
            if fixed_dict["t_m"] > 1:
                raise ValueError(
                    "If proportional t_m is wanted, please input t_m as a proportion (between 0 and 1)."
                )
            fixed_dict["t_m"] = fixed_dict["t_1"] * fixed_dict["t_m"]
        else:
            raise ValueError(
                "Proportional t_m is only supported for the case where only 't_1' is given, please input t_m as an absolute value (in generations) if you also input 't_A', 't_B' or 't_C'."
            )

    # for i, param in enumerate(optim_variables):
    #    if param in fixed_params:
    #        raise ValueError(
    #            f"Parameter '{param}' cannot be present in both fixed and optimized parameters."
    #        )
    #    float_value = float(optim_list[i])
    #    if not isinstance(float_value, (int, float)) or float_value <= 0:
    #        raise ValueError(f"Value for '{param}' must be a positive number.")
    #    if param == "r":
    #        optim_list[i] = float_value / float(mu)
    #    else:
    #        optim_list[i] = float_value * float(mu)

    for param, values in fixed_dict.items():
        if param != "n_int_AB" and param != "n_int_ABC":
            if param == "r":
                fixed_dict[param] = float(values) / float(mu)
            else:
                fixed_dict[param] = float(values) * float(mu)

    # for i, param in enumerate(optim_variables):
    #    fixed_dict[param] = optim_list[i]
    if fixed_dict["t_upper"] < 0:
        raise ValueError(
            "Parameter 't_upper' must be a positive number. "
            f"Given/calculated value: {fixed_dict['t_upper']}"
        )
    if case == frozenset(["t_A", "t_B", "t_C"]):
        t_out = (
            (
                (
                    (fixed_dict["t_A"] + (fixed_dict["t_B"] + fixed_dict["t_m"])) / 2
                    + fixed_dict["t_2"]
                )
                + (fixed_dict["t_C"] + fixed_dict["t_m"] + fixed_dict["t_2"]) / 2
                + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_out"] = t_out

    elif case == frozenset(["t_1", "t_A"]):
        t_B = t_C = fixed_dict["t_1"] - fixed_dict["t_m"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
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
        t_C = fixed_dict["t_1"] - fixed_dict["t_m"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
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
        t_B = fixed_dict["t_1"] - fixed_dict["t_m"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
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
        t_C = (fixed_dict["t_B"] + fixed_dict["t_A"] + fixed_dict["t_m"]) / 2
        t_out = (
            (
                (
                    (fixed_dict["t_A"] + (fixed_dict["t_B"] + fixed_dict["t_m"])) / 2
                    + fixed_dict["t_2"]
                )
                + (t_C + fixed_dict["t_m"] + fixed_dict["t_2"]) / 2
                + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_C"] = t_C
        fixed_dict["t_out"] = t_out
    elif case == frozenset(["t_A", "t_C"]):
        t_B = (fixed_dict["t_C"] + fixed_dict["t_A"] + fixed_dict["t_m"]) / 2
        t_out = (
            (
                (
                    (fixed_dict["t_A"] + (t_B + fixed_dict["t_m"])) / 2
                    + fixed_dict["t_2"]
                )
                + (fixed_dict["t_C"] + fixed_dict["t_m"] + fixed_dict["t_2"]) / 2
                + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_B"] = t_B
        fixed_dict["t_out"] = t_out
    elif case == frozenset(["t_B", "t_C"]):
        t_A = (fixed_dict["t_C"] + fixed_dict["t_B"] + fixed_dict["t_m"]) / 2
        t_out = (
            (
                (
                    (t_A + (fixed_dict["t_B"] + fixed_dict["t_m"])) / 2
                    + fixed_dict["t_2"]
                )
                + (fixed_dict["t_C"] + fixed_dict["t_m"] + fixed_dict["t_2"]) / 2
                + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
                + fixed_dict["t_upper"]
                + 2 * fixed_dict["N_ABC"]
            )
            if "t_out" not in fixed_dict
            else fixed_dict["t_out"]
        )
        fixed_dict["t_A"] = t_A
        fixed_dict["t_out"] = t_out
    elif case == frozenset(["t_1"]):
        t_A = fixed_dict["t_1"]
        t_C = t_B = fixed_dict["t_1"] - fixed_dict["t_m"]
        t_out = (
            fixed_dict["t_1"]
            + fixed_dict["t_2"]
            + norm_cut_ABC[-2] * fixed_dict["N_ABC"]
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

    lower = pre_t_A
    upper = pre_t_A + pre_t_2

    too_early = (abs_cut_AB[0] < lower) and not math.isclose(
        abs_cut_ABC[0], lower, rel_tol=1e-9, abs_tol=1e-12
    )
    too_late = (abs_cut_AB[-1] > upper) and not math.isclose(
        abs_cut_ABC[-1], upper, rel_tol=1e-9, abs_tol=1e-12
    )

    if too_early or too_late:
        raise ValueError(
            "cutpoints_AB must lie within [t_A, t_A + t_2]."
            f"Given cutpoints_AB: {abs_cut_AB}, t_A: {lower}, t_A + t_2: {upper}."
        )

    lower = pre_t_A + pre_t_2
    upper = fixed_dict["t_out"] / mu
    too_early = (abs_cut_ABC[0] < lower) and not math.isclose(
        abs_cut_ABC[0], lower, rel_tol=1e-9, abs_tol=1e-12
    )
    too_late = (abs_cut_ABC[-2] > upper) and not math.isclose(
        abs_cut_ABC[-2], upper, rel_tol=1e-9, abs_tol=1e-12
    )

    if too_early or too_late:
        raise ValueError(
            "cutpoints_ABC must lie within [t_A + t_2, t_out]."
            f"Given cutpoints_ABC: {abs_cut_ABC}, t_A + t_2: {lower}, t_out: {upper}."
        )

    print("Parameters validated:")
    print(f"Cutpoints AB: {abs_cut_AB}")
    print(f"Cutpoints ABC: {abs_cut_ABC}")
    for key, value in fixed_dict.items():
        if key != "n_int_AB" and key != "n_int_ABC" and key != "r":
            print(f"{key}: {value / mu}")
        elif key == "n_int_AB" or key == "n_int_ABC":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value * mu}")

    print("Reading MAF alignment file.")
    maf_alignment = maf_parser(maf_path, species_list)
    if maf_alignment is None:
        raise ValueError("Error reading MAF alignment file.")

    print("Calculating transition and emission probability matrices.")
    a, b, pi, hidden_names, observed_names = trans_emiss_calc_introgression(
        fixed_dict["t_A"],
        fixed_dict["t_B"],
        fixed_dict["t_C"],
        fixed_dict["t_2"],
        fixed_dict["t_upper"],
        fixed_dict["t_out"],
        fixed_dict["t_m"],
        fixed_dict["N_AB"],
        fixed_dict["N_BC"],
        fixed_dict["N_ABC"],
        fixed_dict["r"],
        fixed_dict["m"],
        fixed_dict["n_int_AB"],
        fixed_dict["n_int_ABC"],
        norm_cut_AB,
        norm_cut_ABC,
    )

    hidden_file = os.path.join(output_dir, f"{output_prefix}.hidden_states.csv")
    if os.path.exists(hidden_file):
        print(f"Warning: File '{hidden_file}' already exists.")
        hidden_file = os.path.join(output_dir, f"{output_prefix}.hidden_states_2.csv")
        print("Using an alternative file name: {hidden_file}")
    topology_map = {  # Ask iker
        0: "({sp1,sp2},sp3)",
        1: "((sp1,sp2),sp3)",
        2: "((sp1,sp3),sp2)",
        3: "((sp2,sp3),sp1)",
        4: "({sp2,sp3},sp1)",
    }
    records = []
    for state_idx, shorthand in hidden_names.items():
        key_val, idx1, idx2 = shorthand

        # first‐coalescent interval
        if key_val == 0:
            a, b = abs_cut_AB[idx1], abs_cut_AB[idx1 + 1]
        else:
            a, b = abs_cut_ABC[idx1], abs_cut_ABC[idx1 + 1]
        interval_1 = f"{a:.2f}-{b:.2f}"

        # second‐coalescent always in ABC
        c, d = abs_cut_ABC[idx2], abs_cut_ABC[idx2 + 1]
        interval_2 = f"{c:.2f}-{d:.2f}"

        records.append(
            {
                "state_idx": state_idx,
                "topology": topology_map.get(key_val, "Unknown"),
                "interval_1st_coalescent": interval_1,
                "interval_2nd_coalescent": interval_2,
                "shorthand_name": shorthand,
            }
        )

    # --- 4) dump via pandas in one shot ---
    out_df = pd.DataFrame.from_records(
        records,
        columns=[
            "state_idx",
            "topology",
            "interval_1st_coalescent",
            "interval_2nd_coalescent",
            "shorthand_name",
        ],
    )
    out_df.to_csv(hidden_file, index=False)

    print(f"Hidden states written to file {hidden_file}.")

    print("Running posterior decoding.")

    posterior_results = post_prob_wrapper(a=a, b=b, pi=pi, V_lst=maf_alignment)

    print("Writing results to file.")

    output_file = os.path.join(output_dir, f"{output_prefix}.posterior.csv")
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if posterior_results:
            n_states = posterior_results[0].shape[1]
        else:
            n_states = 0

        header = ["alignment_block_idx", "position_idx"] + [
            f"prob_state_{i}" for i in range(n_states)
        ]
        writer.writerow(header)

        for block_idx, arr in enumerate(posterior_results):
            for pos_idx, row in enumerate(arr):
                writer.writerow([block_idx, pos_idx] + row.tolist())
    print(f"Posterior decoding complete. Results saved to {output_file}.")


if __name__ == "__main__":
    main()
