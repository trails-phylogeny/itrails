import os
import sys

import yaml


class FlowSeq(list):
    """A list subclass to indicate that this sequence should be represented inline in YAML."""

    pass


def flow_seq_representer(dumper, data):
    """Custom representer that dumps FlowSeq instances in flow style."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


# Register the custom representer with PyYAML.
yaml.add_representer(FlowSeq, flow_seq_representer)


def load_config(config_file):
    """Load the YAML configuration file."""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)


def update_best_model(
    best_model_yaml, optim_variables, current_optim_params, current_result, iteration
):
    """
    Updates a YAML file that stores the best model information.

    This function checks whether the current result (typically a -log likelihood value)
    is better (i.e. lower) than the one stored in the YAML file. If it is better or if
    no best result has been stored yet, the function updates (or creates) the file with
    the following four main fields:

    - ``fixed_parameters``: A nested dictionary of the fixed parameters (written only once).
    - ``optimized_parameters``: A dictionary with keys corresponding to the optimized
      parameters (from ``optim_variables``) and their current values (from ``current_optim_params``).
    - ``results``: The best (lowest) -log likelihood value of the model.
    - ``settings``: A copy of the settings from the original model configuration.

    :param best_model_yaml: Path to the YAML file (e.g., "best_model.yaml").
    :type best_model_yaml: str
    :param fixed_params: Dictionary of fixed parameters.
    :type fixed_params: dict
    :param optim_variables: List of names for optimized parameters.
    :type optim_variables: list
    :param current_optim_params: List of current optimized parameter values.
                                 The order should correspond to the order of ``optim_variables``.
    :type current_optim_params: list
    :param current_result: The current -log likelihood value to compare against the stored value.
                           Lower values are considered better.
    :type current_result: float
    :param settings: Settings from the original model configuration.
    :type settings: dict
    """
    # Attempt to load existing best model information (if any)
    if os.path.exists(best_model_yaml):
        with open(best_model_yaml, "r") as f:
            try:
                best_model_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error loading best model file: {e}")
                sys.exit(1)
    else:
        raise FileNotFoundError(f"Best model file not found: {best_model_yaml}")

    # Retrieve the stored best result, if available
    mu = float(best_model_data["fixed_parameters"]["mu"])
    prev_loglik = best_model_data["results"]["log_likelihood"]
    update_flag = False

    # Update if no stored result exists or if the current result is better (i.e. lower)
    if prev_loglik is None or current_result > prev_loglik:
        update_flag = True

    if update_flag:
        # Build the optimized_parameters dictionary
        optim_dict = {
            vari: current_optim_params[i] for i, vari in enumerate(optim_variables)
        }
        for param, value in optim_dict.items():
            if param == "r":
                optim_dict[param] = float(value) * mu
            else:
                optim_dict[param] = float(value) / mu

        # Update the best model data dictionary with the four main fields
        best_model_data["optimized_parameters"] = optim_dict
        best_model_data["results"]["log_likelihood"] = current_result
        best_model_data["results"]["iteration"] = iteration

        with open(best_model_yaml, "w") as f:
            yaml.dump(best_model_data, f)
