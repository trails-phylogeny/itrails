import os
import sys

import yaml


class FlowSeq(list):
    """
    A list subclass that indicates the sequence should be represented inline in YAML.

    This class is used to signal that the list is to be serialized using flow style (i.e. inline)
    when dumped by PyYAML.
    """

    pass


def flow_seq_representer(dumper, data):
    """
    Represent a FlowSeq instance in YAML flow style.

    This custom representer instructs PyYAML to dump FlowSeq instances using a flow-style sequence.

    :param dumper: The YAML dumper instance.
    :param data: The FlowSeq data to be represented.
    :return: A YAML node representing the sequence in flow style.
    """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


# Register the custom representer with PyYAML.
yaml.add_representer(FlowSeq, flow_seq_representer)


def load_config(config_file):
    """
    Load a YAML configuration file.

    Reads the YAML file specified by ``config_file`` and returns the parsed configuration.
    If an error occurs during file reading or parsing, the error is printed to stderr and
    the process exits.

    :param config_file: Path to the YAML configuration file.
    :type config_file: str
    :return: Parsed configuration data.
    :rtype: Any
    """
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
    Update the YAML file storing the best model information.

    This function compares the current log likelihood with the stored value in the YAML file and updates the file if the current value is better (i.e. higher than the stored value). It updates the following fields in the YAML data:

    - ``optimized_parameters``: A dictionary mapping optimized parameter names to their adjusted values. The adjustment is done by scaling each parameter using the fixed parameter ``mu`` found in the YAML file's ``fixed_parameters`` field. For the parameter named "r", the value is multiplied by ``mu``; for all other parameters, the value is divided by ``mu``.
    - ``results``: A dictionary containing the best log likelihood value and the corresponding iteration number.

    **Note:** The YAML file is expected to exist and contain the required fields (``fixed_parameters`` and ``results``). If the file does not exist or cannot be read, the function raises an error or exits.

    :param best_model_yaml: Path to the YAML file (e.g., "best_model.yaml").
    :type best_model_yaml: str
    :param optim_variables: List of names for optimized parameters.
    :type optim_variables: list
    :param current_optim_params: List of current optimized parameter values corresponding to ``optim_variables``.
    :type current_optim_params: list
    :param current_result: The current log likelihood value to compare against the stored value.
    :type current_result: float
    :param iteration: The current iteration number in the model training process.
    :type iteration: int
    """
    # Attempt to load existing best model information (if available)
    if os.path.exists(best_model_yaml):
        with open(best_model_yaml, "r") as f:
            try:
                best_model_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error loading best model file: {e}")
                sys.exit(1)
    else:
        raise FileNotFoundError(f"Best model file not found: {best_model_yaml}")

    # Retrieve the stored best result and the fixed parameter 'mu'
    mu = float(best_model_data["fixed_parameters"]["mu"])
    prev_loglik = best_model_data["results"]["log_likelihood"]
    update_flag = False

    # Update if no stored result exists or if the current result is better (i.e. higher than the stored value)
    if prev_loglik is None or current_result > prev_loglik:
        update_flag = True

    if update_flag:
        # Build the optimized_parameters dictionary by matching variables with their current values.
        optim_dict = {
            vari: current_optim_params[i] for i, vari in enumerate(optim_variables)
        }
        for param, value in optim_dict.items():
            if param == "r":
                optim_dict[param] = float(value) * mu
            else:
                optim_dict[param] = float(value) / mu

        # Update the best model data with new optimized parameters and results
        best_model_data["optimized_parameters"] = optim_dict
        best_model_data["results"]["log_likelihood"] = current_result
        best_model_data["results"]["iteration"] = iteration

        with open(best_model_yaml, "w") as f:
            yaml.dump(best_model_data, f)
