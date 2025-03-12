def validate_time_params(fixed_dict, optim_dict):

    # Found values tracking
    found_values = set()
    time_list = []
    optim_list = []
    bounds_list = []
    final_fixed = {}

    # Function to process parameters
    def process_parameter(param):
        if param in fixed_dict:
            found_values.add(param)
            return fixed_dict[param], None, None, True  # Value, min, max, fixed=True
        elif param in optim_dict:
            found_values.add(param)
            return (
                optim_dict[param][0],
                optim_dict[param][1],
                optim_dict[param][2],
                False,
            )  # Value, min, max, fixed=False
        return None, None, None, None  # Not found

    # Process each parameter
    t_1, t_1_min, t_1_max, t_1_fixed = process_parameter("t1")
    t_A, t_A_min, t_A_max, t_A_fixed = process_parameter("tA")
    t_B, t_B_min, t_B_max, t_B_fixed = process_parameter("tB")
    t_C, t_C_min, t_C_max, t_C_fixed = process_parameter("tC")

    # Define the set of allowed combinations
    allowed_combinations = {
        frozenset(["tA", "tB", "tC"]),
        frozenset(["t1", "tA"]),
        frozenset(["t1", "tB"]),
        frozenset(["t1", "tC"]),
        frozenset(["tA", "tB"]),
        frozenset(["tA", "tC"]),
        frozenset(["tB", "tC"]),
        frozenset(["t1"]),
    }

    # Check if found_values is an allowed combination
    if frozenset(found_values) not in allowed_combinations:
        raise ValueError(f"Invalid combination of t values: {found_values}")

    # Check which valid combination exists and append accordingly
    if frozenset(found_values) == frozenset(["t1"]):
        if t_1_fixed:
            final_fixed["t1"] = t_1
        else:
            time_list.append("t1")
            optim_list.append("t1")
            bounds_list.append((t_1_min, t_1_max))

    elif frozenset(found_values) == frozenset(["tA", "tB", "tC"]):
        if t_A_fixed:
            final_fixed["tA"] = t_A
        else:
            time_list.append("tA")
            optim_list.append("tA")
            bounds_list.append((t_A_min, t_A_max))

        if t_B_fixed:
            final_fixed["tB"] = t_B
        else:
            time_list.append("tB")
            optim_list.append("tB")
            bounds_list.append((t_B_min, t_B_max))

        if t_C_fixed:
            final_fixed["tC"] = t_C
        else:
            time_list.append("tC")
            optim_list.append("tC")
            bounds_list.append((t_C_min, t_C_max))

    elif frozenset(found_values) == frozenset(["t1", "tA"]):
        if t_1_fixed:
            final_fixed["t1"] = t_1
        else:
            time_list.append("t1")
            optim_list.append("t1")
            bounds_list.append((t_1_min, t_1_max))

        if t_A_fixed:
            final_fixed["tA"] = t_A
        else:
            time_list.append("tA")
            optim_list.append("tA")
            bounds_list.append((t_A_min, t_A_max))

    elif frozenset(found_values) == frozenset(["t1", "tB"]):
        if t_1_fixed:
            final_fixed["t1"] = t_1
        else:
            time_list.append("t1")
            optim_list.append("t1")
            bounds_list.append((t_1_min, t_1_max))

        if t_B_fixed:
            final_fixed["tB"] = t_B
        else:
            time_list.append("tB")
            optim_list.append("tB")
            bounds_list.append((t_B_min, t_B_max))

    elif frozenset(found_values) == frozenset(["t1", "tC"]):
        if t_1_fixed:
            final_fixed["t1"] = t_1
        else:
            time_list.append("t1")
            optim_list.append("t1")
            bounds_list.append((t_1_min, t_1_max))

        if t_C_fixed:
            final_fixed["tC"] = t_C
        else:
            time_list.append("tC")
            optim_list.append("tC")
            bounds_list.append((t_C_min, t_C_max))

    elif frozenset(found_values) == frozenset(["tA", "tB"]):
        if t_A_fixed:
            final_fixed["tA"] = t_A
        else:
            time_list.append("tA")
            optim_list.append("tA")
            bounds_list.append((t_A_min, t_A_max))

        if t_B_fixed:
            final_fixed["tB"] = t_B
        else:
            time_list.append("tB")
            optim_list.append("tB")
            bounds_list.append((t_B_min, t_B_max))

    elif frozenset(found_values) == frozenset(["tA", "tC"]):
        if t_A_fixed:
            final_fixed["tA"] = t_A
        else:
            time_list.append("tA")
            optim_list.append("tA")
            bounds_list.append((t_A_min, t_A_max))

        if t_C_fixed:
            final_fixed["tC"] = t_C
        else:
            time_list.append("tC")
            optim_list.append("tC")
            bounds_list.append((t_C_min, t_C_max))

    elif frozenset(found_values) == frozenset(["tB", "tC"]):
        if t_B_fixed:
            final_fixed["tB"] = t_B
        else:
            time_list.append("tB")
            optim_list.append("tB")
            bounds_list.append((t_B_min, t_B_max))

        if t_C_fixed:
            final_fixed["tC"] = t_C
        else:
            time_list.append("tC")
            optim_list.append("tC")
            bounds_list.append((t_C_min, t_C_max))
    case = frozenset(found_values)
    return case, final_fixed, time_list, optim_list
