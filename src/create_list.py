import re


def load_input_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()  # Read each line into a list
    return lines


def parse_line(line):
    # Regular expression pattern to match the line format
    pattern = r"\(\((\d+),\s*(\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+),\s*(\d+)\)\)\s*([\d\.eE+-]+)"
    match = re.match(pattern, line)

    if match:
        # Extract integers for tuple values and float for the number
        tuple_data = (
            (int(match.group(1)), int(match.group(2)), int(match.group(3))),
            (int(match.group(4)), int(match.group(5)), int(match.group(6))),
        )
        value = float(match.group(7))
        return (tuple_data, value)
    else:
        print(f"Line could not be parsed: {line.strip()}")
        return None


def parse_file_to_tuple_list(file_path):
    lines = load_input_from_file(file_path)
    tuple_list = []

    for line in lines:
        parsed = parse_line(line.strip())  # Strip any extra whitespace
        if parsed:
            tuple_list.append(parsed)

    return tuple_list


def compare_tuple_lists(list1, list2):
    # Convert lists to dictionaries for faster lookup
    dict1 = dict(list1)
    dict2 = dict(list2)

    # Initialize dictionaries to store results
    matching_tuples = {}
    mismatched_tuples = {}
    unique_in_list1 = {}
    unique_in_list2 = {}

    # Check for tuples in list1 that match in list2
    for key, value in dict1.items():
        if key in dict2:
            if dict2[key] == value:
                matching_tuples[key] = value  # Match found
            else:
                mismatched_tuples[key] = (value, dict2[key])  # Values differ
        else:
            unique_in_list1[key] = value  # Only in list1

    # Check for tuples in list2 that are not in list1
    for key, value in dict2.items():
        if key not in dict1:
            unique_in_list2[key] = value  # Only in list2

    # Return results
    return {
        "matching_tuples": matching_tuples,
        "mismatched_tuples": mismatched_tuples,
        "unique_in_list1": unique_in_list1,
        "unique_in_list2": unique_in_list2,
    }


# Load input from file
file_path1 = "/home/davidmartin/work/trails/optim_trails_dev/src/old.txt"
file_path2 = "/home/davidmartin/work/trails/optim_trails_dev/src/new.txt"

# input_string1 = load_input_from_file(file_path1)
# input_string2 = load_input_from_file(file_path2)
# print(input_string1)
# Parse and compare lists
list1 = parse_file_to_tuple_list(file_path1)
list2 = parse_file_to_tuple_list(file_path2)


result = compare_tuple_lists(list1, list2)

print("Matching tuples:", result["matching_tuples"])
print("Mismatched tuples:", result["mismatched_tuples"])
print("Unique in list1:", result["unique_in_list1"])
print("Unique in list2:", result["unique_in_list2"])
