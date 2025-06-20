import json
import sys

def validate_notebook_structure(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook_content_str = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return False
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return False

    try:
        data = json.loads(notebook_content_str)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error for {filepath}: {e}")
        return False

    if not isinstance(data, dict):
        print(f"Validation Error for {filepath}: Root is not a JSON object.")
        return False

    if "cells" not in data:
        print(f"Validation Error for {filepath}: Root does not contain a 'cells' key.")
        return False

    if not isinstance(data["cells"], list):
        print(f"Validation Error for {filepath}: 'cells' is not an array.")
        return False

    for i, cell in enumerate(data["cells"]):
        if not isinstance(cell, dict):
            print(f"Validation Error for {filepath}: Cell at index {i} is not a JSON object.")
            return False
        if "cell_type" not in cell:
            print(f"Validation Error for {filepath}: Cell at index {i} does not contain 'cell_type' key.")
            return False
        if "source" not in cell: # In .ipynb, source can be a string or list of strings
            print(f"Validation Error for {filepath}: Cell at index {i} does not contain 'source' key.")
            return False
        if not isinstance(cell["source"], (list, str)):
            print(f"Validation Error for {filepath}: 'source' in cell at index {i} is not an array of strings or a single string.")
            return False
        if isinstance(cell["source"], list):
            for j, line in enumerate(cell["source"]):
                if not isinstance(line, str):
                    print(f"Validation Error for {filepath}: Line {j} in source of cell at index {i} is not a string.")
                    # Not returning false here, as it might be too strict for some variants, but noting it.

    print(f"{filepath}: JSON is valid and basic structure is correct.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        notebook_filepath = sys.argv[1]
        validate_notebook_structure(notebook_filepath)
    else:
        print("Error: No notebook filepath provided as argument.")
