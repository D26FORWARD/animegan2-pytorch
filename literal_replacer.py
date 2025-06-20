import re
import sys
import json

def replace_python_literals(content):
    # Order of replacement matters to avoid issues like 'None' in 'TrueNonesia' if not using word boundaries properly.
    # Using \b for word boundaries.
    # Pattern to find : None, [None, None], etc. also "key": None
    # Need to be careful not to replace these if they are within actual string content.
    # JSON strings are double-quoted. Python literals are not quoted.
    # This regex approach assumes that `None`, `True`, `False` appearing as actual data *within* JSON strings
    # are not the ones we want to replace. We only want to replace them when they are used as if they were
    # JSON `null`, `true`, `false` keywords.

    # Regex to match 'None' as a whole word, not preceded or followed by other word characters.
    # This should correctly replace 'None' when it's used as a value, e.g. "key": None, [None], etc.
    content = re.sub(r'\bNone\b', 'null', content)
    content = re.sub(r'\bTrue\b', 'true', content)
    content = re.sub(r'\bFalse\b', 'false', content)

    return content

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python literal_replacer.py <filepath>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)

    modified_content = replace_python_literals(original_content)

    # Overwrite the original file with the modified content
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"Successfully replaced literals in {filepath}")
    except Exception as e:
        print(f"Error writing modified content to {filepath}: {e}")
        sys.exit(1)

    # Validate after replacement
    try:
        json.loads(modified_content)
        print(f"VALIDATION SUCCESS: {filepath} is now valid JSON.")
    except json.JSONDecodeError as e:
        print(f"VALIDATION ERROR for {filepath} after replacement: {e}")

    # As an alternative to overwriting directly, the script could print the modified content to stdout
    # and the agent could use overwrite_file_with_block.
    # For this task, direct overwrite is requested by "Save the modified content back to the file."
    # print(modified_content) # This would be for the agent to capture for overwrite_file_with_block

    # For the agent's flow, the validation part is better done by validate_notebook.py
    # So this script will just do the replacement and save.
    # The agent will then call validate_notebook.py.
    # Re-adjusting: the script will just print the modified content for the agent to handle.

    # --- Re-adjusting for agent flow ---
    # The agent will read, then call this script to get modified content, then use overwrite_file_with_block.

    # Let's make this script output the modified content for the agent.
    # The agent will then handle file I/O via its tools.
    # This means the __main__ part of this script is more for testing it,
    # the agent will primarily call the function `replace_python_literals`.
    # For this problem, let's assume the agent can provide content to this script
    # or the script reads, modifies, and prints to stdout for the agent to capture.
    # The task description says "Apply this script to modify the file content." and "Save the modified content".
    # This implies the script should do the saving, or the agent uses tools.
    # Given the agent's toolset, it's better if this script prints to stdout, and agent uses overwrite.

    # Final decision for script: it will take a filepath, read it, transform content, and print transformed content.
    # The agent will then use `overwrite_file_with_block`.
    # This avoids this script needing write permissions directly in a way that bypasses agent's file tools.

    # --- Adjusted __main__ for agent flow ---
    # This script will now be called with the *content* as an argument, not a filepath.
    # Or, more simply, the agent reads the file, calls the function, then uses overwrite_file_with_block.
    # The current script structure (taking filepath, reading, writing) is okay IF the agent calls it via bash
    # and the script has permissions. Let's stick to the problem's "Apply this script... Save the modified content".
    # This implies the script itself should try to save.
    # The agent can run `python literal_replacer.py <filepath>`

    # The previous __main__ block that reads and writes is fine.
    # The validation part in this script is redundant if the agent calls validate_notebook.py separately.
    # Let's remove the validation from this script to keep it focused.

# Adjusted __main__ for agent flow (script modifies file directly)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python literal_replacer.py <filepath_to_modify_in_place>")
        sys.exit(1)

    filepath_to_modify = sys.argv[1]

    try:
        with open(filepath_to_modify, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath_to_modify}")
        # Agent should handle this, but script will exit.
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filepath_to_modify}: {e}")
        sys.exit(1)

    modified_content = replace_python_literals(original_content)

    if original_content == modified_content:
        print(f"No literals found needing replacement in {filepath_to_modify}.")
    else:
        try:
            with open(filepath_to_modify, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"Successfully replaced literals and saved {filepath_to_modify}.")
        except Exception as e:
            print(f"Error writing modified content to {filepath_to_modify}: {e}")
            sys.exit(1)

    # The agent will call validate_notebook.py separately.
