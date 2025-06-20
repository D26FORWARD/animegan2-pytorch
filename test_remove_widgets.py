import json
import sys

def test_without_widgets(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Original JSON Parsing Error for {filepath}: {e}")
        # We already know this fails, so we continue to try removing widgets
        pass # Fall through to try removing widgets

    # Try to remove metadata.widgets
    # This is a crude way, assuming 'data' might be a dict from a partially successful parse
    # or we just operate on string level if full parse failed.
    # For robustness, it's better to work with the string if full parsing is already known to fail.

    # Let's try a string replacement approach first, then a dict based one if initial load works

    modified_content = None
    if '"widgets": {' in content:
        # This is a very basic removal and might break if "widgets": { appears elsewhere
        # A more robust way would use regex or parse and reconstruct
        start_index = content.rfind('"widgets": {') # find last occurrence, likely in metadata
        if start_index != -1:
            # Find the corresponding closing brace for this widgets block
            brace_level = 0
            end_index = -1
            # Start searching from the opening brace of "widgets": {
            search_after_widgets_key = content.find('{', start_index + len('"widgets": {') -1)
            if search_after_widgets_key != -1:
                for i in range(search_after_widgets_key, len(content)):
                    if content[i] == '{':
                        brace_level += 1
                    elif content[i] == '}':
                        brace_level -= 1
                        if brace_level == 0:
                            end_index = i + 1 # include the closing brace
                            break

            if end_index != -1:
                # Need to be careful about removing the comma if 'widgets' was not the last key
                # This is getting complicated. Let's try removing the whole metadata.widgets section
                # by loading, deleting, and re-dumping if the initial load works.
                # If initial load fails, this string manipulation is too risky.

                # Given that initial load fails, let's try a simpler string removal
                # This will remove the "widgets" key and its value up to the next key or end of object
                # This is still risky and might create invalid JSON if not done carefully
                pass # String manipulation is too risky for now.

    # If we have parsed data (even if original failed, maybe a more tolerant parser could work)
    # For now, since `json.loads` is strict, if it fails, `data` is not populated.
    # The problem is that `json.loads()` itself fails, so we can't easily manipulate the dict.

    # The jq error "Invalid numeric literal at line 88, column 36"
    # Line 88: "736138a274cc431b943753e139c6523f",
    # Column 36 is the comma.
    # This suggests the string "736138a274cc431b943753e139c6523f" might be misinterpreted
    # or something before it is causing a cascading failure.

    # Let's try to fix the specific error jq mentioned if it's a trailing comma issue.
    # jq pointed to line 88, column 36. That's the comma in:
    # "736138a274cc431b943753e139c6523f",
    # If this was *mistakenly* the last item in an array, the comma would be an error.
    # But it's not the last item in "referenced_widgets".

    # "Invalid numeric literal" is the key from jq.
    # This could happen if there's something like `077` (octal-like but not valid in JSON unless it's just `0` or `77`).
    # Or if a number is not formatted correctly e.g. `1.2.3` or `1e--2`.
    # The line 88 is a string.
    # What if there is a unicode character that looks like a number but isn't, or some other encoding issue?

    # The python json parser error was "Expecting value".
    # This means it read a comma (expecting another value in a list/object) but found something that isn't a value (e.g. a closing bracket, or nothing).
    # Or it read a key, then a colon, then expected a value but found something else.

    # Let's assume the file has an encoding issue or a problematic character.
    # The `read_files` tool should return UTF-8.
    # The file itself might not be pure ASCII.

    # What if I replace the content of `referenced_widgets` with an empty list?
    try:
        data = json.loads(content) # Try loading again to work with dict
        # If this works, the previous error was a fluke or my reasoning was wrong
        # (unlikely given multiple attempts)
        print(f"Loaded {filepath} successfully after all?")
        return

    except json.JSONDecodeError as e:
        # This is expected.
        # Let's try to find the referenced_widgets part via string manipulation and replace it.
        # This is fragile.
        import re
        # This regex will find "referenced_widgets": [ ... ]
        # It assumes the array doesn't contain nested brackets a specific way, which is true for this key.
        pattern = r'("referenced_widgets":\s*\[)([^\]]*?)(\])'

        def replacer(match):
            # match.group(1) is "referenced_widgets": [
            # match.group(3) is ]
            # We replace the content in between with nothing, making it an empty list.
            return match.group(1) + match.group(3)

        modified_content_str, num_replacements = re.subn(pattern, replacer, content)

        if num_replacements > 0:
            print(f"Attempting to parse {filepath} after replacing content of 'referenced_widgets' with an empty list.")
            try:
                cleaned_data = json.loads(modified_content_str)
                print(f"{filepath} (with empty referenced_widgets) parsed successfully!")
                # Now, check structure on this cleaned data
                # (Copy relevant checks from validate_notebook.py)
                if not isinstance(cleaned_data, dict):
                    print(f"Validation Error: Root is not a JSON object.")
                elif "cells" not in cleaned_data:
                    print(f"Validation Error: Root does not contain a 'cells' key.")
                # ... (add more checks as needed)
                else:
                    print("Basic structure (root, cells key) is okay for the modified content.")

            except json.JSONDecodeError as e_cleaned:
                print(f"JSON Parsing Error for {filepath} (with empty referenced_widgets): {e_cleaned}")
                # This means the error is likely outside referenced_widgets or the regex was bad

                # Let's try removing the entire "colab": { ... } block from metadata of cells
                # if "colab": { exists in a cell's metadata

                # This is getting too complex for a single script.
                # The original error points to line 88.
                # "Invalid numeric literal" by jq, at the comma.
                # "Expecting value" by python, near the end of the string preceding the comma.

                # This could be an encoding problem with ONE specific character in that string on line 88
                # or a preceding string in that array.

                # Let's try to print the problematic string from line 88 with escaped characters.
                lines = content.splitlines()
                if len(lines) >= 88:
                    problem_line_content = lines[87] # 0-indexed
                    print(f"Problematic line (88) content: {problem_line_content!r}")
                    # !r will show escapes

                    # Try to parse just that string, or a small part of the JSON around it.
                    # Example: '{"key": "value"}'
                    # The string on line 88 is: "736138a274cc431b943753e139c6523f"
                    # Let's test if this specific string is problematic in isolation
                    test_str_json = f'{{"test": {problem_line_content.strip()}}}' # remove comma for test
                    if test_str_json.endswith(","):
                         test_str_json = test_str_json[:-1]
                    print(f"Testing isolated string JSON: {test_str_json!r}")
                    try:
                        json.loads(test_str_json)
                        print("Isolated string from line 88 parses fine as part of a simple JSON.")
                    except json.JSONDecodeError as e_iso:
                        print(f"Isolated string from line 88 fails to parse: {e_iso}")
                        # This would indicate the string itself has an issue.
        else:
            print("Could not find 'referenced_widgets' using regex, or it was already empty.")
            # This means the error is likely elsewhere.


if __name__ == "__main__":
    if len(sys.argv) > 1:
        notebook_filepath = sys.argv[1]
        test_without_widgets(notebook_filepath)
    else:
        print("Error: No notebook filepath provided as argument.")
