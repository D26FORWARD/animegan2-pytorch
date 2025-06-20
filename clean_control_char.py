import sys
import json

def clean_and_validate_notebook(filepath, char_offset_to_remove):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    if not (0 <= char_offset_to_remove < len(content)):
        print(f"Error: Character offset {char_offset_to_remove} is out of bounds for file length {len(content)}.")
        # Try to parse original content anyway to see the original error
        try:
            json.loads(content)
            print(f"{filepath} is already valid JSON.")
            return content # Return original content if it's somehow already valid
        except json.JSONDecodeError as e_orig:
            print(f"Original JSON Parsing Error for {filepath}: {e_orig}")
            return None


    # Remove the character at the specified offset
    cleaned_content_list = list(content)
    removed_char_repr = repr(cleaned_content_list[char_offset_to_remove])
    del cleaned_content_list[char_offset_to_remove]
    cleaned_content = "".join(cleaned_content_list)

    print(f"Attempting to clean char {removed_char_repr} at offset {char_offset_to_remove}.")

    try:
        json.loads(cleaned_content)
        print(f"Successfully parsed {filepath} after removing character at offset {char_offset_to_remove}.")
        # Return the cleaned content so it can be used to overwrite the file
        return cleaned_content
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error for {filepath} after cleaning attempt (removed char at {char_offset_to_remove}): {e}")
        # Print info about the new error location
        # The error 'e' has 'pos', 'lineno', 'colno'
        print(f"New error at: line {e.lineno}, column {e.colno}, char_offset {e.pos}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_control_char.py <filepath> <char_offset_to_remove>")
    else:
        filepath_arg = sys.argv[1]
        try:
            offset_arg = int(sys.argv[2])
            cleaned_json_content = clean_and_validate_notebook(filepath_arg, offset_arg)
            if cleaned_json_content:
                # Instead of printing the whole content, which can be huge,
                # we'll just confirm it's ready and then use overwrite_file_with_block in the next step.
                print(f"File {filepath_arg} processed. Cleaned content is ready if parsing succeeded.")
                # To actually use it with overwrite_file_with_block, the agent would capture this output.
                # For now, a success message indicates the script logic itself worked.
                # If this script were to be used directly to fix, it would write to the file.
                # For this agent, it should output the content to be used by overwrite_file_with_block.
                # However, printing huge content here is problematic.
                # So, the agent will have to re-read the file, then this script will output the cleaned version
                # which the agent then puts into overwrite_file_with_block.
                # For this turn, let's just confirm if cleaning works.
                # A real fix would be:
                # with open(filepath_arg, 'w', encoding='utf-8') as f:
                # f.write(cleaned_json_content)
                # print(f"File {filepath_arg} overwritten with cleaned content.")
        except ValueError:
            print("Error: char_offset_to_remove must be an integer.")
