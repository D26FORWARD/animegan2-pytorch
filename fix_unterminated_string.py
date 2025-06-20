import sys
import json

def fix_specific_unterminated_string(content, problematic_char_offset):
    """
    Attempts to fix an unterminated string by REPLACING the character
    at problematic_char_offset with a double quote.
    """
    if not (0 <= problematic_char_offset < len(content)):
        print(f"Error: Character offset {problematic_char_offset} is out of bounds for file length {len(content)}.")
        return None # Indicate failure

    content_list = list(content)
    original_char_repr = repr(content_list[problematic_char_offset])

    # Replace the character at the specified offset with a quote
    content_list[problematic_char_offset] = '"'
    print(f"Replaced character {original_char_repr} at offset {problematic_char_offset} with a double quote.")

    return "".join(content_list)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_unterminated_string.py <filepath_to_modify_in_place>")
        sys.exit(1)

    filepath_to_modify = sys.argv[1]
    # This is the offset of the original newline character that was identified as problematic.
    offset_to_replace = 33134

    try:
        with open(filepath_to_modify, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath_to_modify}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filepath_to_modify}: {e}")
        sys.exit(1)

    # Check if the offset is valid for modification
    if not (0 <= offset_to_replace < len(original_content)):
        print(f"Error: Calculated offset {offset_to_replace} is out of bounds for current file length {len(original_content)}.")
        print("This might happen if the file has been modified differently than expected.")
        print("---ORIGINAL_CONTENT_START---")
        print(original_content)
        print("---ORIGINAL_CONTENT_END---")
        sys.exit(1)

    modified_content = fix_specific_unterminated_string(original_content, offset_to_replace)

    if modified_content is None:
        # In case fix_specific_unterminated_string returns None on error
        print("---ORIGINAL_CONTENT_START---")
        print(original_content) # Print original if modification failed
        print("---ORIGINAL_CONTENT_END---")
        sys.exit(1)

    # Output the modified content for the agent to use with overwrite_file_with_block
    print("---MODIFIED_CONTENT_START---")
    print(modified_content)
    print("---MODIFIED_CONTENT_END---")
