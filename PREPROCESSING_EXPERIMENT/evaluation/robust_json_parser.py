"""
Robust JSON Parser for Robot Command Mapping Models.

Provides safe, streaming extraction of the first valid JSON dictionary object from
raw LLM generation strings using `json.JSONDecoder().raw_decode()`.

Key Features:
- Handles extra prefix text and suffix hallucinations/repetitions safely.
- Stops immediately at the closing brace of the first valid JSON object.
- Rejects non-standard constants (NaN, Infinity, -Infinity).
- Scans subsequent candidate opening braces if a leading '{' is malformed.
- Validates that the parsed root entity is a JSON Object (dict).
- Distinguishes parsing failures cleanly from semantic mismatches without inventing fields.
"""

import json
from typing import Tuple, Optional, Any, Dict


def _reject_non_standard_constant(constant_name: str) -> None:
    """Callback for json.JSONDecoder to disallow NaN, Infinity, -Infinity."""
    raise ValueError(f"Non-standard JSON constant rejected: {constant_name}")


# Pre-configured decoder that rejects non-standard numeric constants
_DECODER = json.JSONDecoder(parse_constant=_reject_non_standard_constant)


def parse_robot_json(raw_text: Any) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """
    Safely extract and parse the first valid JSON dictionary from raw model output.

    Parameters
    ----------
    raw_text : Any
        The raw output string from the language model.

    Returns
    -------
    Tuple[Optional[Dict[str, Any]], bool, str]
        - parsed_dict: The parsed Python dictionary, or None if extraction/parsing failed.
        - is_valid: True if a valid JSON dictionary was parsed, False otherwise.
        - error_message: An empty string if valid, or a descriptive error message on failure.
    """
    if raw_text is None:
        return None, False, "Input is None"

    if not isinstance(raw_text, str):
        # Convert non-string if possible, or fail if completely unexpected
        raw_text = str(raw_text)

    text = raw_text.strip()
    if not text:
        return None, False, "Input text is empty"

    # Handle optional markdown code blocks if the opening fence occurs before JSON
    clean_text = text
    if "```json" in clean_text:
        # Check if ```json precedes the first '{'
        json_fence_idx = clean_text.find("```json")
        first_brace = clean_text.find("{")
        if first_brace == -1 or json_fence_idx < first_brace:
            clean_text = clean_text[json_fence_idx + len("```json"):].strip()
    elif "```" in clean_text:
        fence_idx = clean_text.find("```")
        first_brace = clean_text.find("{")
        if first_brace == -1 or fence_idx < first_brace:
            clean_text = clean_text[fence_idx + len("```"):].strip()

    # Search for candidate '{' positions
    start_pos = 0
    last_error = "No opening brace found in text"

    while True:
        brace_idx = clean_text.find('{', start_pos)
        if brace_idx == -1:
            break

        try:
            parsed_obj, end_idx = _DECODER.raw_decode(clean_text, idx=brace_idx)
            
            # Ensure the parsed root is a dictionary (JSON object)
            if not isinstance(parsed_obj, dict):
                last_error = f"Parsed JSON root is {type(parsed_obj).__name__}, expected dict"
                start_pos = brace_idx + 1
                continue

            return parsed_obj, True, ""

        except (json.JSONDecodeError, ValueError) as err:
            last_error = f"JSON decode error: {str(err)}"
            start_pos = brace_idx + 1

    # If markdown strip missed something or failed, fallback to searching original text
    if clean_text != text:
        start_pos = 0
        while True:
            brace_idx = text.find('{', start_pos)
            if brace_idx == -1:
                break
            try:
                parsed_obj, end_idx = _DECODER.raw_decode(text, idx=brace_idx)
                if isinstance(parsed_obj, dict):
                    return parsed_obj, True, ""
                start_pos = brace_idx + 1
            except (json.JSONDecodeError, ValueError) as err:
                last_error = f"JSON decode error: {str(err)}"
                start_pos = brace_idx + 1

    return None, False, last_error
