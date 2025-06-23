"""
This module provides a class for converting string parameters into their appropriate Python types.
It handles conversion for strings representing None, booleans, numbers 
and collections like lists and dictionaries.
It uses `ast.literal_eval` for safe evaluation of complex structures.
"""
import ast

class ParamConverter:
    """
    ParamConverter class for converting string parameters to appropriate Python types.
    """
    def __init__(self):
        pass

    def _convert_param(self, v):
        """
        Convert a string parameter to its appropriate Python type.

        Handles None, booleans, numbers, and collections like lists and dictionaries.
        Args:
            v (str): The string parameter to convert.
        Returns:
            The converted value in its appropriate Python type.
        """
        if isinstance(v, str):
            v = v.strip("'\"")  # Remove surrounding quotes

            # Handle None, booleans
            if v.lower() == "none":
                return None
            if v.lower() == "true":
                return True
            if v.lower() == "false":
                return False

            # Try parsing numbers directly
            try:
                if "." in v or "e" in v.lower():  # Likely a float or scientific notation
                    return float(v)
                return int(v)  # Try integer conversion
            except ValueError:
                pass  # Continue processing

            # Use `literal_eval` for safer conversion of lists, tuples, and dictionaries
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple, dict)):
                    return parsed
            except (ValueError, SyntaxError):
                pass  # Ignore errors and treat as a string

        return v # Return as a string if no conversion succeeded
