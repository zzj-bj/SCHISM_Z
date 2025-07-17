"""
This module provides a class for converting string parameters into their appropriate Python types.
It handles conversion for strings representing None, booleans, numbers 
and collections like lists and dictionaries.
It uses `ast.literal_eval` for safe evaluation of complex structures.
"""
import ast

# pylint: disable=too-few-public-methods
class ParamConverter:
    """
    ParamConverter class for converting string parameters to appropriate Python types.
    """
    def __init__(self):
        pass

    def convert_param(self, v):
        """
        Convert a string parameter to its appropriate Python type.

        Handles None, booleans, numbers, and collections like lists and dictionaries.
        Args:
            v (str): The string parameter to convert.
        Returns:
            The converted value in its appropriate Python type.
        """
        result = v
        if isinstance(v, str):
            v = v.strip("'\"")
            if v.lower() == "none":
                result = None
            elif v.lower() == "true":
                result = True
            elif v.lower() == "false":
                result = False
            else:
                try:
                    if "." in v or "e" in v.lower():
                        result = float(v)
                    else:
                        result = int(v)
                except ValueError:
                    try:
                        parsed = ast.literal_eval(v)
                        if isinstance(parsed, (list, tuple, dict)):
                            result = parsed
                    except (ValueError, SyntaxError):
                        pass
        return result
