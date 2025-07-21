# -*- coding: utf-8 -*-

"""
This class displays messages in the requested color.
The allowed colors are 'red', 'green', 'yellow', 'magenta', 
 'gray' and 'white'.
If the color is not specified, the text is displayed in white.

The message can be displayed in bold.

@author: Pierre.FANCELLI
"""

# pylint: disable=too-few-public-methods
class DisplayColor:
    """
    This class displays messages in the requested color.
    """
    COLORS = {
        "RED":     "\033[31m",
        "GREEN":   "\033[32m",
        "YELLOW":  "\033[33m",
        "MAGENTA": "\033[35m",
        "WHITE":   "\033[37m",
        "GRAY":    "\033[90m",
    }
    RESET = "\033[0m"

    BOLD =   "\033[1m"

    def __init__(self):
        """
        Initializes the DisplayColor class.
        """
        pass

    def print(self, text, color="WHITE", bold=False):
        """
        Prints the text in the specified color and boldness.
        """
        color_code = self.COLORS.get(color.upper(), self.COLORS["WHITE"])
        bold_code = self.BOLD if bold else ""
        print(f"{color_code}{bold_code}{text}{self.RESET}")