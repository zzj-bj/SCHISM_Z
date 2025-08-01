# -*- coding: utf-8 -*-

"""
This class displays messages in the requested color.
The allowed colors are 'red', 'green', 'yellow', 'magenta', 
 'gray' and 'white'.
If the color is not specified, the text is displayed in white.

The message can be displayed in bold.

Behavior:
If the RGB value matches predefined colors for error, warning, or success,
 it prepends a corresponding symbol to the message:
- Error: "[X] "
- Warning: "[!] "
- Success: "[√] "   

@author: Pierre.FANCELLI
"""


class DisplayColor:
    """
    This class displays messages in the requested color.
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self):
        """
        Initializes the DisplayColor class.
        """


    def print(self, text, rgb=(255, 255, 255), bold=False):
        """
        Prints the text in the specified RGB color and boldness.
        """

        rgb_code = f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        bold_code = self.BOLD if bold else ""
        print(f"{rgb_code}{bold_code}{rgb[3]}{text}{self.RESET}")
        