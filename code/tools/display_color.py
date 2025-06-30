# -*- coding: utf-8 -*-
"""
This class displays messages in the requested color.
The allowed colors are 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan',
 'gray', orange and 'white'.
If the color is not specified, the text is displayed in white.

The message can be displayed in bold.

@author: Pierre.FANCELLI
"""

class DisplayColor:
    """
    This class displays messages in the requested color.
    """
    COLORS = {
        "BLACK":   "\033[30m",
        "RED":     "\033[31m",
        "GREEN":   "\033[32m",
        "YELLOW":  "\033[33m",
        "BLUE":    "\033[34m",
        "MAGENTA": "\033[35m",
        "CYAN":    "\033[36m",
        "WHITE":   "\033[37m",
        "GRAY":    "\033[90m",
        "ORANGE":  "\033[38;5;214m"
    }
    RESET = "\033[0m"

    BOLD =   "\033[1m"
    ITALIC = "\033[3m"


    def __init__(self, text, color="WHITE", bold=False):
        self.text = text
        self.color = self.COLORS.get(color.upper(), self.COLORS["WHITE"])
        self.bold = self.BOLD if bold else ""

    def display(self):
        """Display the text in the specified color."""
        print(f"{self.color}{self.bold}{self.text}{self.RESET}")

def display_color(text, color="WHITE", bold=False):
    """
    Display a text with color.
    The allowed colors are 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan',
    'gray', 'white', and 'orange'.
    If the color is not specified, the text is displayed in white.
    The message can be displayed in bold.
    """
    display = DisplayColor(text, color, bold)
    display.display()
