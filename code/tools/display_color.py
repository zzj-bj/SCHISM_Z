# -*- coding: utf-8 -*-
"""
This class displays messages in the requested color.
The allowed colors are 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'gray', and 'white'.
If the color is not specified, the text is displayed in white.

@author: Pierre.FANCELLI
"""

class DisplayColor:

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
    }
    RESET = "\033[0m"

    def __init__(self, text, color="WHITE"):
        self.text = text
        self.color = self.COLORS.get(color.upper(), self.COLORS["WHITE"])

    def display(self):
        """Display the text in the specified color."""
        print(f"{self.color}{self.text}{self.RESET}")

def display_color(text, color="WHITE"):
    """
    Display a text with color.
    The allowed colors are 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'gray', and 'white'.
    If the color is not specified, the text is displayed in white.
    """
    display = DisplayColor(text, color)
    display.display()
