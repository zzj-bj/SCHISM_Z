# -*- coding: utf-8 -*-
"""
Set of constants used in the program:
    - Logo for SCHISM
    - Sound system for Error
    - symbols for frame creation
    - Constants for displaying menus

@author: Pierre.FANCELLI
"""
#------------------------------------------------------------------

# Logo for SCHISM
LOGO_IN = """
╔══════════════════════════════════════════════════╗
║  ███████╗ ██████╗██╗  ██╗██╗███████╗███╗   ███╗  ║
║  ██╔════╝██╔════╝██║  ██║██║██╔════╝████╗ ████║  ║
║  ███████╗██║     ███████║██║███████╗██╔████╔██║  ║
║  ╚════██║██║     ██╔══██║██║╚════██║██║╚██╔╝██║  ║
║  ███████║╚██████╗██║  ██║██║███████║██║ ╚═╝ ██║  ║
║  ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝  ║
╠══════════════════════════════════════════════════╣
║        -Let's do some cool segmentations-        ║
╚══════════════════════════════════════════════════╝
"""

LOGO_OUT = """
╔══════════════════════════════════════════════════╗
║  ███████╗ ██████╗██╗  ██╗██╗███████╗███╗   ███╗  ║
║  ██╔════╝██╔════╝██║  ██║██║██╔════╝████╗ ████║  ║
║  ███████╗██║     ███████║██║███████╗██╔████╔██║  ║
║  ╚════██║██║     ██╔══██║██║╚════██║██║╚██╔╝██║  ║
║  ███████║╚██████╗██║  ██║██║███████║██║ ╚═╝ ██║  ║
║  ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝  ║
╠══════════════════════════════════════════════════╣
║       -That's all folks!-  See You soon-         ║
╚══════════════════════════════════════════════════╝
"""

# Sound system for Error.
BELL = "\a"


# symbols for frame creation
PATTERN ={
"double" : [".","╔", "╦", "╗",
                "╠", "╬", "╣",
                "╚", "╩", "╝",
            "═", "║"
           ],
"simple" : [".","┌", "┬", "┐",
                "├", "┼", "┤",
                "└", "┴", "┘",
            "─", "│"
           ],
"rounds" : [".","╭", "┬", "╮",
                "├", "┼", "┤",
                "╰", "┴", "╯",
            "─", "│"
           ],
    }


# Constants for displaying menus
# MAIN MENU_   : Represents the main menu options
# PREPROCESSING : Represents the preprocessing menu options
MENUS = {
    'MAIN': [
        "MAIN MENU",
        "Preprocessing",
        "Training",
        "Inference",
        "Quit"    ],
    'Preprocessing': [
        "PREPROCESSING",
        "Json generation",
        "Normalization",
        "Back to the main menu"]
}


# Code colors for display
DISPLAY_COLORS = {
    'error': (204, 51, 0),
    'warning': (204, 204, 0),
    'input': (153, 204, 51),
    'ok': (51, 153, 0),
    'text': (255, 255, 255),
}
