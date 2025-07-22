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
        "Json Generation",
        "Normalization",
        "Back to the main menu"]
}
