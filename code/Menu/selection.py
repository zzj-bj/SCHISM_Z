# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:14:12 2025

@author: Pierre.FANCELLI
"""

import os
import sys

from Menu import menu as me

BELL = "\a" # Son système pour Erreur

LOGO = """
╔══════════════════════════════════════════════════╗
║  ███████╗ ██████╗██╗  ██╗██╗███████╗███╗   ███╗  ║
║  ██╔════╝██╔════╝██║  ██║██║██╔════╝████╗ ████║  ║
║  ███████╗██║     ███████║██║███████╗██╔████╔██║  ║
║  ╚════██║██║     ██╔══██║██║╚════██║██║╚██╔╝██║  ║
║  ███████║╚██████╗██║  ██║██║███████║██║ ╚═╝ ██║  ║
║  ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝  ║
╠══════════════════════════════════════════════════╣
║        -Let's do some cool segmentation-         ║
╚══════════════════════════════════════════════════╝
"""

class Menu :
    """
    This classe display the menu
    """

    def __init__(self, sel_menu, select=None):
        self.sel_menu = sel_menu

        # Check if menu exsite
        self.unknow_menu = False
        if self.sel_menu in me.MENUS:
            self.tableau = me.MENUS[self.sel_menu]
        else:
            self.unknow_menu = True

        self.select = select
        self.ligne = None
        self.vali = None

    def display_menu(self):
        """ display the menu """
        if self.unknow_menu :
            print(f" '{self.sel_menu}' : This menu doesn't existe !!!")
            sys.exit()
        else:
            longueur_max = max(len(chaine) for chaine in self.tableau)
            self.ligne = len(self.tableau)

            box_width = longueur_max + 6

            # Create the frame
            print(f"╔{'═' * (box_width )}╗")
            print(f"║{self.tableau[0].center(box_width )}║")
            print(f"╠{'═' * (box_width )}╣")
            for i in range(1, self.ligne):
                print(f"║ {i} : {self.tableau[i].ljust(longueur_max)} ║")
            print(f"╚{'═' * (box_width )}╝")

    def selection(self):
        """
        Choise an option
        """
        while True:
            try:
                self.select = int(input("[?] Make your selection: "))
                if 1 <= self.select <= self.ligne - 1 :
                    return self.select
                else:
                    print(f"[X] Invalid selection. Try again ! {BELL}")
            # Input is not a number
            except ValueError:
                print(f"[X] Waiting a number. Try again ! {BELL}")


class InvalidInput:
    """ Input error """
    def __init__(self, message):
        self.message = message

    def invalid_input(self):
        print(f"[X] {self.message} Try again. {BELL}")

def get_path(prompt):
    """Requests a valid path from the user."""
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        InvalidInput('Invalid path.').invalid_input()

