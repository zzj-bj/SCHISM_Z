# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:14:12 2025

@author: Pierre.FANCELLI
"""
import sys

from tools import menu as me

#============================================================================
BELL = "\a" # Sound system for Error.

class Menu :
    """
    This class allows displaying a menu and controlling the choice.
    """

    def __init__(self, sel_menu,  menu_dynamique = None):

        self.sel_menu = sel_menu
        self.menu_dynamique = menu_dynamique

        # Check selec menu
        self.unknow_menu = False
        if self.sel_menu == 'Dynamic':
            if self.menu_dynamique is None :
                print('!!! The menu is absent !!!')
                sys.exit()
            else:
                self.tableau = [str(element) for element in self.menu_dynamique]
        elif self.sel_menu in me.MENUS:
            self.tableau = [str(element) for element in me.MENUS[self.sel_menu]]
        else:
            self.unknow_menu = True

        self.select = None
        self.ligne = None
        self.vali = None

    def display_menu(self):
        """ display the menu """
        if self.unknow_menu :
            print(f" '{self.sel_menu}' : This menu doesn't existe in the dictionary !!!")
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

            if self.ligne < 2 :
                print(f" '{self.tableau[0]}' : Menu without Choise !!!")
                sys.exit()

    def selection(self):
        """
        Choice an option
        """
        while True:
            try:
                select = int(input("[?] Make your selection: "))
                if 1 <= select <= self.ligne - 1 :
                    return select
                else:
                    print(f"[X] Invalid selection. Try again ! {BELL}")
            # Input is not a number
            except ValueError:
                print(f"[X] Waiting for number. Try again ! {BELL}")


def answer_yes_or_non(message):
    """
    This function retum.
      True for yes, y, oui, o.
      False for no, non n.
     The input does not take uppercase letters into account."
    """
    while True:
        reponse = input(f"[?] {message} (Y/N) ? : ").strip().lower()
        if reponse in ['yes', 'y', 'oui', 'o']:
            return True
        elif reponse in ['no', 'non', 'n']:
            return False
        else:
            print("incorrect answer !!! ")
