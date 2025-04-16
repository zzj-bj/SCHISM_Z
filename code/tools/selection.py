# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:14:12 2025

@author: Pierre.FANCELLI
"""

import sys

from tools import menu as me

BELL = "\a" # Son système pour Erreur
#============================================================================

class Menu :
    """
    This class display the menu
    """

    def __init__(self, sel_menu,  menu_dynamique = None):

        self.sel_menu = sel_menu
        self.menu_dynamique = menu_dynamique

        # Check selec menu
        self.unknow_menu = False
        if self.sel_menu == 'Dynamic':
            if self.menu_dynamique == None :
                print('!!! The menu is absent !!!')
                sys.exit()
            else:
                # Recover the menu and convert in 'str''
                self.tableau = [str(element) for element in self.menu_dynamique]
        elif self.sel_menu in me.MENUS:
                self.tableau = me.MENUS[self.sel_menu]
        else:
                self.unknow_menu = True

        self.select = None
        self.ligne = None
        self.vali = None

    def display_menu(self):
        """ display the menu """
        if self.unknow_menu :
            print(f" '{self.sel_menu}' : This menu doesn't existe !!!")
            sys.exit()
        else:
            try:
                longueur_max = max(len(chaine) for chaine in self.tableau)
            except TypeError:
                print("Format menu must be 'str !!!")
                sys.exit()

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
        Choise an option
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
                print(f"[X] Waiting a number. Try again ! {BELL}")


def choiice_y_n(message):
    """
    This function retum.
     True for Yes & False for No
    """
    while True:
        reponse = input(f"[?] {message} (Y/N) ? : ").strip().lower()
        if reponse in ['yes', 'y', 'oui', 'o']:
            return True
        elif reponse in ['no', 'non', 'n']:
            return False
        else:
            print("Réponse invalide !!! ")
