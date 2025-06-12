# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:14:12 2025

@author: Pierre.FANCELLI
"""
import sys

from tools import constants as ct

#============================================================================
BELL = "\a" # Sound system for Error.

class Menu :
    """
    This class allows displaying a menu and controlling the choice.
    You can chose the style of the pattern between:
       simple, double, rounds, ASCII or Unicode.
    The default value is 'double'
    """

    def __init__(self, sel_menu,  dynamic_menu = None, style = None):

        self.sel_menu = sel_menu
        self.dynamic_menu = dynamic_menu
        self.style = style

        # Check selec menu
        self.unknow_menu = False
        if self.sel_menu == 'Dynamic':
            if self.dynamic_menu is None :
                print('!!! The menu is absent !!!')
                sys.exit()
            else:
                self.board = [str(element) for element in self.dynamic_menu]
        elif self.sel_menu in ct.MENUS:
            self.board = [str(element) for element in ct.MENUS[self.sel_menu]]
        else:
            self.unknow_menu = True

        # Select pattern
        if self.style == 'simple' :
            self.frame = ct.PATTERN['simple']
        elif self.style == 'rounds' :
            self.frame = ct.PATTERN['rounds']
        elif self.style == 'ASCII' :
            self.frame = ct.PATTERN['ASCII']
        elif self.style == 'Unicode' :
            self.frame = ct.PATTERN['Unicode']
        else:
            self.frame = ct.PATTERN['double']

        self.select = None
        self.ligne = None
        self.vali = None

    def display_menu(self):
        """ display the menu """
        if self.unknow_menu :
            print(f" '{self.sel_menu}' : This menu doesn't existe in the dictionary !!!")
            sys.exit()
        else:
            max_length = max(len(chaine) for chaine in self.board)
            self.ligne = len(self.board)
            box_width = max_length + 6

            # Create the pattern
            print(f"{self.frame[0]}{self.frame[10] * box_width}{self.frame[2]}")
            print(f"{self.frame[9]}{self.board[0].center(box_width )}{self.frame[9]}")
            print(f"{self.frame[3]}{self.frame[10] * box_width}{self.frame[5]}")
            for i in range(1, self.ligne):
                print(f"{self.frame[9]} {i} : {self.board[i].ljust(max_length)} {self.frame[9]}")
            print(f"{self.frame[6]}{self.frame[10] * box_width}{self.frame[8]}")

            if self.ligne < 2 :
                print(f" '{self.board[0]}' : Menu without Choise !!!")
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
