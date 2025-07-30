# -*- coding: utf-8 -*-
"""
This module allows creating a menu and managing the input choice based on the selection.
You can choose the frame format of the menu.
It is possible to opt for a static menu or a dynamic menu.

@author: Pierre.FANCELLI
"""
import sys

from tools import constants as ct
from tools import display_color as dc
from tools.constants import DISPLAY_COLORS as colors

#============================================================================

class Menu :
    """
    This class allows creating a menu and managing the input choice based on the selection.
    You can choose the frame format of the menu.

    You can choose the style of the menu frame between:
    'simple', 'double', 'rounds', 'ASCII', or 'Unicode'.
    The default style is 'double'.
    """

    def __init__(self, selected_menu,  dynamic_menu = None, style = ''):

        self.selected_menu = selected_menu
        self.dynamic_menu = dynamic_menu
        self.style = style

        # Check selec menu
        self.unknown_menu = False
        if self.selected_menu == 'Dynamic':
            if self.dynamic_menu is None :
                print('!!! The menu is absent !!!')
                sys.exit()
            else:
                self.board = [str(element) for element in self.dynamic_menu]
        elif self.selected_menu in ct.MENUS:
            self.board = [str(element) for element in ct.MENUS[self.selected_menu]]
        else:
            self.unknown_menu = True

        # Select pattern
        style_mapping = {
            'simple': ct.PATTERN['simple'],
            'double' : ct.PATTERN['double'],
            'rounds': ct.PATTERN['rounds'],
        }
        self.frame = style_mapping.get(self.style, ct.PATTERN['double'])

        self.ligne = 0


    def display_menu(self):
        """        
        Method to display the menu based on the selected style and content.
        """
        if self.unknown_menu :
            print(f" '{self.sel_menu}' : This menu doesn't existe in the dictionary !!!")
            sys.exit()
        else:
            max_length = max(len(chaine) for chaine in self.board)
            self.ligne = len(self.board)
            box_width = max_length + 6

            # Create the pattern
            print(f"{self.frame[1]}{self.frame[10] * box_width}{self.frame[3]}")
            print(f"{self.frame[11]}{self.board[0].center(box_width )}{self.frame[11]}")
            print(f"{self.frame[4]}{self.frame[10] * box_width}{self.frame[6]}")
            for i in range(1, self.ligne):
                print(f"{self.frame[11]} {i} : {self.board[i].ljust(max_length)} {self.frame[11]}")
            print(f"{self.frame[7]}{self.frame[10] * box_width}{self.frame[9]}")

            if self.ligne < 2 :
                print(f" '{self.board[0]}' : Menu without Choise !!!")
                sys.exit()

    def selection(self):
        """
        Method to manage the user's choice in the menu.
        """
        display = dc.DisplayColor()
        while True:
            try:
                select = int(input("[?] Make your choice: "))
                if 1 <= select <= self.ligne - 1 :
                    return select
                text = f"[X] Oops! That selection isn't valid. Please try again! {ct.BELL}"
                display.print(text, colors['error']) 

            # Input is not a number
            except ValueError:
                text = f"[X] It looks like you didn't enter a valid number. Please try again! {ct.BELL}"
                display.print(text, colors['error'])
