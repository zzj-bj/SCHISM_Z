# -*- coding: utf-8 -*-
"""
This module allows creating a menu and managing the input choice based on the selection.
You can choose the frame format of the menu.
It is possible to opt for a static menu or a dynamic menu.

@author: Pierre.FANCELLI
"""
import sys

from tools import constants as ct

#============================================================================

class Menu :
    """
    This class allows displaying a menu and controlling the choice.
    You can chose the style of the pattern between:
       simple, double, rounds, ASCII or Unicode.
    The default value is 'double'
    """

    def __init__(self, selected_menu,  dynamic_menu = None, style = None):

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
            'rounds': ct.PATTERN['rounds'],
            'ASCII': ct.PATTERN['ASCII'],
            'Unicode': ct.PATTERN['Unicode'],
            'None': ct.PATTERN['double']
        }
        # self.frame = style_mapping.get(self.style, ct.PATTERN['double'])
        self.frame = style_mapping.get(str(self.style))

        self.ligne = 0


    def display_menu(self):
        """ display the menu """
        if self.unknown_menu :
            print(f" '{self.selected_menu}' : This menu doesn't existe in the dictionary !!!")
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
        Choice an option
        """
        while True:
            try:
                select = int(input("[?] Make your choice: "))
                if 1 <= select <= self.ligne - 1 :
                    return select
                else:
                    print(f"[X] Invalid choice. Try again ! {ct.BELL}")
            # Input is not a number
            except ValueError:
                print(f"[X] Waiting for number. Try again ! {ct.BELL}")
