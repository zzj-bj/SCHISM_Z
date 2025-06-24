# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.

@author: Pierre.FANCELLI
"""

from tools import constants as ct

#==============================================================================
def answer_yes_or_no(message):
    """
    This function retum.
      True for yes, y, oui, o.
      False for no, non n.
     The input does not take uppercase letters into account."
    """
    while True:
        reponse = input(f"[?] {message} (y/n) ? : ").strip().lower()
        if reponse in ['yes', 'y', 'oui', 'o']:
            return True
        elif reponse in ['no', 'non', 'n']:
            return False
        else:
            print(f"incorrect answer !!! {ct.BELL}")

def input_percentage(message):
    """
    This function returns a real number between 1 and 100%
    """
    while True:
        try:
            value = float(input(f"[?] {message} : "))
            if 1 <= value <= 100:
                return value / 100
            else:
                print(f"Error: The value must be between 1 and 100. {ct.BELL}")
        except ValueError:
            print(f"Error: Please enter a valid number. {ct.BELL}")
