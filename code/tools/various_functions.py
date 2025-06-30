# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.
    - Waiting for a Yes or No Answer
    - Enter a percentage

@author: Pierre.FANCELLI
"""

from tools import constants as ct
from tools import display_color as dc

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
    This function returns a real number between 0 and 1
    that corresponds to a percentage.
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

def warning_message(folder_1, folder_2):
    """
    This function displays a warning message if directories have been removed
    because they did not comply with the rules.
    """
    warning = len(folder_1) - len(folder_2)
    if warning != 0 :
        text = f" {warning} directories removed because they do not comply the rules"
        dc.display_color(text, 'ORANGE', bold=True)
        dc.display_color(' See the report at the end of the process.', 'ORANGE')
