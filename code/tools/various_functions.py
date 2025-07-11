# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.
    - Waiting for a Yes or No Answer
    - Enter a percentage

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
        if reponse in ['no', 'non', 'n']:
            return False
        print("incorrect answer !!! ")

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
            print(f"Error: The value must be between 1 and 100. {ct.BELL}")
        except ValueError:
            print(f"Error: Please enter a valid number. {ct.BELL}")
