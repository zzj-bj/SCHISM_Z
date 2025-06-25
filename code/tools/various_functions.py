# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.

@author: Pierre.FANCELLI
"""

def answer_yes_or_non(message):
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
    This function allows you to enter a percentage value between 1 and 100.

    It returns the value as a decimal (e.g., 50% becomes 0.5).
    
    If the input is not a valid number or is outside the range, it prompts the user
    """
    while True:
        try:
            value = float(input(f"[?] {message} : "))
            if 1 <= value <= 100:
                return value / 100
            else:
                print("Error: The value must be between 1 and 100.")
        except ValueError:
            print("Error: Please enter a valid number.")

