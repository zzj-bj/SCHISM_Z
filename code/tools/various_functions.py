# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:19:45 2025

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
        elif reponse in ['no', 'non', 'n']:
            return False
        else:
            print("incorrect answer !!! ")
