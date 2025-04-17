# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:28:36 2025

@author: Pierre.FANCELLI
"""

import os
from pathlib import Path

#==========================================================================
BELL = "\a" # Son système pour Erreur


def get_path(prompt):
    """Requests a valid path from the user."""
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        print(f"[X] Invalid path. Try again. {BELL}")
        # InvalidInput('Invalid path.').invalid_input()

def create_name_path(root, first, follow):
    """

    """
    return os.path.join(root, first, follow)


def compter_files(repertoire):
    chemin = Path(repertoire)
    return sum(1 for fichier in chemin.iterdir() if fichier.is_file())





