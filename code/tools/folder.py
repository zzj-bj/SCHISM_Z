# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:28:36 2025

@author: Pierre.FANCELLI
"""

import os
from pathlib import Path

#==========================================================================
BELL = "\a" # Sound system for Error.

def get_path(prompt):
    """Requests a valid path from the user."""
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        print(f"[X] Invalid path. Try again. {BELL}")
        # InvalidInput('Invalid path.').invalid_input()

def create_name_path(root, directory, subdirectory):
    """
    This function returns the full name of a directory based on a root,
    a directory, and a subdirectory.".

    Parameters
    ----------
    root : str
        directory where find the files.
    repertoire : str
        directory where find the files.
    subrepertoire : str
        directory where find the files.

    Returns
    -------
    full name.

    """
    return os.path.join(root, directory, subdirectory)


def compter_files(folder):
    """
    Count the number of files in the given directory.

    Parameters
    ----------
    repertoire : str
        directory where find the files.

    Returns
    -------
    Numbers of files.

    """
    chemin = Path(folder)
    return sum(1 for fichier in chemin.iterdir() if fichier.is_file())

def compter_tif_files(folder):

    files = [f for f in os.listdir(folder)
             if f.endswith(".tif")]
    return len(files)


