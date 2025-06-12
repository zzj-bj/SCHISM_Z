# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:28:36 2025

@author: Pierre.FANCELLI
"""

import os
from pathlib import Path
from tools import constants as ct

#==========================================================================

def get_path(prompt):
    """Requests a valid path from the user."""
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        print(f"[X] Invalid path. Try again. {ct.BELL}")

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


def get_name_at_index(input_path, pos):
    """
    This function takes a file path as input.
    and returns the component of the path at the specified index.

    Parameters
    ----------
    input_path : str
        The complete file path as a string.
    pos : int
        The index of the component to retrieve from the path.

    Returns
    -------
    str
        The component of the path at the specified index.
    """
    name = input_path.split("\\")
    return name[pos]

def count_files(folder):
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

def count_tif_files(folder):
    """
    Count the number of files with extention '.tif' in the given directory.

    Parameters
    ----------
    repertoire : str
        directory where find the files.

    Returns
    -------
    Numbers of files.

    """
    files = [f for f in os.listdir(folder)
             if f.endswith(".tif")]
    return len(files)
