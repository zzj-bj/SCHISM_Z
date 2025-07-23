# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.
@author: Pierre.FANCELLI
"""

import os

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
    display = dc.DisplayColor()
    while True:
        reponse = input(f"[?] {message} (y/n) ? : ").strip().lower()
        if reponse in ['yes', 'y']:
            return True
        if reponse in ['no',  'n']:
            return False
        text = f"Please provide a valid answer (y/n) {ct.BELL}"
        display.print(text, "red")


def validate_subfolders(data_dir, subfolders, valid_subfolders, report, folder_type='images'):
    """
    Validate the existence and content of specified subfolders.

    This function checks if the specified subfolders contain the required folder type
    (e.g., 'images' or 'masks'). It verifies if the folder exists and if it contains
    any TIFF files. If a folder is missing or empty, a message is added to the provided
    report object.

    Parameters:
    ----------
    data_dir : str
        The base directory where the subfolders are located.

    subfolders : list of str
        A list of subfolder names to validate.

    valid_subfolders : list of str
        A list that will be populated with valid subfolder names
        that contain the required files.

    report : object
        An object that has an `add` method for logging messages about missing folders
          or empty directories.

    folder_type : str, optional
        The type of folder to check for (default is 'images').
        This can be changed to 'masks' or any other type.


    Returns:
    --The function modifies the `valid_subfolders` list in place
    and adds messages to the `report`.-----

    """

    nb_files = 0
    if not os.path.isdir(data_dir):
        report.add(f" - Folder '{folder_type}' missing in : ", subfolders)
    else:
        nb_files = count_tif_files(data_dir)
        if nb_files == 0:
            report.add(f" - No file (*.tif) in folder '{folder_type}' : ", subfolders)
        else:
            valid_subfolders.append(subfolders)
    return nb_files

def get_path(prompt):
    """Requests a valid path from the user."""
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        print(f"[X] Invalid path. Try again. {ct.BELL}")

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

def print_box(text):

    # Determine the width of the box based on the string length
    box_width = len(text) + 2  # Add padding for the box

    # Create the box
    print(f" ╔{'═' * (box_width)}╗")
    print(f" ║{text.center(box_width)}║")
    print(f" ╚{'═' * (box_width)}╝")
