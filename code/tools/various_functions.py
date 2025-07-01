# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.
    - Waiting for a Yes or No Answer
    - Enter a percentage

@author: Pierre.FANCELLI
"""

import os

from tools import folder as fo
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
        dc.display_color(text, 'yellow', bold=True)
        dc.display_color(' See the report at the end of the process.', 'yellow')

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
        An object that has an `add` method for logging messages about missing folders or empty directories.

    folder_type : str, optional
        The type of folder to check for (default is 'images').
        This can be changed to 'masks' or any other type.


    Returns:
    --The function modifies the `valid_subfolders` list in place
    and adds messages to the `report`.-----


    """
    for f in subfolders:
        folder_path = fo.create_name_path(data_dir, f, folder_type)

        if not os.path.isdir(folder_path):
            report.add(f" - Folder '{folder_type}' missing in : ", f)
        else:
            nb_files = fo.count_tif_files(folder_path)
            if nb_files == 0:
                report.add(f" - No file (*.tif) in folder '{folder_type}' : ", f)
            else:
                valid_subfolders.append(f)
    return nb_files
