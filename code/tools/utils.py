# -*- coding: utf-8 -*-
"""
Collection of non-specific functions used in the program.
@author: Pierre.FANCELLI
"""

import os
from colorama import init, Style

from tools import constants as ct
from tools import display_color as dc
from tools.constants import DISPLAY_COLORS as colors

# Initialiser colorama
init(autoreset=True)

#==============================================================================
def rgb_to_ansi(rgb):
    """Convert RGB color to ANSI escape code."""
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"

def chck_color(color_key):
    """
    Check if the color key exists in the DISPLAY_COLORS dictionary.
    If it does, return the corresponding RGB value.
    If not, return a default color (Light Green).
    """
    try:
        color = colors[color_key]
    except KeyError:
        color = (153, 204, 51)  # Default to Light Green

    # Check if the color is a valid RGB tuple.
    if not (isinstance(color, tuple) and len(color) == 3 and
             all(isinstance(c, int) and 0 <= c <= 255 for c in color)):
        color = (153, 204, 51)

    return color



def get_path(prompt):
    """Requests a valid path from the user."""
    display = dc.DisplayColor()
    while True:
        path = input(f"[?] {prompt}: ").strip()
        if os.path.exists(path):
            return path
        text = f"[X] Invalid path: {path}. Please try again. {ct.BELL}"
        display.print(text, colors['error'])

def get_path_color(prompt, color_key='input'):
    """
    Requests a valid path from the user.
    Displays the prompt in the specified color.
    If the specified color key is invalid, the prompt will be displayed in Light Green.
    """
    display = dc.DisplayColor()

    color = chck_color(color_key)
    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {prompt}: {Style.RESET_ALL}"
        path = input(colored_prompt).strip()
        if os.path.exists(path):
            return path
        text = f"[X] Invalid path: {path}. Please try again."
        display.print(text, colors['error'])


def answer_yes_or_no(message, color_key='input'):
    """
    This function retum.
      True for yes, y, oui, o.
      False for no, non n.
     The input does not take uppercase letters into account."
    Displays the prompt in the specified color.
    If the specified color key is invalid, the prompt will be displayed in Light Green.

    """
    display = dc.DisplayColor()

    color = chck_color(color_key)
    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {message} (y/n) ? : {Style.RESET_ALL}"

        reponse = input(colored_prompt).strip()
        if reponse in ['yes', 'y']:
            return True
        if reponse in ['no',  'n']:
            return False
        text = f"[X] Please provide a valid answer (y/n) {ct.BELL}"
        display.print(text, colors['error'])

def input_percentage(message, color_key='input'):
    """
    This function returns a real number between 0 and 1
    that corresponds to a percentage.
    """
    display = dc.DisplayColor()

    color = chck_color(color_key)

    while True:
        # Convert the input color from DISPLAY_COLORS to ANSI
        input_color = rgb_to_ansi(color)
        # Displays the prompt in color
        colored_prompt = f"{input_color}[?] {message} : {Style.RESET_ALL}"

        try:
            # enter = input(colored_prompt).strip()
            value = float(input(colored_prompt).strip())
            if 1 <= value <= 100:
                return value / 100

            text = f"[X] Error: The value must be between 1 and 100. {ct.BELL}"
            display.print(text, colors['error'])
        except ValueError:
            text = f"[X] Error: Please enter a valid number. {ct.BELL}"
            display.print(text, colors['error'])




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
    """
    Prints a text string inside a decorative box.

    The box is created with a width that is determined by the length of the input text,
    with additional padding on both sides. The box is framed with special characters
    to enhance its appearance.

    Parameters:
    text (str): The text to be displayed inside the box.

    """
    # Determine the width of the box based on the string length
    box_width = len(text) + 2  # Add padding for the box

    # Create the box
    print(f" ╔{'═' * (box_width)}╗")
    print(f" ║{text.center(box_width)}║")
    print(f" ╚{'═' * (box_width)}╝")
