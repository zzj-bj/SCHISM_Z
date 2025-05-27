# -*- coding: utf-8 -*-
""" SCHISM """

import sys

from tools import logo as lo
from tools import selection as sl

from preprocessing import preprocessing as pr
from training import launch_training as lt
from inference import launch_inference as li

#=======================================================================
def exit_prog():
    """ End of Program """
    print("\n[!! End of program !!]")
    select = sl.answer_yes_or_non("Are you sure")
    if select :
        print(f"[<3] Goodbye! {sl.BELL}")
        sl.display_color(lo.LOGO_OUT, "gray")
        sys.exit()

def menu_preprocessing():
    """
    "This module allows for the following operations:
        - Json generation   *** In development ***
        - Normalization of masks in 8-bit grayscale format."
    """
    preprocessing_menu = sl.Menu('Preprocessing')
    while True:
        preprocessing_menu.display_menu()
        choice = preprocessing_menu.selection()

        #TODO In development
        # **** Json generation ****
        if choice == 1:
            pr.launch_json_generation()

        # **** Normalization ****
        elif choice == 2:
            pr.launch_normalisation()

        # **** Return main menu ****
        elif choice == 3:
            return

def main():
    """Displays the CLI menu and handles user choices."""
    sl.display_color(lo.LOGO_IN, "Yellow")
    main_menu = sl.Menu('MAIN')
    while True:
        main_menu.display_menu()
        choice = main_menu.selection()

        # Menu Preprocessing
        if choice == 1:
            menu_preprocessing()

        # Training
        elif choice == 2:
            lt.train_model()

         # Inference
        elif choice == 3:
            li.run_inference()

        # Fin de Programme
        elif choice == 4:
            exit_prog()

if __name__ == "__main__":
    main()
