# -*- coding: utf-8 -*-
""" SCHISM """

import sys

from tools import logo as lo
from tools import selection as sl

from preprocessing import preprocessing  as pr
from training import launch_training as tr
from inference import launch_inference as li


#=======================================================================
def exit_prog():
    print("\n[!! End of programme !!]")
    select = sl.choiice_y_n("Are you sure")
    if select :
        print(f"[<3] Goodbye! o/{sl.BELL}")
        print(lo.LOGO_OUT)
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

        # **** Normalisation ****
        elif choice == 2:
            pr.launch_normalisation()

        # **** Return main menu ****
        elif choice == 3:
            return

def main():
    """Displays the CLI menu and handles user choices."""
    print(lo.LOGO_IN) # "Display the logo SCHISM
    main_menu = sl.Menu('MAIN')
    while True:

        main_menu.display_menu()
        choice = main_menu.selection()

        # Menu Preprocessing
        if choice == 1:
            menu_preprocessing()

        # Training
        elif choice == 2:
            tr.train_model()

         # Inference
        elif choice == 3:
            li.run_inference()

        # Fin de Programme
        elif choice == 4:
            exit_prog()

if __name__ == "__main__":
    main()
