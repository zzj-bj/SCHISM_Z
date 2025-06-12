# -*- coding: utf-8 -*-
""" SCHISM """

import sys

from tools import selection as sl
from tools import display_color as dc
from tools import various_functions as vf
from tools import constants as ct


from preprocessing import launch_preprocessing as lp
from training import launch_training as lt
from inference import launch_inference as li

#=======================================================================
def exit_prog():
    """ End of Program """
    print("\n[!! End of program !!]")
    select = vf.answer_yes_or_non("Are you sure")
    if select :
        print(f"[<3] Goodbye! {ct.BELL}")
        dc.display_color(ct.LOGO_OUT, "gray")
        sys.exit()

def menu_preprocessing():
    """
    This module allows for the following operations:
        - Json generation   *** In development ***
        - Normalization of masks in 8-bit grayscale format.
    """
    preprocessing_menu = sl.Menu('Preprocessing', style = 'Unicode')
    while True:
        preprocessing_menu.display_menu()
        choice = preprocessing_menu.selection()

        #TODO In development
        # **** Json generation ****
        if choice == 1:
            lp.launch_json_generation()

        # **** Normalization ****
        elif choice == 2:
            lp.launch_normalisation()

        # **** Return main menu ****
        elif choice == 3:
            return

def main():
    """Displays the CLI menu and handles user choices."""
    dc.display_color(ct.LOGO_IN, "Yellow")
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
