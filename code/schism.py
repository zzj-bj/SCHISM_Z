# -*- coding: utf-8 -*-
""" SCHISM """

import sys

import torch

from tools import menu as sl
from tools import display_color as dc
from tools import various_functions as vf
from tools import constants as ct


from preprocessing import launch_preprocessing as lp
from training import launch_training as lt
from inference import launch_inference as li


#=======================================================================




def main():
    """Displays the CLI menu and handles user choices."""
    display = dc.DisplayColor()
    display.print(ct.LOGO_IN, "Yellow")

    if torch.cuda.is_available():
        display.print("CUDA is available! Running on GPU.\n", "green")
    else:
        display.print("CUDA is NOT available! Running on CPU.\n", "red")

    main_menu = sl.Menu('MAIN')
    while True:
        main_menu.display_menu()
        choice = main_menu.selection()

        # Menu Preprocessing
        if choice == 1:
            lp.LaunchPreprocessing().menu_preprocessing()

        # Training
        elif choice == 2:
            lt.LaunchTraining().train_model()

         # Inference
        elif choice == 3:
            li.LaunchInference().run_inference()

        # Fin de Programme
        elif choice == 4:
            exit_prog()

def exit_prog():
    """ End of Program """
    print("\n[!! End of program !!]")
    select = vf.answer_yes_or_no("Are you sure")
    if select :
        print(f"[<3] Goodbye! {ct.BELL}")

        dc.DisplayColor().print(ct.LOGO_OUT, "gray", bold=True)
        sys.exit()


if __name__ == "__main__":
    main()
