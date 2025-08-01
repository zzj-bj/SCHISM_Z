# -*- coding: utf-8 -*-
""" SCHISM """

import sys

import torch


from tools import menu
from tools import display_color as dc
from tools import utils as vf
from tools import constants as ct
from tools.constants import DISPLAY_COLORS as colors

from preprocessing import launch_preprocessing as lp
from training import launch_training as lt
from inference import launch_inference as li


#=======================================================================

def main():
    """Displays the CLI menu and handles user choices."""
    display = dc.DisplayColor()
    print(ct.LOGO_IN)

    if torch.cuda.is_available():
        display.print("CUDA is available! Running on GPU.\n", colors['ok'])
    else:
        display.print("CUDA is not available! Running on CPU.\n", colors['warning'])

    main_menu = menu.Menu('MAIN')
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
    text = "Do you really want to leave the program? We'll miss you!"
    select = vf.answer_yes_or_no(text)
    if select :
        print(f"[<3] Goodbye! {ct.BELL}")
        print(ct.LOGO_OUT)
        sys.exit()


if __name__ == "__main__":
    main()
