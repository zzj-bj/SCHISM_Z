# -*- coding: utf-8 -*-
""" SCHISM """

# Third-party
import torch


# Local application imports
from tools import menu, utils as vf
from tools import display_color as dc
from tools import constants as ct
from tools.constants import DISPLAY_COLORS as colors
from preprocessing import launch_preprocessing as lp
from training import launch_training as lt
from inference import launch_inference as li


#=======================================================================

def main() -> None:

    print(f"\n****** {ct.init_random_seed} ********\n")

    # Displays the CLI menu and handles user choices.
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

        if choice == 1:
            lp.LaunchPreprocessing().menu_preprocessing()
        elif choice == 2:
            lt.LaunchTraining().train_model()
        elif choice == 3:
            li.LaunchInference().run_inference()
        elif choice == 4:
            if _confirm_exit():
                break  # exit the loop and end program

def _confirm_exit() -> bool:
    """
    Ask user to confirm exit. Returns True if exiting, False otherwise.
    """
    display = dc.DisplayColor()
    prompt = "Do you really want to leave the program? We'll miss you!"
    if vf.answer_yes_or_no(prompt):
        display.print("Goodbye! \n", colors['babye'])
        # print(ct.LOGO_OUT)
        return True
    return False


if __name__ == "__main__":
    main()
