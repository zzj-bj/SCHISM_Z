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

    display = dc.DisplayColor()

    # Displays the CLI menu and handles user choices.

    print(ct.LOGO_IN)

    display.print(f"Debug mode {'ON' if ct.DEBUG_MODE else 'OFF'}.", colors['warning'])

    prompt = f"Data augmentation {'ON' if ct.DATA_AUGMENTATION else 'OFF'}."
    display.print(prompt, colors['warning'])

    prompt = (
        f"CUDA{'' if torch.cuda.is_available() else ' not'} available"
        f" - Running on {'GPU' if torch.cuda.is_available() else 'CPU'}.\n"
    )
    display.print(prompt, colors['warning'])


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
    prompt = "Do you want to quit the program"
    if vf.answer_yes_or_no(prompt):
        display.print("Goodbye! \n", colors['babye'])
        # print(ct.LOGO_OUT)
        return True
    return False


if __name__ == "__main__":
    main()
