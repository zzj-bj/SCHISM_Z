# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:14:12 2025

@author: Pierre.FANCELLI
"""

BELL = "\a"

LOGO = """
╔══════════════════════════════════════════════════╗
║  ███████╗ ██████╗██╗  ██╗██╗███████╗███╗   ███╗  ║
║  ██╔════╝██╔════╝██║  ██║██║██╔════╝████╗ ████║  ║
║  ███████╗██║     ███████║██║███████╗██╔████╔██║  ║
║  ╚════██║██║     ██╔══██║██║╚════██║██║╚██╔╝██║  ║
║  ███████║╚██████╗██║  ██║██║███████║██║ ╚═╝ ██║  ║
║  ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝  ║
╠══════════════════════════════════════════════════╣
║        -Let's do some cool segmentation-         ║
╚══════════════════════════════════════════════════╝
"""

MAIN_MENU = ["MAIN MENU",
              "Preprocessing",
             "Training",
             "Inference",
             "Quit"
             ]

MENU_Preprocessing = ["PREPROCESSING",
                       "Json generation",
                       "Normalisation",
                       "Return Main Menu"
                       ]

SELECTION = [ "Training",       # 0
             "Inference",       # 1
             "Json generation", # 2
             "normalisation"    # 3
            ]


class AffichageMenu :
    """
    Cette classe permet l'affichage des menus & des sélections
    """

    def __init__(self, select):
        self.select = select

    def print_menu(self):
        """ Affichage d'un menu """
        if self.select == 1:
            menu = MAIN_MENU
        if self.select == 2:
            menu = MENU_Preprocessing

        longueur_max = max(len(chaine) for chaine in menu)
        ligne = len(menu)

        box_width = longueur_max + 6

        # Créer la boîte
        print(f"╔{'═' * (box_width )}╗")
        print(f"║{menu[0].center(box_width )}║")
        print(f"╠{'═' * (box_width )}╣")
        for i in range(1, ligne):
            print(f"║ {i} : {menu[i].ljust(longueur_max)} ║")
        print(f"╚{'═' * (box_width )}╝")

    # def print_selection(self):
    #     """ Affichage de la sélection """
    #     longueur_max = max(len(chaine) for chaine in SELECTION)
    #     box_width = longueur_max + 6

    #     # Créer la boîte
    #     print(f"╔{'═' * (box_width )}╗")
    #     print(f"║{SELECTION[self.select].center(box_width )}║")
    #     print(f"╚{'═' * (box_width )}╝")
