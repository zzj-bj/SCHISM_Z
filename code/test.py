# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 09:06:02 2025

@author: Pierre.FANCELLI
"""

import random
from tools import selection as sl


#====================================================================

main_menu = sl.Menu('MAIN')
main_menu.display_menu()

metric =['A', 'x']
met_menu = sl.Menu('Dynamic', metric)
met_menu.display_menu()

metric =['a', 'b ggg  bb', 8.3, 'd']
met_menu = sl.Menu('Dynamic', metric)
met_menu.display_menu()

metric =['a']
met_menu = sl.Menu('Dynamic', metric)
met_menu.display_menu()



# ma_liste =['a', 'b', 'c', 'v', 'w', 'y', 'x']
# sortie = ma_liste.copy()


# print(ma_liste)
# print(sortie)
# element_choisi = random.choice(ma_liste)
# print(element_choisi)
# sortie.remove(element_choisi)
# print(sortie)

# ajout = ['---','****']

# sorties = sortie + ajout
# print(sorties)

# encore = 5

# gg = sorties + [5]

# print(gg)

