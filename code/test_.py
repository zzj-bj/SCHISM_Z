# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 08:36:57 2025

@author: Pierre.FANCELLI
"""
import os
import sys
import keyboard



def attente_(touche):   # Attente l'appui d'une touche
    print(f"Appuyez sur la touche '{touche}' pour continuer...")
    keyboard.wait(touche)


#------------------------------

data_dir = "C:/Users/Pierre.FANCELLI/Documents/Projet_SCHISM/_Source_Data"
subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]






