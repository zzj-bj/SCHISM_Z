# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:49:41 2025

@author: Pierre.FANCELLI
"""

class Rapport:
    def __init__(self):
        """
        Initialise un dictionnaire vide.
        """
        self.dictionnaire = {}
        self.report = None

    def ajouter_element(self, cle, element):
        """
        Ajoute un élément à la liste associée à la clé spécifiée.
        Si la clé n'existe pas, elle est créée avec une nouvelle liste.

        :param cle: La clé à laquelle l'élément doit être ajouté
        :param element: L'élément à ajouter à la liste
        """
        if cle not in self.dictionnaire:
            self.dictionnaire[cle] = []
        self.dictionnaire[cle].append(element)

    def exsit_reppot(self):
        self.report = False
        if self.dictionnaire:
            self.report =  True
        return self.report

    def afficher_rapport(self):
        """
        Affiche le contenu du dictionnaire.
        """
        total_def = sum(len(liste)for liste in self.dictionnaire.values())

        print(f"*** {total_def} Folders with problems :***")
        for cle, elements in self.dictionnaire.items():
            print(f"{cle} {', '.join(elements)}")
        print("")
