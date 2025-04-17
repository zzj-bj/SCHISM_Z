# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:49:41 2025

@author: Pierre.FANCELLI
"""

class Rapport:
    def __init__(self):
        """
        Initialize an empty dictionary.
        """
        self.dictionary = {}
        self.report = None

    def ajouter_element(self, cle, element):
        """
        Add an element to the list associated with the specified key.
        If the key does not exist, it is created with a new list.

        :param key: The key to which the element should be added
        :param element: The element to add to the list

        """
        if cle not in self.dictionary:
            self.dictionary[cle] = []
        self.dictionary[cle].append(element)

    def exist_reppot(self):
        self.report = False
        if self.dictionary:
            self.report =  True
        return self.report

    def afficher_rapport(self):
        """
        Display the contents of the dictionary..
        """
        total_def = sum(len(liste)for liste in self.dictionary.values())

        print(f"*** !!! {total_def} problems founds !!! :***")
        for cle, elements in self.dictionary.items():
            print(f"{cle} {', '.join(elements)}")
        print("")
