# -*- coding: utf-8 -*-
"""
This class allows for tracking errors during the execution of processes

@author: Pierre.FANCELLI
"""

from tools import display_color as dc
from tools.constants import DISPLAY_COLORS as colors

#============================================================================

class ErrorReport:
    """
    This class allows for tracking errors during the execution of processes.
    """
    def __repr__(self):
        """
        Returns a string representation of the ErrorReport class.

        Returns:
            str: A string indicating the class name.
        """
        return 'ErrorReport'

    def __init__(self):
        """
        Initialize an empty dictionary.
        """
        self.dictionary = {}
        self.report = None
        self.display = dc.DisplayColor()

    def add(self, key, element, is_critical=True):
        """
        Add an element to the list associated with the specified key.
        If the key does not exist, it is created with a new list.

        :param key: The key to which the element should be added
        :param element: The element to add to the list (should be a string)
        :param is_critical: A boolean indicating if the error is critical (default is True)
        """
        if key not in self.dictionary:
            self.dictionary[key] = []
        # Store the element as a tuple (element, is_critical)
        self.dictionary[key].append((element, is_critical))

    def is_report(self):
        """ Check if there is a report. """
        return bool(self.dictionary)

    def display_report(self):
        """
        Display the contents of the report
        """
        for key, elements in self.dictionary.items():
            # Séparer les éléments en fonction de la valeur booléenne
            critical_elements = [elem for elem in elements if elem[1]]
            non_critical_elements = [elem for elem in elements if not elem[1]]

            # Afficher d'abord les éléments critiques
            if critical_elements:
                self.display.print(f"{key} (Critical Errors): \n   {' / '.join(elem[0] for elem in critical_elements)}", colors['error'])

            # Afficher ensuite les éléments non critiques
            if non_critical_elements:
                self.display.print(f"{key} (Warning): \n   {' / '.join(elem[0] for elem in non_critical_elements)}", colors['warning'])

        print("")

    def status(self, process):
        """
        Displaying the correct end process message
        """
        if self.is_report():
            self.display.print(f"[X] {process} failed", colors['error'])
            self.display_report()
        else:
            self.display.print(f"[√] {process} completed \n", colors['ok'])