# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:49:41 2025

@author: Pierre.FANCELLI
"""

from tools import display_color as dc

#============================================================================
# Format for Date & Time
F_DATE = "%Y-%m-%d <> %H:%M:%S"

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

    def add(self, key, element):
        """
        Add an element to the list associated with the specified key.
        If the key does not exist, it is created with a new list.

        :param key: The key to which the element should be added
        :param element: The element to add to the list

        """
        if key not in self.dictionary:
            self.dictionary[key] = []
        self.dictionary[key].append(element)

    def is_report(self):
        """ Check if there is a report. """
        return bool(self.dictionary)

    def display_report(self):
        """
        Display the contents of the report
        """
        total_def = sum(len(liste)for liste in self.dictionary.values())

        print(f"*** !!! {total_def} problem(s) found(s) !!! :***")
        for key, elements in self.dictionary.items():
            print(f"{key} {', '.join(elements)}")
        print("")

    def status(self, process):
        """
        Displaying the correct end process message
        """
        if self.is_report():
            dc.display_color(f"[X] {process} completed with error(s)", "red")
            self.display_report()
        else:
            dc.display_color(f"[√] {process} completed without error\n", "green")
