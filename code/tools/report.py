# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:49:41 2025

@author: Pierre.FANCELLI
"""

from datetime import datetime

from tools import folder as fo
from tools import selection as sl

#============================================================================
# Formar for Date & Time
F_DATE = "%Y-%m-%d <> %H:%M:%S"

class ErrorReport:
    """
    This class allows for tracking errors during the execution of processes.
    It also generates a file in 'txt' format based on the name of the process
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
        Display the contents of the dictionary..
        """
        total_def = sum(len(liste)for liste in self.dictionary.values())

        print(f"*** !!! {total_def} problems founds !!! :***")
        for key, elements in self.dictionary.items():
            print(f"{key} {', '.join(elements)}")
        print("")

    def print_report(self, process, instance_name):
        """ Write report into a file """
        now = datetime.now()
        total_def = sum(len(liste)for liste in self.dictionary.values())

        filename = f"{instance_name}_report.txt"
        with open(filename, 'a') as file:
            file.write("\n*------------------------------------------\n")
            file.write(f"-- {now.strftime(F_DATE  )} -- \n")
            if total_def == 0:
                file.write(f" - {process} finished whitout error\n")
            else:
                file.write(f" *** !!! {total_def} problems founds !!! :***\n")
                for key, elements in self.dictionary.items():
                    file.write(f"{key} : {', '.join(elements)}\n")
            file.write("------------------------------------------*\n")

    def status(self, process, file):
        if self.is_report():
            sl.display_color(f"[X] {process} finished with error", "red")
            self.display_report()
        else:
            sl.display_color(f"[√] {process} finished without error\n", "green")

        self.print_report(process, file)
