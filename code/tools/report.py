# -*- coding: utf-8 -*-
"""
This class allows for tracking errors during the execution of processes

@author: Pierre.FANCELLI
"""

from datetime import datetime
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
        total_def = sum(len(liste) for liste in self.dictionary.values())
        total_topic = len(self.dictionary)

        # Display the message in "MAGENTA" color
        print("\033[35m", end="")
        print(f"*** !!! {total_def} problem(s) found(s) in {total_topic} topic(s) !!! ***")
        for key, elements in self.dictionary.items():
            print(f"\033[31m{key}\033[35m \n   {' / '.join(elements)}")
        # Return to a white display
        print("\033[0m")

    def print_report(self, process, instance_name):
        """ Write report into a file """
        now = datetime.now()
        total_def = sum(len(liste) for liste in self.dictionary.values())
        total_topic = len(self.dictionary)

        filename = f"{instance_name}_report.txt"
        with open(filename, 'a', encoding="utf-8") as file:
            file.write("\n*------------------------------------------\n")
            file.write(f"-- {now.strftime(F_DATE)} -- \n")
            if total_def == 0:
                file.write(f" - {process} finished without error\n")
            else:
                file.write(f"*** !!! {total_def} problem(s) found(s) in {total_topic}"
                           " topic(s) !!! ***\n")
                for key, elements in self.dictionary.items():

                    file.write(f"{key}:\n {'/ '.join(elements)}\n")
            file.write("------------------------------------------*\n")

    def status(self, process, file=None, generate_report=True):
        """
        Displaying the correct end process message
        """
        if self.is_report():
            dc.display_color(f"[X] {process} completed with error(s)", "red")
            self.display_report()
        else:
            dc.display_color(f"[√] {process} completed without error\n", "green")

        # Call print_report only if generate_report is True
        if generate_report:
            self.print_report(process, file)
            