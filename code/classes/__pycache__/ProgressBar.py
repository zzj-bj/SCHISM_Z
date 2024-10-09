# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:15:09 2022

@author: florent.brondolo
"""
import sys
class ProgressBar:
   """
   A class to display a progress bar in the console.
   """
   DEFAULT_BAR_LENGTH = 35
   #DEFAULT_CHAR_ON  = '■█'□■
   DEFAULT_CHAR_ON = '■'
   DEFAULT_CHAR_OFF = '□'
   def __init__(self, end, start=0, txt=""):
       """
       Initialize the progress bar.
       Parameters:
       end (int): The final value of the progress bar.
       start (int): The starting value of the progress bar.
       txt (str): Optional text to display with the progress bar.
       """
       self.end = end
       self.start = start
       self.txt = txt
       self._bar_length = self.__class__.DEFAULT_BAR_LENGTH
       self.set_level(self.start)
       self._plotted = False
   def set_level(self, level):
       """
       Set the current level of the progress bar.
       Parameters:
       level (int): The current progress value to be set.
       """
       self._level = max(self.start, min(level, self.end))
       self._ratio = float(self._level - self.start) / float(self.end - self.start)
       self._level_chars = int(self._ratio * self._bar_length)
   def plot_progress(self):
       """
       Plot the progress bar in the console.
       """
       sys.stdout.write("\r %3i%% %s%s %s" % (
           int(self._ratio * 100.0),
           self.__class__.DEFAULT_CHAR_ON * int(self._level_chars),
           self.__class__.DEFAULT_CHAR_OFF * int(self._bar_length - self._level_chars),
           '- [' + self.txt + ']',
       ))
       sys.stdout.flush()
       self._plotted = True
   def set_and_plot(self, level):
       """
       Set the level and plot the progress bar if the display changes.
       Parameters:
       level (int): The current progress value to be set.
       """
       old_chars = self._level_chars
       self.set_level(level)
       if not self._plotted or old_chars != self._level_chars:
           self.plot_progress()
   def __add__(self, other):
       """
       Add a value to the progress bar.
       Parameters:
       other (int or float): The value to add to the current level.
       """
       assert isinstance(other, (float, int)), "Can only add a number"
       self.set_and_plot(self._level + other)
       return self
   def __sub__(self, other):
       """
       Subtract a value from the progress bar.
       Parameters:
       other (int or float): The value to subtract from the current level.
       """
       return self.__add__(-other)
   def __iadd__(self, other):
       """
       In-place addition for the progress bar.
       Parameters:
       other (int or float): The value to add to the current level.
       """
       return self.__add__(other)
   def __isub__(self, other):
       """
       In-place subtraction for the progress bar.
       Parameters:
       other (int or float): The value to subtract from the current level.
       """
       return self.__add__(-other)
   def __del__(self):
       """
       Ensures a newline is printed when the progress bar is deleted.
       """
       sys.stdout.write("\n")