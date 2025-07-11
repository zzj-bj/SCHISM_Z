# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:54:43 2025

@author: Pierre.FANCELLI
"""
import os
import re

from tools import folder as fo
from tools import report as rp
from tools import various_functions as vf

from training.training import Training
from commun.hyperparameters import Hyperparameters

#===========================================================================
def compare_number(repertoire1, repertoire2):
    """
   Compares the numbers extracted from the filenames in two specified directories.

   Parameters:
   repertoire1 (str): The path to the first directory.
   repertoire2 (str): The path to the second directory.

   Returns:
   tuple: A tuple containing:
       - folder_ok (bool): True if the sets of numbers are identical, False otherwise.
       - dif_1 (list): A sorted list of numbers found in the first directory but not in the second.
       - dif_2 (list): A sorted list of numbers found in the second directory but not in the first.

   The function works as follows:
   1. It retrieves the list of files from both directories.
   2. It initializes two sets to store the extracted numbers from each directory.
   3. It uses a regular expression to find all numeric sequences in the filenames.
   4. For each filename, it adds the last found number to the corresponding set.
   5. After processing both directories, it compares the two sets of numbers.
   6. If the sets are not equal, it identifies the differences and returns them.
   """

    fichiers1 = os.listdir(repertoire1)
    fichiers2 = os.listdir(repertoire2)

    numeros1 = set()
    numeros2 = set()

    pattern = re.compile(r'(\d+)')


    for f in fichiers1:
        matches = pattern.findall(f)
        if matches:
            numeros1.add(matches[-1])

    for f in fichiers2:
        matches = pattern.findall(f)
        if matches:
            numeros2.add(matches[-1])

    dif_1 = []
    dif_2 = []

    folder_ok = True
    if numeros1 != numeros2:
        folder_ok = False
        dif_1 = sorted(numeros1 - numeros2)
        dif_2 = sorted(numeros2 - numeros1)

    return folder_ok , dif_1, dif_2

def check_folder(folder, root, report):
    """
    This function checks that all the directories passed as parameters comply
    with the expected requirements.

    Parameters
    ----------
    folder : liste
        List of directories to analyze.
    root : str
        Reference path for analysis.
    report : str
        File for report.

    Returns
    -------
    valid_subfolders : liste
        List of valid directories.
    num_file : int
        Total number of files considered.

    """
    valid_subfolders =[]
    num_file = 0
    for f in folder:
        images_path = fo.create_name_path(root, f, 'images')
        masks_path = fo.create_name_path(root, f, 'masks')

        if not os.path.isdir(images_path):
            report.add(" - Folder 'images' missing in :", f)
        else :
            if not os.path.isdir(masks_path):
                report.add(" - Folder 'masks' missing in :", f)
            else:
                nb_f_image = fo.count_tif_files(images_path)
                nb_f_masks = fo.count_tif_files(masks_path)
                if nb_f_image == 0:
                    report.add(" - No file (*.tif') in folder 'image' :", f)
                else:
                    if nb_f_masks == 0:
                        report.add(" - No file (*.tif') in folder 'masks' :", f)
                    else:
                        if nb_f_image != nb_f_masks :
                            text = " - number of images not equal between 'images/masks' :"
                            report.add(text, f)
                        else:
                            ok ,ima_1, mask_1 = compare_number(images_path, masks_path)
                            if not ok:
                                report.add(" - Single images whitout masks :",
                                              f"in {f} : {ima_1} ")
                                report.add(" - Single masks without images :",
                                              f"in {f} : {mask_1} ")
                            else:
                                num_file += nb_f_image
                                valid_subfolders.append(f)
    return  valid_subfolders, num_file

def train_model():
    """Executes the training process in CLI."""

    print("\n[ Training Mode ]")

    initial_condition = True
    report_training = rp.ErrorReport()

    data_dir = fo.get_path("Enter the data directory")
    run_dir = fo.get_path("Enter the directory to save runs")
    path_hyperparameters = fo.get_path("Enter the path to the hyperparameters INI file")
    hyperparameters_path = os.path.join(path_hyperparameters, "hyperparameters.ini")
    if not os.path.exists(hyperparameters_path):
        initial_condition = False
        report_training.add(' - hyperparameters', 'The hyperparameters file was not found')
    else:
        try:
            hyperparameters = Hyperparameters(hyperparameters_path)
        except FileNotFoundError:
            initial_condition = False
            text = f" The file '{hyperparameters_path}' was not found."
            report_training.add(' - hyperparameters', text)
        except ValueError as ve:
            initial_condition = False
            text = f"ValueError: {ve}"
            report_training.add(' - hyperparameters', text)
        except IOError as ioe:
            initial_condition = False
            text = f"IOError: An error occurred while trying to read the file: {ioe}"
            report_training.add(' - hyperparameters', text)
        except Exception as e:
            initial_condition = False
            text = f"{e}"
            report_training.add(' - hyperparameters', text)

    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    valid_subfolders = []
    num_file = 0
    if len(subfolders) == 0 :
        report_training.add(" - The data directory is Empty ", '')
    else:
        valid_subfolders, num_file = check_folder(subfolders, data_dir, report_training)


    vf.warning_message(subfolders, valid_subfolders)

    if initial_condition and report_training.is_report() == 0 :
    # if initial_condition and len(valid_subfolders) != 0 :
        print("[!] Starting training.")

        try:
            train_object = Training(
                data_dir = data_dir,
                subfolders = valid_subfolders,
                run_dir = run_dir,
                hyperparameters = hyperparameters,
                report = report_training,
                num_file = num_file
                )

            train_object.load_segmentation_data()
            train_object.train()


        except KeyError as e:
            report_training.add(" - Training",
                                "Caught KeyError in DataLoader worker process :\n"
                                f"{e}")
        except ValueError as e:
            report_training.add(" - hyperparameters :", f"{e}" )
        except Exception as e:
            report_training.add(" - Other defects ", f"{e}")

    file_name_1 = fo.create_name_path(path_hyperparameters, '', 'Training')
    report_training.status("Training", file_name_1)
