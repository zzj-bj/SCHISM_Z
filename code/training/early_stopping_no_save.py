from early_stopping_pytorch import EarlyStopping
"""Z: used by training.py"""
class EarlyStoppingNoSave(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        # On override la fonction pour ne rien faire
        pass