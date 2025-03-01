import matplotlib.pyplot as plt
import numpy as np

import mplhep
plt.style.use(mplhep.style.ATLAS)

figsize = (6, 6)

def plot_loss(train_loss, val_loss, save=''):
    plt.figure(figsize=figsize)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch', loc='right')
    plt.ylabel('Loss', loc='top')
    plt.legend()
    plt.xlim(0, len(train_loss))
    if save != '':
        plt.savefig(save, dpi=300)
    plt.close()
    
def compare_distributions(predictions, targets, save='', xlim=None):
    plt.figure(figsize=figsize)
    plt.hist(predictions, bins=50, alpha=0.5, label='Predictions')
    plt.hist(targets, bins=50, alpha=0.5, label='Targets')
    plt.xlabel('Value', loc='right')
    plt.ylabel('Frequency', loc='top')
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    if save != '':
        plt.savefig(save, dpi=300)
    plt.close()