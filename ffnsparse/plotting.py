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
    if save != '':
        plt.savefig(save, dpi=300)
    plt.close()