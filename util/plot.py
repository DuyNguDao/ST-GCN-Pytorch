import warnings
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_cm(CM, normalize=True, save_dir='', names_x=(), names_y=(), show=True):
    """
    function: plot confusion matrix
    :param CM: array cm
    :param normalize: normaize 0-1
    :param save_dir: path save
    :param names: name class
    :param show: True
    :return:
    """
    if True:
        import seaborn as sn
        array = CM / ((CM.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
        if not normalize:
            array = np.asarray(array, dtype='int')
        fmt = 'd'
        if normalize:
            fmt = '.2f'
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=2.0 if 2 < 50 else 0.8)  # for label size
        labels_x = (0 < len(names_x) < 99) and len(names_x) == 7  # apply names to ticklabels
        labels_y = (0 < len(names_y) < 99) and len(names_y) == 7  # apply names to ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array, annot=2 < 30, annot_kws={"size": 16}, cmap='Blues', fmt=fmt, square=True,
                       xticklabels=names_x if labels_x else "auto",
                       yticklabels=names_y if labels_y else "auto").set_facecolor((1, 1, 1))
        accu = sum([CM[i, i] for i in range(CM.shape[0])])/sum(sum(CM))
        fig.axes[0].set_xlabel('True', fontweight='bold', fontsize=20)
        fig.axes[0].set_ylabel('Predicted', fontweight='bold', fontsize=20)
        plt.title('Accuracy: {}%'.format(round(accu*100)), fontweight='bold', fontsize=20)
        if show:
            plt.show()
        name_save = 'confusion_matrix.png'
        if normalize:
            name_save = 'confusion_matrix_normalize.png'
        fig.savefig(Path(save_dir) / name_save, dpi=300)
        plt.close()


