import os
import scipy.io as sio
import numpy as np
import sklearn.metrics
from skimage.transform import resize
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

def plot_roc(y_trues, y_scores, loss, num_classes):
      
    states, colors = get_states_and_colors(num_classes)
    if num_classes == 2:
        y_pred = [1 * (y>=0.5) for y in y_scores]
        print(classification_report(y_trues, y_pred, target_names=states))
        print(f"Cohen Kappa Score: {cohen_kappa_score(y_trues, y_pred)}")
    
    else: 
        if 'focal' in loss: 
            print(classification_report(y_trues, np.argmax(y_scores, axis=1), target_names=states))
            print(f"Cohen Kappa Score: {cohen_kappa_score(y_trues, np.argmax(y_scores, axis=1))}")
        else:
            print(classification_report(np.argmax(y_trues,axis=1),  np.argmax(y_scores, axis=1), target_names=states))
            print(f"Cohen Kappa Score: {cohen_kappa_score(np.argmax(y_trues, axis=1), np.argmax(y_scores, axis=1))}")
    
def get_states_and_colors(num_classes):
    if num_classes == 2:
        states = ['Awake', 'NREM']
        colors = cycle(['c', 'forestgreen'])
    elif num_classes == 3:
        states = ['Awake', 'NREM', 'REM']
        colors = cycle(['c', 'forestgreen', 'slateblue'])
    elif num_classes == 4:
        states = ["Awake", 'NREM', 'K/X', 'Movement']
        colors = cycle(['c', 'forestgreen', 'slateblue', 'darkorange'])

    return states, colors
