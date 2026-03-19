import sys
import os
import json
import time
import pickle
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None
from typing import *

from .utils import TabPFNClassifier
from .styles import setup_figure, create_savefig_partial

def get_tabpfn_embeddings(X_train, y_train, X_test, y_test):
    """
    Extract TabPFN embeddings for test samples
    using K-fold embedding extraction.
    """

    clf = TabPFNClassifier(
        n_estimators=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    clf.fit(X_train, y_train)

    train_embeddings = clf.get_embeddings(X_test, data_source='train')[0]
    test_embeddings = clf.get_embeddings(X_test, data_source='test')[0]
    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    
    return embeddings
