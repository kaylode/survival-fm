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
from .embedding import TabPFNEmbedding
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

    embedding_extractor = TabPFNEmbedding(
        tabpfn_clf=clf,
        n_fold=5,            # important for stability
    )

    embeddings = embedding_extractor.get_embeddings(
        X_train,
        y_train,
        X_test,
        data_source="test"
    )

    return embeddings
