from sklearn.cluster import MeanShift
import numpy as np
from easydl.evaluator.clustering import evaluate_clustering_with_labels
from easydl import TimerContext, prepare_path
import pandas as pd
from os.path import join


def tune_mean_shift(x, y,  bandwidth_range=None, save_path='meanshift-tune.csv', disable_tqdm=False):
    """
    x, y are testing set
    """
    from tqdm import tqdm

    rows = []
    if bandwidth_range is None:
        bandwidth_range = np.arange(0.05, 1, 0.05)

    for bandwidth in tqdm(bandwidth_range, disable=disable_tqdm):
        cls = MeanShift(bandwidth=bandwidth)
        ypred = cls.fit_predict(x)
        metrics1 = evaluate_clustering_with_labels(y, ypred)

        row = {'bandwidth': bandwidth}
        row.update(metrics1)

        rows.append(row)

    rows = pd.DataFrame(rows)
    prepare_path(save_path)
    rows.to_csv(save_path, index=False)

