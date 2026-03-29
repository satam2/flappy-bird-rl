import numpy as np


def summarize_scores(scores, early_deaths):
    return {
        "mean_pipes": float(np.mean(scores)),
        "median_pipes": float(np.median(scores)),
        "std_pipes": float(np.std(scores)),
        "early_deaths": early_deaths,
    }
