"""Rolling-origin protocol: fit strictly before a window, predict the window, build out-of-time calibration blocks.

The test split is never reachable from here: frames stop at TEST_START and windows come from config.EVAL_WINDOWS.
"""
import numpy as np
import pandas as pd

from alertforecast import config, splits
from alertforecast.data import Intervals, load_alerts
from alertforecast.frame import build_frame
from alertforecast.neighbors import neighbor_features


def load_full_frame(neighbors=False):
    """Features and labels for every origin from the first train origin up to (not including) the test start.

    With neighbors=True the Stage C neighbour-region group is appended (columns start with 'nb_').
    """
    alerts = load_alerts()
    state_iv, label_iv = Intervals.for_region_state(alerts), Intervals.for_region(alerts)
    t = splits.grid(config.TRAIN_ORIGIN_START, config.TEST_START)
    frame = build_frame(t, state_iv, label_iv)
    if neighbors:
        frame = pd.concat([frame, neighbor_features(t, alerts)], axis=1)
    return frame


def rows(full, origins):
    """Rows of the full frame at the given origins (the frame is a regular 30-minute grid)."""
    origins = np.asarray(origins, dtype="int64")
    if len(origins) == 0:
        return full.iloc[0:0]
    pos = (origins - full["t"].iloc[0]) // config.GRID_S
    assert pos.min() >= 0 and pos.max() < len(full) and (full["t"].values[pos] == origins).all()
    return full.iloc[pos]


def fit_and_predict(recipe, full, h, fit_before_s, predict_origins):
    """Fit `recipe` on origins strictly purged before `fit_before_s`; predict the given origins."""
    fit = rows(full, splits.fit_origins_before(fit_before_s, h))
    model = recipe.fit(fit, fit[f"y{h}"].values, fit_before_s)
    target = rows(full, predict_origins)
    return target, model.predict(target)


def run_window(recipe, full, window, h):
    """Predictions of `recipe` on a method-selection window. Returns the window rows and the probabilities."""
    start, _ = config.EVAL_WINDOWS[window]
    return fit_and_predict(recipe, full, h, start, splits.window_origins(window))


def calibration_origins(block, window_start_s, h):
    """Origins of one calibration block, dropping those whose label window (t, t+h] reaches the evaluated window."""
    b_start, b_end = block
    t = splits.grid(b_start, b_end)
    return t[t + h * config.HOUR_S < window_start_s]


def oot_calibration_data(recipe, full, window, h):
    """Pooled out-of-time predictions for the K blocks before a window.

    Each block is predicted by the same recipe fit strictly before that block. Returns a frame with the block index
    (0 = nearest to the window), labels, probabilities and the clear flag for view (b).
    """
    window_start, _ = config.EVAL_WINDOWS[window]
    parts = []
    for k, block in enumerate(splits.calibration_blocks(window)):
        origins = calibration_origins(block, window_start, h)
        target, p = fit_and_predict(recipe, full, h, block[0], origins)
        part = target[["t", "clear", "day"]].copy()
        part["y"] = target[f"y{h}"].values
        part["p"] = p
        part["block"] = k
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
