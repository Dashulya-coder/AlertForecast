"""Chronological splits and the purge gap.

Fit rows are purged per horizon with a STRICT boundary: a fit row with origin t is kept only if t + h is before the
first origin of the evaluated split, so no fit label touches the evaluation period, not even its first instant.
Evaluation origins are not purged, but they are limited to windows that end inside the recorded data.
"""
import numpy as np

from alertforecast import config


def grid(start_s, end_s):
    """30-minute UTC grid in [start_s, end_s)."""
    return np.arange(start_s, end_s, config.GRID_S, dtype="int64")


EVAL_RANGES = {
    "val": (config.VAL_START, config.TEST_START),
    "test": (config.TEST_START, config.TEST_END),
}


def eval_origins(name, allow_test=False, horizon_h=None, complete_until=None):
    """Origins of an evaluation split. The test split is locked until the pipeline is frozen.

    With `horizon_h` and `complete_until` (see data.complete_until), origins whose label window (t, t + h] would
    reach beyond the recorded data are dropped, because the labels there are unreliable (missing alerts).
    """
    if name == "test" and not allow_test:
        raise PermissionError("Test origins are locked. Pass allow_test=True only for the single final evaluation.")
    start, end = EVAL_RANGES[name]
    t = grid(start, end)
    if horizon_h is not None and complete_until is not None:
        t = t[t + horizon_h * config.HOUR_S <= complete_until]
    return t


def fit_origins_before(start_s, horizon_h):
    """Purged origins for fitting a model that will be evaluated from `start_s` on. Strict: t + h < start_s."""
    return grid(config.TRAIN_ORIGIN_START, start_s - horizon_h * config.HOUR_S)  # end is exclusive


def fit_origins(eval_name, horizon_h):
    """Purged origins used for fitting when evaluating on `eval_name`.

    val  -> train only; test -> train + val. Strict: t + h < first evaluation origin.
    """
    return fit_origins_before(EVAL_RANGES[eval_name][0], horizon_h)


def window_origins(window_name):
    """All origins of a method-selection window (val, w1, w2). The test split is not available here."""
    start, end = config.EVAL_WINDOWS[window_name]
    return grid(start, end)


def calibration_blocks(window_name):
    """The K 28-day blocks directly before a window, nearest first: [(start, end), ...]."""
    start, _ = config.EVAL_WINDOWS[window_name]
    blocks, end = [], start
    for _ in range(config.CAL_BLOCKS):
        begin = config.shift_local_days(end, -28)
        blocks.append((begin, end))
        end = begin
    return blocks
