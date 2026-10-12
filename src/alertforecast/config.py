"""Paths, fixed split dates and constants. Split dates are defined in CLAUDE.md; change both together."""
import os
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOLUNTEER_CSV = os.path.join(ROOT, "data", "raw", "vadimkin_2026-10-12", "volunteer_data_en.csv")
KYIV_REGION = "Kyiv City"  # city only, not "Kyivska oblast"

KYIV_TZ = ZoneInfo("Europe/Kyiv")
HORIZONS_H = (1, 3, 6)
GRID_S = 30 * 60
HOUR_S = 3600
DAY_S = 86400


def kyiv_midnight_utc(year, month, day):
    """Start of a Europe/Kyiv local day as a UTC epoch second."""
    return int(pd.Timestamp(year=year, month=month, day=day, tz=KYIV_TZ).tz_convert("UTC").timestamp())


# Fixed split dates. Test end is exclusive.
VAL_START = kyiv_midnight_utc(2026, 8, 17)
TEST_START = kyiv_midnight_utc(2026, 9, 14)
TEST_END = kyiv_midnight_utc(2026, 10, 12)
# Provisional: the first days of the war are sparse and unusual.
TRAIN_ORIGIN_START = int(pd.Timestamp("2022-03-01", tz="UTC").timestamp())
