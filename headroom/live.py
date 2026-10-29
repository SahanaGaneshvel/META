"""Live feeds for the demo's live mode (real data, fetched now). Endpoints verified in docs/DATA_SPEC.md §5."""
from __future__ import annotations

import io

import pandas as pd
import requests

PALO_ALTO = (37.4419, -122.1430)
TIMEOUT = 20


def weather_forecast(lat: float = PALO_ALTO[0], lon: float = PALO_ALTO[1], days: int = 2) -> pd.DataFrame:
    """Open-Meteo hourly temperature forecast (deg C), local time index."""
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=TIMEOUT, params={
        "latitude": lat, "longitude": lon, "hourly": "temperature_2m", "forecast_days": days,
        "timezone": "America/Los_Angeles"})
    r.raise_for_status()
    hourly = r.json()["hourly"]
    return pd.DataFrame({"temp_c": hourly["temperature_2m"]},
                        index=pd.DatetimeIndex(hourly["time"], name="local_time"))


def caiso_demand_today() -> pd.DataFrame:
    """CAISO system demand today (5-minute, MW): current, day-ahead and hour-ahead forecasts."""
    r = requests.get("https://www.caiso.com/outlook/current/demand.csv", timeout=TIMEOUT)
    r.raise_for_status()
    frame = pd.read_csv(io.StringIO(r.text))
    return frame.rename(columns=str.strip)
