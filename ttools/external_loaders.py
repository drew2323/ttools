from ctypes import Union
from ttools import zoneUTC
from ttools.config import *
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from ttools.config import ACCOUNT1_LIVE_API_KEY, ACCOUNT1_LIVE_SECRET_KEY
from datetime import timedelta, datetime, time
from alpaca.data.enums import DataFeed
from typing import List, Union
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed 
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def load_history_bars(symbol: Union[str, List[str]], datetime_object_from: datetime, datetime_object_to: datetime, timeframe: TimeFrame, main_session_only: bool = True):
    """Returns dataframe fetched remotely from Alpaca.

    Args:
        symbol: symbol or list of symbols
        datetime_object_from: datetime in zoneNY
        datetime_object_to: datetime in zoneNY
        timeframe: timeframe
        main_session_only: boolean to fetch only main session data

    Returns:
        dataframe

    Example:
    ```python
    from ttools.external_loaders import load_history_bars
    from ttools.config import zoneNY
    from datetime import datetime
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    symbol = "AAPL"
    start_date = zoneNY.localize(datetime(2023, 2, 27, 18, 51, 38))
    end_date = zoneNY.localize(datetime(2023, 4, 27, 21, 51, 39))
    timeframe = TimeFrame(amount=1,unit=TimeFrameUnit.Minute)

    df = load_history_bars(symbol, start_date, end_date, timeframe)
    ```
    """
    client = StockHistoricalDataClient(ACCOUNT1_LIVE_API_KEY, ACCOUNT1_LIVE_SECRET_KEY, raw_data=False)
    #datetime_object_from = datetime(2023, 2, 27, 18, 51, 38, tzinfo=datetime.timezone.utc)
    #datetime_object_to = datetime(2023, 2, 27, 21, 51, 39, tzinfo=datetime.timezone.utc)
    bar_request = StockBarsRequest(symbol_or_symbols=symbol,timeframe=timeframe, start=datetime_object_from, end=datetime_object_to, feed=DataFeed.SIP)
    #print("before df")
    df = client.get_stock_bars(bar_request).df
    df.index = df.index.set_levels(df.index.get_level_values(1).tz_convert(zoneNY), level=1)
    if main_session_only:
        start_time = time(9, 30, 0)
        end_time = time(15, 30, 0)
        df = df.loc[(df.index.get_level_values(1).time >= start_time) & (df.index.get_level_values(1).time <= end_time)]
    return df
