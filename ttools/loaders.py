
from ctypes import Union
from dotenv import load_dotenv
from appdirs import user_data_dir
from ttools.utils import find_dotenv
from ttools.config import *
import os
from datetime import datetime

print(DATA_DIR)

def load_data(symbol: Union[str, list], day_start: datetime, day_end: datetime, agg_type: AGG_TYPE, resolution: Union[str, int], excludes: list = EXCLUDE_CONDITIONS, minsize: int = MINSIZE, ext_hours: bool = False, align: StartBarAlign =StartBarAlign.ROUND, as_df: bool = False, force_reload: bool = False) -> None:
    """
    Returns requested aggregated data for give symbol(s)
    - if requested agg data already exists in cache, returns it
    - otherwise get the trades (from trade cache or Alpaca) and aggregate them and store to cache
    
    For location of both caches, see config.py

    Note both trades and agg cache are daily_files

    LOCAL_TRADE_CACHE
    LOCAL_AGG_CACHE

    Parameters
    ----------
    symbol : Union[str, list]
        Symbol or list of symbols
    day_start : datetime
        Start date, zone aware
    day_end : datetime
        End date, zone aware
    agg_type : AGG_TYPE
        Type of aggregation
    resolution : Union[str, int]
        Resolution of aggregation nased on agg_type 
    excludes : list
        List of trade conditions to exclude
    minsize : int
        Minimum size of trade to be included
    ext_hours : bool
        If True, requests extended hours data
    align : StartBarAlign
        How to align first bar RANDOM vs ROUND
    as_df : bool
        If True, returns dataframe, otherwise returns vbt Data object
    force_reload : bool
        If True, forces cache reload (doesnt use cache both for trade and agg data)

    Returns
    -------
    DF or vbt.Data object
    """
    pass


