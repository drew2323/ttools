
from ctypes import Union
from ttools.config import *
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
import pandas as pd
from ttools.config import TRADE_CACHE, AGG_CACHE, EXCLUDE_CONDITIONS, ACCOUNT1_LIVE_API_KEY, ACCOUNT1_LIVE_SECRET_KEY
from alpaca.data.requests import StockTradesRequest
import time as time_module
from traceback import format_exc
from datetime import timedelta, datetime, time
from time import time as timetime
from concurrent.futures import ThreadPoolExecutor
from alpaca.data.enums import DataFeed
import random
from ttools.utils import AggType, fetch_calendar_data, print, print_matching_files_info, set_verbose, list_matching_files
from tqdm import tqdm
import threading
from typing import List, Union
from ttools.aggregator_vectorized import aggregate_trades, aggregate_trades_optimized
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import math
import os
"""
Module for fetching stock data. Supports
1) cache management
    - Trade Cache - daily files for all trades of that day
    - Agg Cache - cache of whole requested period identified by aggtype, resolution etc.

2) custom vectorized aggregation of trades
    - time based OHLCV
    - volume OHLCV
    - dollar OHLCV
    - renko OHCLV

"""

trade_cache_lock = threading.Lock()

# Function to ensure fractional seconds are present
def ensure_fractional_seconds(timestamp):
    if '.' not in timestamp:
        # Inserting .000000 before the timezone indicator 'Z'
        return timestamp.replace('Z', '.000000Z')
    else:
        return timestamp

def convert_dict_to_multiindex_df(tradesResponse, rename_labels = True, keep_symbols=True):
    """"
    Converts dictionary from cache or from remote (raw input) to multiindex dataframe.
    with microsecond precision (from nanoseconds in the raw data)

    keep_symbols - if true, then output is multiindex indexed by symbol. Otherwise, symbol is removed and output is simple df
    """""
    # Create a DataFrame for each key and add the key as part of the MultiIndex
    dfs = []
    for key, values in tradesResponse.items():
        df = pd.DataFrame(values)
        # Rename columns
        # Select and order columns explicitly
        #print(df)
        df = df[['t', 'x', 'p', 's', 'i', 'c','z']]
        if rename_labels:
            df.rename(columns={'t': 'timestamp', 'c': 'conditions', 'p': 'price', 's': 'size', 'x': 'exchange', 'z':'tape', 'i':'id'}, inplace=True)
            timestamp_col = 'timestamp'
        else:
            timestamp_col = 't'

        df['symbol'] = key  # Add ticker as a column

        # Apply the function to ensure all timestamps have fractional seconds
        #zvazit zda toto ponechat a nebo dat jen pri urcitem erroru pri to_datetime
        #pripadne pak pridelat efektivnejsi pristup, aneb nahrazeni NaT - https://chatgpt.com/c/d2be6f87-b38f-4050-a1c6-541d100b1474
        df[timestamp_col] = df[timestamp_col].apply(ensure_fractional_seconds)

        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')  # Convert 't' from string to datetime before setting it as an index

        #Adjust to microsecond precision
        df.loc[df[timestamp_col].notna(), timestamp_col] = df[timestamp_col].dt.floor('us')

        df.set_index(['symbol', timestamp_col], inplace=True)  # Set the multi-level index using both 'ticker' and 't'
        df = df.tz_convert(zoneNY, level=timestamp_col)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame with MultiIndex
    final_df = pd.concat(dfs)

    if keep_symbols is False:
        final_df.reset_index(inplace=True) # Reset index to remove MultiIndex levels, making them columns
        final_df.drop(columns=['symbol'], inplace=True) #remove symbol column
        final_df.set_index(timestamp_col, inplace=True) #reindex by timestamp
        #print index datetime resolution
        #print(final_df.index.dtype)

    return final_df

def filter_trade_df(df: pd.DataFrame, start: datetime = None, end: datetime = None, exclude_conditions = None, minsize = None, main_session_only = True, symbol_included=True):
    """
    Filters trade dataframe based on start and end, main_session and also applies exclude_conditions and minsize filtering if required.

    Parameters:
    df: pd.DataFrame
    start: datetime
    end: datetime
    exclude_conditions: list of string conditions to exclude from the data
    minsize: minimum size of trade to be included in the data
    main_session_only: boolean, if True, only trades between 9:30 and 15:40 are included
    symbol_included: boolean, if True, DataFrame contains symbol (tbd dynamic)

    Returns:
    df: pd.DataFrame
    """
    def fast_filter(df, exclude_conditions):
        # Convert arrays to strings once
        str_series = df['c'].apply(lambda x: ','.join(x))
        
        # Create mask using vectorized string operations
        mask = np.zeros(len(df), dtype=bool)
        for cond in exclude_conditions:
            mask |= str_series.str.contains(cond, regex=False)
        
        # Apply filter
        return df[~mask]

    def vectorized_string_sets(df, exclude_conditions):
        # Convert exclude_conditions to set for O(1) lookup
        exclude_set = set(exclude_conditions)
        
        # Vectorized operation using sets intersection
        arrays = df['c'].values
        mask = np.array([bool(set(arr) & exclude_set) for arr in arrays])
        
        return df[~mask]

    # 9:30 to 16:00
    if main_session_only:

        if symbol_included:
            # Create a mask to filter rows within the specified time range
            mask = (df.index.get_level_values('t') >= time(9, 30)) & \
                (df.index.get_level_values('t') < time(16, 0))
            df = df[mask]
        else:
            df = df.between_time("9:30","16:00")#TODO adapt to market type

    #REQUIRED FILTERING
    # Create a mask to filter rows within the specified time range
    if start is not None and end is not None:
        print(f"Trimming {start} {end}")
        if symbol_included:
            mask = (df.index.get_level_values('t') >= start) & \
                (df.index.get_level_values('t') <= end)
            df = df[mask]
        else:
            df = df.loc[start:end]

    if exclude_conditions is not None:
        print(f"excluding {exclude_conditions}")
        df = vectorized_string_sets(df, exclude_conditions)
        print("exclude done")

    if minsize is not None:
        print(f"minsize {minsize}")
        #exclude conditions
        df = df[df['s'] >= minsize]
        print("minsize done")
    return df

def calculate_optimal_workers(file_count, min_workers=4, max_workers=32):
    """
    Calculate optimal number of workers based on file count and system resources
    
    Rules of thumb:
    - Minimum of 4 workers to ensure parallelization
    - Maximum of 32 workers to avoid thread overhead
    - For 100 files, aim for around 16-24 workers
    - Scale with CPU count but don't exceed max_workers
    """
    cpu_count = os.cpu_count() or 4
    
    # Base calculation: 2-4x CPU count for I/O bound tasks
    suggested_workers = cpu_count * 3
    
    # Scale based on file count (1 worker per 4-6 files is a good ratio)
    files_based_workers = math.ceil(file_count / 5)
    
    # Take the smaller of the two suggestions
    optimal_workers = min(suggested_workers, files_based_workers)
    
    # Clamp between min and max workers
    return max(min_workers, min(optimal_workers, max_workers))

def fetch_daily_stock_trades(symbol, start, end, exclude_conditions=None, minsize=None, main_session_only=True, no_return=False,force_remote=False, rename_labels = False, keep_symbols=False, max_retries=5, backoff_factor=1, data_feed: DataFeed = DataFeed.SIP, verbose = None):
    #doc for this function
    """
    Attempts to fetch stock trades either from cache or remote. When remote, it uses retry mechanism with exponential backoff.
    Also it stores the data to cache if it is not already there.
    by using force_remote - forcess using remote data always and thus refreshing cache for these dates
        Attributes:
        :param symbol: The stock symbol to fetch trades for.
        :param start: The start time for the trade data.
        :param end: The end time for the trade data.
        :exclude_conditions: list of string conditions to exclude from the data
        :minsize minimum size of trade to be included in the data
        :no_return: If True, do not return the DataFrame. Used to prepare cached files.
        :force_remote will always use remote data and refresh cache
        :param max_retries: Maximum number of retries.
        :param backoff_factor: Factor to determine the next sleep time.
        :param rename_labels: Rename t to timestamp, c to condition etc.
        :param keep_symbols: Whether to keep symbols in the DataFrame (as hierarchical index)
        :return: TradesResponse object.
        :raises: ConnectionError if all retries fail.
    
    In parquet tradecache there are daily files including all trades  incl ext hours
    BAC-20240203.parquet
    """
    is_same_day = start.date() == end.date()
    # Determine if the requested times fall within the main session
    #in_main_session = (time(9, 30) <= start.time() < time(16, 0)) and (time(9, 30) <= end.time() <= time(16, 0))
    
    if not is_same_day:
        raise ValueError("fetch_daily_stock_trades is not implemented for multiple days!")
    
    if verbose is not None:
        set_verbose(verbose)  # Change global verbose if specified    

    #exists in cache?
    daily_file = f"{symbol}-{str(start.date())}.parquet"
    file_path = TRADE_CACHE / daily_file
    if file_path.exists() and (not force_remote or not no_return):
        with trade_cache_lock:
            df = pd.read_parquet(file_path)
        print("Loaded from CACHE", file_path)
        df = filter_trade_df(df, start, end, exclude_conditions, minsize, symbol_included=False, main_session_only=main_session_only)
        return df

    day_next = start.date() + timedelta(days=1)

    print("Fetching from remote.")
    client = StockHistoricalDataClient(ACCOUNT1_LIVE_API_KEY, ACCOUNT1_LIVE_SECRET_KEY, raw_data=True)
    stockTradeRequest = StockTradesRequest(symbol_or_symbols=symbol, start=start.date(), end=day_next, feed=data_feed)
    last_exception = None

    for attempt in range(max_retries):
        try:
            tradesResponse = client.get_stock_trades(stockTradeRequest)
            print(f"Remote fetched completed.", start.date(), day_next)
            if not tradesResponse[symbol]:
                print(f"EMPTY")
                return pd.DataFrame()
            
            df = convert_dict_to_multiindex_df(tradesResponse, rename_labels=rename_labels, keep_symbols=keep_symbols)

            #if today is market still open, dont cache - also dont cache for IEX feeed
            if datetime.now().astimezone(zoneNY).date() < day_next or data_feed == DataFeed.IEX:
                print("not saving trade cache, market still open today or IEX datapoint")
                #ic(datetime.now().astimezone(zoneNY))
                #ic(day.open, day.close)
            else:
                with trade_cache_lock:
                    df.to_parquet(file_path, engine='pyarrow')
                print("Saved to CACHE", file_path)
                if no_return:
                    return pd.DataFrame()

            df = filter_trade_df(df, start, end, exclude_conditions, minsize, symbol_included=False, main_session_only=main_session_only)
            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            last_exception = e
            time_module.sleep(backoff_factor * (2 ** attempt) + random.uniform(0, 1))  # Adding random jitter

    print("All attempts to fetch data failed.")
    raise ConnectionError(f"Failed to fetch stock trades after {max_retries} retries. Last exception: {str(last_exception)} and {format_exc()}")

def fetch_trades_parallel(symbol, start_date, end_date, exclude_conditions = EXCLUDE_CONDITIONS, minsize = 100, main_session_only = True, force_remote = False, max_workers=None, no_return = False, verbose = None):
    """
    Fetch trades between ranges.

    Fetches trades for each day between start_date and end_date during market hours (9:30-16:00) in parallel and concatenates them into a single DataFrame.

    If fetched remotely, the data is stored in tradecache.

    Also if required filters the condition, minsize, main_session if required for results.

    Also can be used just to prepare cached trade files. (noreturn = True)


    :param symbol: Stock symbol.
    :param start_date: Start date as datetime.
    :param end_date: End date as datetime.
    :param exclude_conditions: List of conditions to exclude from the data. None means default.
    :param minsize: Minimum size of trade to be included in the data.
    :param main_session_only: Only include trades during market hours.
    :param no_return: If True, do not return the DataFrame. Used to prepare cached files.
    :return: DataFrame containing all trades from start_date to end_date.
    """
    if verbose is not None:
        set_verbose(verbose)  # Change global verbose if specified

    futures = []
    results = []
    
    market_open_days = fetch_calendar_data(start_date, end_date)
    day_count = len(market_open_days)
    print(f"{symbol} Contains", day_count, " market days")
    max_workers = min(10, max(2, day_count // 2)) if max_workers is None else max_workers  # Heuristic: half the days to process, but at least 1 and no more than 10

    #which days to fetch?
    days_from_remote = []
    days_from_cache = []
    if not force_remote:
        for market_day in market_open_days:
            daily_file_new = str(symbol) + '-' + str(market_day.date) + '.parquet'
            file_path_new = TRADE_CACHE / daily_file_new
            if file_path_new.exists():
                days_from_cache.append((market_day,file_path_new))
            else:
                days_from_remote.append(market_day)         
    else:
        days_from_remote = market_open_days

    remote_df = pd.DataFrame()
    local_df = pd.DataFrame()

    if len(days_from_cache) > 0 and not no_return:
        #speed it up , locals first and then fetches
        s_time = timetime()
        with trade_cache_lock:
            file_paths = [f for _, f in days_from_cache]
            dataset = ds.dataset(file_paths, format='parquet')
            local_df = dataset.to_table().to_pandas()
            del dataset
            #original version          
            #local_df = pd.concat([pd.read_parquet(f) for _,f in days_from_cache])
        final_time = timetime() - s_time
        print(f"{symbol} All {len(days_from_cache)} split files loaded in", final_time, "seconds")
        #the filter is required
        local_df = filter_trade_df(local_df, start_date, end_date, exclude_conditions, minsize, symbol_included=False, main_session_only=main_session_only)
        print(f"{symbol} filtered")

    #do this only for remotes
    if len(days_from_remote) > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            #for single_date in (start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)):
            for market_day in tqdm(days_from_remote, desc=f"{symbol} Remote fetching"):
                #start = datetime.combine(single_date, time(9, 30))  # Market opens at 9:30 AM
                #end = datetime.combine(single_date, time(16, 0))   # Market closes at 4:00 PM
                
                #day interval (min day time to max day time)
                day_date = zoneNY.localize(market_day.open).date()
                min_day_time = zoneNY.localize(datetime.combine(day_date, datetime.min.time()))
                max_day_time = zoneNY.localize(datetime.combine(day_date, datetime.max.time()))
                #print(min_day_time, max_day_time) #zacatek dne

                #TADY JSEM SKONCIL
                #ZKUSME TO NEJDRIV NECHAT puvodne pres market open days
                # a jen vymysleme jak drivejsi start a konec
                # a testnout zda parquety pojedou rychle
                # pripadne pak doresit

                #pripadne orizneme pokud je pozadovane pozdejsi zacatek a drivejsi konek
                start = max(start_date, min_day_time)
                end = min(end_date, max_day_time)

                future = executor.submit(fetch_daily_stock_trades, symbol, start, end, exclude_conditions, minsize, main_session_only, no_return, force_remote)
                futures.append(future)
            
            for future in tqdm(futures, desc=f"{symbol} Receiving trades"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error fetching data for a day: {e}")

        if not no_return:
            # Batch concatenation to improve speed
            batch_size = 10
            batches = [results[i:i + batch_size] for i in range(0, len(results), batch_size)]
            remote_df = pd.concat([pd.concat(batch, ignore_index=False) for batch in batches], ignore_index=False)

    #merge local and remote
    if not remote_df.empty and not no_return:
        return pd.concat([local_df, remote_df], ignore_index=False)
    return local_df

def load_data(symbol: Union[str, List[str]],
    agg_type: AggType,
    resolution: str,
    start_date: datetime,
    end_date: datetime,
    exclude_conditions: list = EXCLUDE_CONDITIONS,
    minsize = None,
    main_session_only = True,
    force_remote=False,
    return_vbt = False,
    verbose = None):
    """Main function to fetch data.

    Returns requested aggregated data for give symbol(s)
    - if requested agg data already exists in aggcache, returns it
    - otherwise get the trades (from trade cache or Alpaca) and aggregate them and store to cache
    
    For location of both caches, see config.py

    Args:
        symbol (Union[str, list]): Symbol
        agg_type (AggType): Type of aggregation
        resolution (Union[str, int]) Resolution of aggregation nased on agg_type 
        start_date (datetime): Start period, timezone aware
        end_date (datetime): Start period, timezone aware
        exclude_conditions (list, optional): Trade conditions to exclude. Defaults to None.
        minsize (_type_, optional): Minimum trade size to include. Defaults to None.
        main_session_only (bool, optional): Main or ext. hours.. Defaults to True.
        force_remote (bool, optional): If True, forces cache reload (doesnt use cache both for trade and agg data)
        return_vbt (bool): If True returns vbt Data object with symbols as columns, otherwise returns symbol keyed dict with pd.DataFrame
        verbose (bool): If True, prints debug messages. None means keep global settings.

    Returns
    -------
    dict keyed by SYMBOL and pd.DataFrame as value
    OR
    vbt.Data object (keyed by SYMBOL)
    """
    symbols = [symbol] if isinstance(symbol, str) else symbol

    if verbose is not None:
        set_verbose(verbose)  # Change global verbose if specified    

    if start_date.tzinfo is None:
        start_date = zoneNY.localize(start_date)

    if end_date.tzinfo is None:
        end_date = zoneNY.localize(end_date)

    if exclude_conditions is None:
        exclude_conditions = EXCLUDE_CONDITIONS

    def load_data_single(symbol, agg_type, resolution, start_date, end_date, exclude_conditions, minsize, main_session_only, force_remote):
        exclude_conditions.sort()
        excludes_str = ''.join(map(str, exclude_conditions))  
        file_ohlcv = AGG_CACHE / f"{symbol}-{str(agg_type)}-{str(resolution)}-{start_date.strftime('%Y-%m-%dT%H-%M-%S')}-{end_date.strftime('%Y-%m-%dT%H-%M-%S')}-{str(excludes_str)}-{minsize}-{main_session_only}.parquet"

        #if matching files with same condition and same or wider date span
        matched_files = list_matching_files(
            symbol=symbol,
            agg_type=str(agg_type),
            resolution=str(resolution),
            start_date=start_date,
            end_date=end_date,
            excludes_str=str(excludes_str),
            minsize=minsize,
            main_session_only=main_session_only
        )
        print("matched agg files", len(matched_files))
        print_matching_files_info(matched_files)

        if not force_remote and len(matched_files) > 0:
            ohlcv_df = pd.read_parquet(matched_files[0],
                                       engine='pyarrow',
                                       filters=[('time', '>=', start_date), ('time', '<=', end_date)])
            print("Loaded from agg_cache", matched_files[0])
            return ohlcv_df
        else:
            #neslo by zrychlit, kdyz se zobrazuje pomalu Searching cache - nejaky bottle neck?
            df = fetch_trades_parallel(symbol, start_date, end_date, minsize=minsize, exclude_conditions=exclude_conditions, main_session_only=main_session_only, force_remote=force_remote) #exclude_conditions=['C','O','4','B','7','V','P','W','U','Z','F'])
            ohlcv_df = aggregate_trades_optimized(symbol=symbol, trades_df=df, resolution=resolution, type=agg_type, clear_input = True)

            ohlcv_df.to_parquet(file_ohlcv, engine='pyarrow')
            print(f"{symbol} Saved to agg_cache", file_ohlcv)    
            return ohlcv_df

    ret_dict_df = {}
    for symbol in symbols:
        ret_dict_df[symbol] = load_data_single(symbol, agg_type, resolution, start_date, end_date, exclude_conditions, minsize, main_session_only, force_remote)

    if return_vbt:
        try:
            import vectorbtpro as vbt  # Import only when needed
        except ImportError:
            raise RuntimeError("vectorbtpro is required for return_vbt. Please install it.")
                
        return vbt.Data.from_data(vbt.symbol_dict(ret_dict_df), tz_convert=zoneNY)
    
    return ret_dict_df

def prepare_trade_cache(symbol: Union[str, List[str]],
    start_date: datetime,
    end_date: datetime,
    force_remote=False,
    verbose = None):
    """
    Fetches trade cache daily files for specified symbols and date range and stores
    them to trade cache location.

    Only the dates not in cache are fetched, unles force_remote is set to True.

    Note: daily trade cache files contain all trades (main+ext hours) for that day

    Args:
        symbol Union[str, list]: Symbol or list of symbols
        start_date (datetime): 
        end_date (datetime): 
        force_remote (bool, optional): Force remote fetch. Defaults to False.

    Returns:
        None
    """
    symbols = [symbol] if isinstance(symbol, str) else symbol
    
    for symbol in symbols:
        #just cache update
        print(f"Started for {symbol}")
        df = fetch_trades_parallel(symbol, start_date, end_date, force_remote=force_remote, no_return=True, verbose=verbose)
        print(f"Finished for {symbol}")