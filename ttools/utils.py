from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Tuple
import re
import pytz
import calendar
from ttools.config import AGG_CACHE
import os
from alpaca.trading.models import Order, TradeUpdate, Calendar
import pandas_market_calendars as mcal
#Zones
zoneNY = pytz.timezone('US/Eastern')
zoneUTC = pytz.utc
zonePRG = pytz.timezone('Europe/Amsterdam')
verbose = True #default

# Save the built-in print function to a different name
built_in_print = print

# Custom print function that respects the global verbose setting
def print(*args, **kwargs):
    if verbose:
        built_in_print(*args, **kwargs)

# Function to set the global verbose variable
def set_verbose(value):
    global verbose
    verbose = value

def parse_filename(filename: str) -> dict:
    """Parse filename of AGG_CACHE files into its components using regex.
    https://claude.ai/chat/b869644b-f542-4812-ad58-d4439c15fa78
    """
    pattern = r"""
        ^
        ([A-Z]+)-                     # Symbol
        ([^-]+)-                      # Agg type
        (\d+)-                        # Resolution
        (\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})- # Start date
        (\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})- # End date
        ([A-Z0-9]+)-                  # Excludes string
        (\d+)-                        # Minsize
        (True|False)                  # Main session flag
        \.parquet$                    # File extension
    """
    match = re.match(pattern, filename, re.VERBOSE)
    if not match:
        return None
    
    try:
        symbol, agg_type, resolution, start_str, end_str, excludes, minsize, main_session = match.groups()
        
        return {
            'symbol': symbol,
            'agg_type': agg_type,
            'resolution': resolution,
            'start_date': datetime.strptime(start_str, '%Y-%m-%dT%H-%M-%S'),
            'end_date': datetime.strptime(end_str, '%Y-%m-%dT%H-%M-%S'),
            'excludes_str': excludes,
            'minsize': int(minsize),
            'main_session_only': main_session == 'True'
        }
    except (ValueError, AttributeError):
        return None

def list_matching_files(
    symbol: str = None,
    agg_type: str = None,
    resolution: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    excludes_str: str = None,
    minsize: int = None,
    main_session_only: bool = None
) -> list[Path]:
    """
    List all aggregated files in the cache directory matching the specified criteria.
    If start_date and end_date are provided, returns files that cover this interval
    (meaning their date range encompasses the requested interval).
    If a parameter is None, it matches any value for that component.

    Example:
    ```python
    # Example with all parameters specified
    specific_files = list_matching_files(
        symbol="SPY",
        agg_type="AggType.OHLCV",
        resolution="12",
        start_date=datetime(2024, 1, 15, 9, 30),
        end_date=datetime(2024, 1, 15, 16, 0),
        excludes_str="4679BCFMOPUVWZ",
        minsize=100,
        main_session_only=True
    )

    print_matching_files_info(specific_files)
    ```
    """
    #make date naive
    if start_date is not None:
        start_date = start_date.replace(tzinfo=None)
    if end_date is not None:
        end_date = end_date.replace(tzinfo=None)

    agg_cache_dir = AGG_CACHE
    def matches_criteria(file_info: dict) -> bool:
        """Check if file matches all specified criteria."""
        if not file_info:
            return False
            
        # Check non-date criteria first
        if symbol is not None and file_info['symbol'] != symbol:
            return False
        if agg_type is not None and file_info['agg_type'] != agg_type:
            return False
        if resolution is not None and file_info['resolution'] != resolution:
            return False
        if excludes_str is not None and file_info['excludes_str'] != excludes_str:
            return False
        if minsize is not None and file_info['minsize'] != minsize:
            return False
        if main_session_only is not None and file_info['main_session_only'] != main_session_only:
            return False
            
        # Check date range coverage if both dates are provided
        if start_date is not None and end_date is not None:
            return (file_info['start_date'] <= start_date and 
                   file_info['end_date'] >= end_date)
        
        # If only start_date is provided
        if start_date is not None:
            return file_info['end_date'] >= start_date
            
        # If only end_date is provided
        if end_date is not None:
            return file_info['start_date'] <= end_date
            
        return True

    # Process all files
    matching_files = []
    for file_path in agg_cache_dir.iterdir():
        if not file_path.is_file() or not file_path.name.endswith('.parquet'):
            continue
            
        file_info = parse_filename(file_path.name)
        if matches_criteria(file_info):
            matching_files.append((file_path, file_info))
    
    # Sort files by start date and then end date
    matching_files.sort(key=lambda x: (x[1]['start_date'], x[1]['end_date']))
    
    # Return just the file paths
    return [f[0] for f in matching_files]

def print_matching_files_info(files: list[Path]):
    """Helper function to print detailed information about matching files."""
    for file_path in files:
        file_info = parse_filename(file_path.name)
        if file_info:
            print(f"\nFile: {file_path.name}")
            print(f"Coverage: {file_info['start_date']} to {file_info['end_date']}")
            print(f"Symbol: {file_info['symbol']}")
            print(f"Agg Type: {file_info['agg_type']}")
            print(f"Resolution: {file_info['resolution']}")
            print(f"Excludes: {file_info['excludes_str']}")
            print(f"Minsize: {file_info['minsize']}")
            print(f"Main Session Only: {file_info['main_session_only']}")
            print("-" * 80)

def fetch_calendar_data(start: datetime, end: datetime) -> List[Calendar]:
    """
    Fetches the trading schedule for the NYSE (New York Stock Exchange) between the specified start and end dates.
    Args:
        start (datetime): The start date for the trading schedule.
        end (datetime): The end date for the trading schedule.
    Returns:
        List[Calendar]: A list of Calendar objects containing the trading dates and market open/close times. 
                        Returns an empty list if no trading days are found within the specified range.
    """ 
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start, end_date=end, tz='America/New_York')
    if not schedule.empty: 
        schedule = (schedule.reset_index()
                        .rename(columns={"index": "date", "market_open": "open", "market_close": "close"})
                        .assign(date=lambda day: day['date'].dt.date.astype(str),
                                open=lambda day: day['open'].dt.strftime('%H:%M'), 
                                close=lambda day: day['close'].dt.strftime('%H:%M'))
                        .to_dict(orient="records"))
        cal_dates = [Calendar(**record) for record in schedule]
        return cal_dates
    else:
        return []

def split_range(start: datetime, stop: datetime, period: str = "Y") -> List[Tuple[datetime, datetime]]:
    """
    Splits a range of dates into a list of (start, end) tuples, where end is exclusive (start of next range)
    
    Args:
        start (datetime): start date
        stop (datetime): end date
        period (str): 'Y', 'M', 'W', or 'D' for year, month, week, or day

    
    Returns:
        List[Tuple[datetime, datetime]]: list of (start, end) tuples
    
    Example:

    ```python
    year_ranges = split_range(day_start, day_stop, period="M")
    for start_date,end_date in year_ranges:
        print(start_date,end_date)

    >>> 2023-01-15 09:30:00-05:00 2023-02-01 00:00:00-05:00
    >>> 2023-02-01 00:00:00-05:00 2023-03-01 00:00:00-05:00
    >>> 2023-03-01 00:00:00-05:00 2023-04-01 00:00:00-05:00
    >>> 2023-04-01 00:00:00-05:00 2023-05-01 00:00:00-05:00
    >>> 2023-05-01 00:00:00-05:00 2023-06-01 00:00:00-05:00
    ```
    
    """
    if start > stop:
        raise ValueError("Start date must be before stop date")
    
    ranges = []
    current_start = start

    while current_start < stop:
        if period == "Y":
            next_period_start = datetime(current_start.year + 1, 1, 1, tzinfo=current_start.tzinfo)
        elif period == "M":
            next_year = current_start.year + (current_start.month // 12)
            next_month = (current_start.month % 12) + 1
            next_period_start = datetime(next_year, next_month, 1, tzinfo=current_start.tzinfo)
        elif period == "W":
            next_period_start = current_start + timedelta(weeks=1)
        elif period == "D":
            next_period_start = current_start + timedelta(days=1)
        else:
            raise ValueError("Invalid period specified. Choose from 'Y', 'M', 'W', or 'D'.")

        # Set the end of the current period or stop if within the same period
        current_end = min(next_period_start, stop)
        
        # Append the (start, end) tuple to ranges
        ranges.append((zoneNY.localize(current_start), zoneNY.localize(current_end)))
        
        # Move to the start of the next period
        current_start = next_period_start
    
    return ranges

#create enum AGG_TYPE
class AggType(str, Enum):
    """
    Enum class for aggregation types.
    ohlcv - time based ohlcv (time as resolution)
    ohlcv_vol - volume based ohlcv (volume as resolution)
    ohlcv_dol - dollar volume based ohlcv (dollar amount as resolution)
    ohlcv_renko - renko based ohlcv (brick size as resolution)
    """
    OHLCV = 'ohlcv'
    OHLCV_VOL = 'ohlcv_vol'
    OHLCV_DOL = 'ohlcv_dol'
    OHLCV_RENKO = 'ohlcv_renko'

class StartBarAlign(str, Enum):
    """
    Represents first bar start time alignement according to timeframe
        ROUND = bar starts at 0,5,10 (for 5s timeframe)
        RANDOM = first bar starts when first trade occurs
    """ 
    ROUND = "round"
    RANDOM = "random"