from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Tuple
import pytz
import calendar

#Zones
zoneNY = pytz.timezone('US/Eastern')
zoneUTC = pytz.utc
zonePRG = pytz.timezone('Europe/Amsterdam')

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



def find_dotenv(start_path):
    """
    Searches for a .env file in the given directory or its parents and returns the path.

    Args:
        start_path: The directory to start searching from.

    Returns:
        Path to the .env file if found, otherwise None.
    """
    current_path = Path(start_path)
    for _ in range(6):  # Limit search depth to 5 levels
        dotenv_path = current_path / '.env'
        if dotenv_path.exists():
            return dotenv_path
        current_path = current_path.parent
    return None

# def get_daily_tradecache_file():
#     return Path(DATA_DIR) / "tradecache"

# def get_daily_aggcache_file():
#     #nazev obsahuje i child class
#     #a take excludes result = ''.join(self.excludes.sort())
#     self.excludes.sort()  # Sorts the list in place
#     excludes_str = ''.join(map(str, self.excludes))  # Joins the sorted elements after converting them to strings
#     cache_file = self.__class__.__name__ + '-' + self.symbol + '-' + str(int(date_from.timestamp())) + '-' + str(int(date_to.timestamp())) + '-' + str(self.rectype) + "-" + str(self.resolution) + "-" + str(self.minsize) + "-" + str(self.align) + '-' + str(self.mintick) + str(self.exthours) + excludes_str + '.cache.gz'
#     file_path = DATA_DIR + "/aggcache/" + cache_file
#     #print(file_path)
#     return file_path


#create enum AGG_TYPE
class AGG_TYPE(str, Enum):
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

class RecordType(str, Enum):
    """
    Represents output of aggregator
    """

    BAR = "bar"
    CBAR = "cbar"
    CBARVOLUME = "cbarvolume"
    CBARDOLLAR = "cbardollar"
    CBARRENKO = "cbarrenko"
    TRADE = "trade"


class StartBarAlign(str, Enum):
    """
    Represents first bar start time alignement according to timeframe
        ROUND = bar starts at 0,5,10 (for 5s timeframe)
        RANDOM = first bar starts when first trade occurs
    """ 
    ROUND = "round"
    RANDOM = "random"