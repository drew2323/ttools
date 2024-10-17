import pandas as pd
import vectorbtpro as vbt
import pandas_market_calendars as mcal
from typing import Any
import datetime

def isrising(series: pd.Series, n: int) -> pd.Series:
    """
    Checks if a series is rising over a given window size.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    n : int
        Window size
        
    Returns
    -------
    pd.Series
        Boolean mask indicating when the series is falling
    """
    return series.rolling(n).apply(lambda x: (x == sorted(x, reverse=True)).all(), raw=False).fillna(False).astype(bool)

def isfalling(series: pd.Series, n: int) -> pd.Series:
    """
    Checks if a series is falling over a given window size.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    n : int
        Window size
        
    Returns
    -------
    pd.Series
        Boolean mask indicating when the series is falling
    """
    return series.rolling(n).apply(lambda x: (x == sorted(x, reverse=True)).all(), raw=False).fillna(False).astype(bool)

def create_mask_from_window(series: Any, entry_window_opens:int, entry_window_closes:int, use_cal: bool = True):
    """
    Accepts series and window range (number of minutes from market start) and returns boolean mask denoting 
     series within the window.

    Parameters
    ----------
    series : pd.Series/pd:DataFrame
        series to be masked.
    entry_window_opens : int
        Number of minutes from market start to open the window.
    entry_window_closes : int
        Number of minutes from market start to close the window.
    use_cal : bool, default True
        If True, uses NYSE calendar to determine market hours for each day. Otherwise uses 9:30 to 16:00 constant.

    Returns
    -------
    type of series
    """

    if use_cal:
        # Get the NYSE calendar
        nyse = mcal.get_calendar("NYSE")
        # Get the market hours data
        market_hours = nyse.schedule(start_date=series.index[0].to_pydatetime(), end_date=series.index[-1].to_pydatetime(), tz=nyse.tz)

        market_hours =market_hours.tz_localize(nyse.tz)

        # Ensure both series and market_hours are timezone-aware and in the same timezone
        if series.index.tz is None:
            series.index = series.index.tz_localize('America/New_York')

        # Use merge_asof to align series with the nearest market_open in market_hours
        merged = pd.merge_asof(
            series.to_frame(), 
            market_hours[['market_open', 'market_close']], 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )

        # Calculate the time difference between each entry and its corresponding market_open
        elapsed_time = series.index.to_series() - merged['market_open']

        # Convert the difference to minutes
        elapsed_minutes = elapsed_time.dt.total_seconds() / 60.0

        #elapsed_minutes = pd.DataFrame(elapsed_minutes, index=series.index)

        # Create a boolean mask for series that are within the window
        window_opened = (elapsed_minutes >= entry_window_opens) & (elapsed_minutes < entry_window_closes)

        # Return the mask as a series with the same index as series
        return pd.Series(window_opened.values, index=series.index)
    else:
        # Calculate the time difference in minutes from market open for each timestamp
        market_open = datetime.time(9, 30)
        market_close = datetime.time(16, 0)
        window_open= pd.Series(False, index=series.index)
        elapsed_min_from_open = (series.index.hour - market_open.hour) * 60 + (series.index.minute - market_open.minute)
        window_open[(elapsed_min_from_open >= entry_window_opens) & (elapsed_min_from_open < entry_window_closes)] = True
        return window_open

class AnchoredIndicator:
    """
    Allows to run any VBT indicator in anchored mode (reset per Day, Hour, or Minute).
    """
    def __init__(self, indicator_name: str, anchor='D'):
        """
        Initialize with the name of the indicator (e.g., "talib:MOM").
        
        Parameters:
        - indicator_name: str, the name of the vectorbt indicator.
        - anchor: str, 'D' for day, 'h' for hour, 'min' for minute (default is 'D').
           Any valid frequency string ('D', 'h', 'min', 'W', etc.). can be used as it uses pd.Grouper(freq=anchor)
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """
        self.indicator_name = indicator_name
        self.indicator = vbt.indicator(indicator_name)
        self.anchor = anchor

    def run(self, data, anchor=None, *args, **kwargs):
        """
        Run the indicator on a Series or DataFrame by splitting it by day, hour, or minute,
        applying the indicator to each group, and concatenating the results.
        
        Parameters:
        - data: pd.Series or pd.DataFrame, the input data series or dataframe (e.g., close prices).
        - anchor: str, 'D' for day, 'h' for hour, 'min' for minute (default is 'D'). Override for anchor on the instance.
            Any valid frequency string ('D', 'h', 'min', 'W', etc.). can be used as it uses pd.Grouper(freq=anchor)
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        - *args, **kwargs: Arguments and keyword arguments passed to the indicator.
        """

        if anchor is None:
            anchor = self.anchor

        # Use pd.Grouper for splitting by any valid frequency string
        try:
            grouped_data = data.groupby(pd.Grouper(freq=anchor))
        except ValueError as e:
            raise ValueError(f"Invalid anchor value: {anchor}. Check pandas Grouper frequencies.") from e

        # Run the indicator function for each group and concatenate the results
        results = []
        for date, group in grouped_data:
            if group.empty:
                continue
            # Run the indicator for each group's data
            result = self.indicator.run(group, *args, **kwargs)
            results.append(result)

        # Concatenate the results and return
        return vbt.base.merging.row_stack_merge(results)