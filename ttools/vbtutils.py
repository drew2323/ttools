import pandas as pd
import vectorbtpro as vbt
import pandas_market_calendars as mcal
from typing import Any

def create_mask_from_window(entries: Any, entry_window_opens:int, entry_window_closes:int):
    """
    Accepts entries and window range (number of minutes from market start) and returns boolean mask denoting 
     entries within the window.

    Parameters
    ----------
    entries : pd.Series/pd:DataFrame
        Entries to be masked.
    entry_window_opens : int
        Number of minutes from market start to open the window.
    entry_window_closes : int
        Number of minutes from market start to close the window.

    Returns
    -------
    type of entries
    """
    # Get the NYSE calendar
    nyse = mcal.get_calendar("NYSE")
    # Get the market hours data
    market_hours = nyse.schedule(start_date=entries.index[0].to_pydatetime(), end_date=entries.index[-1].to_pydatetime(), tz=nyse.tz)

    market_hours =market_hours.tz_localize(nyse.tz)

    # Use merge_asof to align entries with the nearest market_open in market_hours
    merged = pd.merge_asof(
        entries, 
        market_hours[['market_open', 'market_close']], 
        left_index=True, 
        right_index=True, 
        direction='backward'
    )

    # Calculate the time difference between each entry and its corresponding market_open
    elapsed_time = entries.index.to_series() - merged['market_open']

    # Convert the difference to minutes
    elapsed_minutes = elapsed_time.dt.total_seconds() / 60.0

    #elapsed_minutes = pd.DataFrame(elapsed_minutes, index=entries.index)

    # Create a boolean mask for entries that are within the window
    window_opened = (elapsed_minutes >= entry_window_opens) & (elapsed_minutes < entry_window_closes)

    return window_opened


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