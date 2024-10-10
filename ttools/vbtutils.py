import pandas as pd
import vectorbtpro as vbt

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
            # Run the indicator for each group's data
            result = self.indicator.run(group, *args, **kwargs)
            results.append(result)

        # Concatenate the results and return
        return vbt.base.merging.row_stack_merge(results)