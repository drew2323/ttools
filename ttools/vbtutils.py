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
        - anchor: str, 'D' for day, 'H' for hour, 'T' for minute (default is 'D').
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
        - anchor: str, 'D' for day, 'H' for hour, 'T' for minute (default is 'D').
        - *args, **kwargs: Arguments and keyword arguments passed to the indicator.
        """

        if anchor is None:
            anchor = self.anchor

        # Group by the specified frequency
        if anchor == 'D':
            grouped_data = data.groupby(data.index.date)
        elif anchor in ['H', 'T']:
            grouped_data = data.groupby(pd.Grouper(freq=anchor))
        else:
            raise ValueError("Invalid anchor value. Use 'D' (day), 'H' (hour), or 'T' (minute).")

        # Run the indicator function for each group and concatenate the results
        results = []
        for date, group in grouped_data:
            # Run the indicator for each group's data
            result = self.indicator.run(group, *args, **kwargs)
            results.append(result)

        # Concatenate the results and return
        return vbt.base.merging.row_stack_merge(results)