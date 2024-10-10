import pandas as pd
import vectorbtpro as vbt

class AnchoredIndicator:
    """
    Allows to run any VBT indicator in anchored mode (reset per Day, Hour, or Minute).
    """
    def __init__(self, indicator_name: str, split_by='D'):
        """
        Initialize with the name of the indicator (e.g., "talib:MOM").
        
        Parameters:
        - indicator_name: str, the name of the vectorbt indicator.
        - split_by: str, 'D' for day, 'H' for hour, 'T' for minute (default is 'D').
        """
        self.indicator_name = indicator_name
        self.indicator = vbt.indicator(indicator_name)
        self.split_by = split_by

    def run(self, data, split_by=None, *args, **kwargs):
        """
        Run the indicator on a Series or DataFrame by splitting it by day, hour, or minute,
        applying the indicator to each group, and concatenating the results.
        
        Parameters:
        - data: pd.Series or pd.DataFrame, the input data series or dataframe (e.g., close prices).
        - split_by: str, 'D' for day, 'H' for hour, 'T' for minute (default is 'D').
        - *args, **kwargs: Arguments and keyword arguments passed to the indicator.
        """

        if split_by is None:
            split_by = self.split_by

        # Group by the specified frequency
        if split_by == 'D':
            grouped_data = data.groupby(data.index.date)
        elif split_by in ['H', 'T']:
            grouped_data = data.groupby(pd.Grouper(freq=split_by))
        else:
            raise ValueError("Invalid split_by value. Use 'D' (day), 'H' (hour), or 'T' (minute).")

        # Run the indicator function for each group and concatenate the results
        results = []
        for date, group in grouped_data:
            # Run the indicator for each group's data
            result = self.indicator.run(group, *args, **kwargs)
            results.append(result)

        # Concatenate the results and return
        return vbt.base.merging.row_stack_merge(results)