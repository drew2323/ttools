# ttools
A Python library for tools, utilities, and helpers for my trading research workflow.

Modules:
# utils
## vbtutils

Contains helpers for vbtpro

`AnchoredIndicator` - allows runing any vbt indicator in anchored mode (reset by Day, Hour etc.)

Example usage:
```python
from ttools import AnchoredIndicator

mom = vbt.indicator("talib:MOM").run(t1data.data["BAC"].close, timeperiod=10, skipna=True) #standard indicator
mom_anch_d = AnchoredIndicator("talib:MOM", anchor='D').run(t1data.data["BAC"].close, timeperiod=10, skipna=True) #anchored to D
```

