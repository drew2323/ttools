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

mom_standard = vbt.indicator("talib:MOM").run(t1data.data["BAC"].close)
mom_anchored_d = AnchoredIndicator("talib:MOM", anchor='D').run(t1data.data["BAC"].close)
```

