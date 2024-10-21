# ttools
A Python library for tools, utilities, and helpers for my trading research workflow.

## Installation

```python
pip install git+https://github.com/drew2323/ttools.git
```
Modules:
# vbtutils

Contains helpers for vbtpro

`AnchoredIndicator` - allows runing any vbt indicator in anchored mode (reset by Day, Hour etc.)

Example usage:
```python
from ttools import AnchoredIndicator

mom = vbt.indicator("talib:MOM").run(t1data.data["BAC"].close, timeperiod=10, skipna=True) #standard indicator
mom_anch_d = AnchoredIndicator("talib:MOM", anchor='D').run(t1data.data["BAC"].close, timeperiod=10, skipna=True) #anchored to D
```

`create_mask_from_window` - creates mask of the same size AS INPUT, True values denotes that the window is open. Used to filter entry window or forced eod window. Range is denoted by pair (start, end) indicating minutes elapsed from the market start of that day.

```python
from ttools import create_mask_from_window

entry_window_opens = 3 #in minutes from start of the market
entry_window_closes = 388
forced_exit_start = 387
forced_exit_end = 390

#create mask based on main session that day
entry_window_opened = create_mask_from_window(entries, entry_window_opens, entry_window_closes)
#limit entries to the window
entries = entries & entry_window_opened

#create forced exits mask
forced_exits_window = create_mask_from_window(exits, forced_exit_start, forced_exit_end)

#add forced_exits to exits
exits = exits | forced_exits_window

exits.tail(20)
```
## is rising/is falling
`isrising(series,n)`,`isfalling(series, n)` - returns mask where the condition is satisfied of rising or falling elements including equal values

`isrisingc(series,n)`,`isfallingc(series, n)`  - same as above but scritly rising/fallinf (no equal values)
# Indicators

Custom indicators in the `indicators` folder.

## Importing
```python
from ttools.vbtindicators import register_custom_inds
register_custom_inds(None, "override") #All indicators from the folder are automatically imported and registered.
register_custom_inds("CUVWAP", "override")#just one
```

After registration they can be listed and used
```python
vbt.IF.list_indicators("ttools")
vbt.phelp(vbt.indicator("ttools:CUVWAP").run)

vwap_cum_d = vbt.indicator("ttools:CUVWAP").run(s12_data.high, s12_data.low, s12_data.close, s12_data.volume, anchor=vbt.Default(value="D"), drag=vbt.Default(value=50), hide_default=True)
```
## Creating

To create custom indicators CUSTOMNAME.py in `indicators` folder is created containing varibles IND_CUSTOMNAME containing the Indicator Factory class.

## Available

- `CUVWAP` - Cumulative VWAP with anchor based on HLCC4 with optional rounding (hlcc4_round, def.3) and drag - warming period from previous anchor unit(def.0).
- `DIVERGENCE` - Various divergences between two timeseries (abs, relative, relative normalized, pct, abs pct)
