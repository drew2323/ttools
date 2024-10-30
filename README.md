# ttools
A Python library for tools, utilities, and helpers for my trading research workflow.

## Installation

```python
pip install git+https://github.com/drew2323/ttools.git
```
Modules:
# loaders

- remotely fetches daily trade data
- manages trade cache (daily trade files per symbol) and aggregation cache (per symbola and requested period)
- numba compiled aggregator for required output (time based, dollars, volume bars, renkos...).

Detailed examples in `tests/data_loader_tryme.ipynb`

## load_data
Returns vectorized aggregation of given type. If trades for given period are not cached they are remotely fetched from Alpaca first.

Example:

```python
#This is how to call LOAD function
symbol = ["BAC"]
#datetime in zoneNY 
day_start = datetime(2024, 10, 14, 9, 45, 0)
day_stop = datetime(2024, 10, 16, 15, 1, 0)
day_start = zoneNY.localize(day_start)
day_stop = zoneNY.localize(day_stop)

#requested AGG
resolution = 1
agg_type = AggType.OHLCV #other types AggType.OHLCV_VOL, AggType.OHLCV_DOL, AggType.OHLCV_RENKO
exclude_conditions = ['C','O','4','B','7','V','P','W','U','Z','F','9','M','6'] #None to defaults
minsize = 100
main_session_only = True
force_remote = True

ohlcv_df = load_data(symbol = symbol,
                     agg_type = agg_type,
                     resolution = resolution,
                     start_date = day_start,
                     end_date = day_stop,
                     #exclude_conditions = None,
                     minsize = 100,
                     main_session_only = True,
                     force_remote = False
                     )
bac_df = ohlcv_df["BAC"]

basic_data = vbt.Data.from_data(vbt.symbol_dict(ohlcv_df), tz_convert=zoneNY)
vbt.settings['plotting']['auto_rangebreaks'] = True
basic_data.ohlcv.plot()
```
## prepare trade cache

To prepare daily trade cache files for given period.
If they are not present in cache, they are fetched.
`force_remote` refetches from remote, even when exists in cache.

```python
from ttools.loaders import prepare_trade_cache

symbols = ["BAC", "AAPL"]
#datetime in zoneNY 
day_start = datetime(2024, 10, 1, 9, 45, 0)
day_stop = datetime(2024, 10, 14, 15, 1, 0)
day_start = zoneNY.localize(day_start)
day_stop = zoneNY.localize(day_stop)
force_remote = False

prepare_trade_cache(symbols, day_start, day_stop, force_remote)
```

### Prepare daily trade cache - cli script

Daily trade cache data can be fetched for given period by CLI script, that can run in the background.

Note: To fetch 1 day takes about 35s. It is stored in /tradescache/ directory as daily file keyed by symbol.

To run this script in the background with specific arguments:

```bash
# Running without forcing remote fetch
python3 prepare_cache.py --symbols BAC AAPL --day_start 2024-10-14 --day_stop 2024-10-18 &

# Running with force_remote set to True
python3 prepare_cache.py --symbols BAC AAPL --day_start 2024-10-14 --day_stop 2024-10-18 --force_remote &

```

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
## display plotly figs in single ntb cells

`figs2cell(figlist)`

Example usage:

```python
figs = []
fig1 = df.groupby([df['Exit Index'].dt.day_name(), 'Direction'])['PnL'].sum().unstack().vbt.barplot()
fig2 = df.groupby([df['Exit Index'].dt.day_name(), 'Direction'])['PnL'].sum().unstack().vbt.barplot()
figs.append(fig1)
figs.append(fig2)
display_figs_side_by_side(figs)
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
