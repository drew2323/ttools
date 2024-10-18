import numpy as np
from numba import jit
import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.utils.template import RepFunc

"""
Contains custom indicators for vectorbtpro.

import and run register_custom_inds() to register all custom indicators.

They are available under `vbt.IF.list_indicators("ttols")`
"""

def substitute_anchor(wrapper: ArrayWrapper, anchor: tp.Optional[tp.FrequencyLike]) -> tp.Array1d:
    """Substitute reset frequency by group lens. It is array of number of elements of each group."""
    if anchor is None:
        return np.array([wrapper.shape[0]])
    return wrapper.get_index_grouper(anchor).get_group_lens()

@jit(nopython=True)
def vwap_cum(high, low, close, volume, group_lens):
    #anchor based grouping - prepare group indexes
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    #prepare output
    out = np.full(volume.shape, np.nan, dtype=np.float_)

    hlcc4 = (high + low + close + close) / 4

    #iterate over groups
    for group in range(len(group_lens)):
        from_i = group_start_idxs[group]
        to_i = group_end_idxs[group]
        nom_cumsum = 0
        denum_cumsum = 0
        #for each group do this (it is just np.cumsum(hlcc4 * volume) / np.sum(volume) iteratively)
        for i in range(from_i, to_i):
            nom_cumsum += volume[i] * hlcc4[i]
            denum_cumsum += volume[i]
            if denum_cumsum == 0:
                out[i] = np.nan
            else:
                out[i] = nom_cumsum / denum_cumsum
    return out

"""
cumulative anchored vwap indicator on HLCC4 price
"""
IND_CUVWAP = vbt.IF(
    class_name='CUVWAP',
    module_name='ttools',
    input_names=['high', 'low', 'close', 'volume'],
    param_names=['anchor'],
    output_names=['vwap']
).with_apply_func(vwap_cum,
                takes_1d=True,
                param_settings=dict(
                    anchor=dict(template=RepFunc(substitute_anchor)),
                ),
                anchor="D",
                )

def register_custom_inds(indicator_name: str = None, if_exists: str ="skip"):
    """Register a custom indicator or all custom indicators.

    If `indicator_name` is provided, only the indicator with that name is registered.
    Otherwise, all indicators are registered - they are the ones starting with "IND_" .

    Argument `if_exists` can be "raise", "skip", or "override".
    """
    if indicator_name is not None:
        var_name = f"IND_{indicator_name}"
        var_value = globals().get(var_name)
        if var_value is not None and isinstance(var_value, vbt.IndicatorFactory):
            vbt.IF.register_custom_indicator(var_value, location="ttools", if_exists=if_exists)
        else:
            raise ValueError(f"Indicator '{indicator_name}' not found")
    else:
        for var_name, var_value in globals().items():
            if var_name.startswith("IND_") and isinstance(var_value, vbt.IndicatorFactory):
                vbt.IF.deregister_custom_indicator(var_value, location="ttools")

def deregister_custom_inds(indicator_name: str = None):
    """Deregister a custom indicator or all custom indicators.

    If `indicator_name` is provided, only the indicator with that name is registered.
    Otherwise, all indicators are registered - they are the ones starting with "IND_" .

    This function does not have an `if_exists` argument.
    """
    if indicator_name is not None:
        var_name = f"IND_{indicator_name}"
        var_value = globals().get(var_name)
        if var_value is not None and isinstance(var_value, vbt.IndicatorFactory):
            vbt.IF.deregister_custom_indicator(var_value, location="ttools")
        else:
            raise ValueError(f"Indicator '{indicator_name}' not found")
    else:
        for var_name, var_value in globals().items():
            if var_name.startswith("IND_") and isinstance(var_value, vbt.IndicatorFactory):
                vbt.IF.rdeegister_custom_indicator(var_value, location="ttools")