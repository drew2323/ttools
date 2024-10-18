import numpy as np
from numba import jit
import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.utils.template import RepFunc

"""
Cumulative Anchored VWAP indicator on HLCC4 price, anchor = "D", "h", or "min" ...
drag = 0 - overlap with previous group. takes into account last N elements from previous group
when calculating (simulating v2realbot logic)
"""

def substitute_anchor(wrapper: ArrayWrapper, anchor: tp.Optional[tp.FrequencyLike]) -> tp.Array1d:
    """Substitute reset frequency by group lens. It is array of number of elements of each group."""
    if anchor is None:
        return np.array([wrapper.shape[0]])
    return wrapper.get_index_grouper(anchor).get_group_lens()

@jit(nopython=True)
def vwap_cum(high, low, close, volume, group_lens, drag):
    #anchor based grouping - prepare group indexes
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    #prepare output
    out = np.full(volume.shape, np.nan, dtype=np.float_)

    hlcc4 = (high + low + close + close) / 4

    #iterate over groups
    for group in range(len(group_lens)):
        from_i = group_start_idxs[group]
        from_i = max(0, from_i - drag)
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
cumulative anchored vwap indicator on HLCC4 price, anchor = "D", "h", or "min" ...
drag = 0 - overlap with previous group. takes into account last N elements from previous group
when calculating (simulating v2realbot logic)
"""
IND_CUVWAP = vbt.IF(
    class_name='CUVWAP',
    module_name='ttools',
    input_names=['high', 'low', 'close', 'volume'],
    param_names=['anchor', "drag"],
    output_names=['vwap']
).with_apply_func(vwap_cum,
                takes_1d=True,
                param_settings=dict(
                    anchor=dict(template=RepFunc(substitute_anchor)),
                ),
                anchor="D",
                drag=0
                )