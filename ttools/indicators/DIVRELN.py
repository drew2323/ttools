import numpy as np
from numba import jit
import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.utils.template import RepFunc

"""
DIVERGENCE - of two time series, same like in v2realbot

    if mode == "abs":
        val =  round(abs(float(source1_series[-1]) - float(source2_series[-1])),4)
    elif mode == "absn":
        val =  round((abs(float(source1_series[-1]) - float(source2_series[-1])))/float(source1_series[-1]),4)
    elif mode == "rel":
        val =  round(float(source1_series[-1]) - float(source2_series[-1]),4)
    elif mode == "reln": #div = a+b   /   a-b  will give value between -1 and 1
        val =  round((float(source1_series[-1]) - float(source2_series[-1]))/(float(source1_series[-1])+float(source2_series[-1])),4)
    elif mode == "pctabs":
        val = pct_diff(num1=float(source1_series[-1]),num2=float(source2_series[-1]), absolute=True)
    elif mode == "pct":
        val = pct_diff(num1=float(source1_series[-1]),num2=float(source2_series[-1]))
"""


@jit(nopython=True)
def divergence(series1, series2, divtype):
        print(divtype)
        #div = a+b   /   a-b  will give value between -1 and 1
        if divtype == "reln":
            return (series1 - series2) / (series1 + series2)
        elif divtype == "rel":
            return series1 - series2
        elif divtype == "abs":
            return np.abs(series1 - series2)
        elif divtype == "absn":
            return np.abs(series1 - series2) / series1
        elif divtype == "pctabs":
            return np.abs(((series1 - series2) / series1) * 100)
        elif divtype == "pct":
            return ((series1 - series2) / series1) * 100
        else:
            return np.full_like(series1, np.nan)

"""
Divergence indicator - various divergences between two series
"""
IND_DIVRELN = vbt.IF(
    class_name='DIVRELN',
    module_name='ttools',
    input_names=['series1', 'series2'],
    param_names=["divtype"],
    output_names=['div']
).with_apply_func(divergence,
                takes_1d=True,
                param_settings=dict(
                ),
                divtype="reln"
                )