
import vectorbtpro as vbt
import importlib
import os

def register_custom_inds(indicator_name: str = None, if_exists: str ="skip"):
    """Register a custom indicator or all custom indicators.

    Each indicator is as NAME.py file in the `ttools.indicators` directory. It should
    contain variable IND_{NAME} with the indicator factory object.

    If `indicator_name` is provided, only the indicator with that name is registered.
    Otherwise, all indicators found in the directory are registered - in each file
    variable with name starting with "IND_" .

    Argument `if_exists` can be "raise", "skip", or "override".
    """
    indicators_dir = os.path.join(os.path.dirname(__file__), "indicators")
    if indicator_name is not None:
        module_name = indicator_name
        file_path = os.path.join(indicators_dir, f"{module_name}.py")
        if os.path.exists(file_path):
            module = importlib.import_module(f"ttools.indicators.{module_name}")
            var_name = f"IND_{indicator_name}"
            var_value = getattr(module, var_name, None)
            if var_value is not None:
                vbt.IF.register_custom_indicator(var_value, location="ttools", if_exists=if_exists)
    else:
        for file_name in os.listdir(indicators_dir):
            if file_name.endswith(".py") and not file_name.startswith("_"):
                module_name = file_name[:-3]
                module = importlib.import_module(f"ttools.indicators.{module_name}")
                for var_name, var_value in vars(module).items():
                    if var_name.startswith("IND_"):
                        vbt.IF.register_custom_indicator(var_value, location="ttools", if_exists=if_exists)

def deregister_custom_inds(indicator_name: str = None):
    """Deregister a custom indicator or all custom indicators.

    If `indicator_name` is provided, only the indicator with that name is deregistered.
    Otherwise, all ttools indicators are deregistered.

    This function does not have an `if_exists` argument.
    """
    if indicator_name is not None:
            vbt.IF.deregister_custom_indicator(indicator_name, location="ttools")
    else:
            vbt.IF.deregister_custom_indicator(location="ttools")