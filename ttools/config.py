
from dotenv import load_dotenv
from appdirs import user_data_dir
import ttools.utils as utils
import os
import pytz
from pathlib import Path
from dotenv import load_dotenv
def find_dotenv():
    """
    Searches for a .env file in the given directory or its parents and returns the path.

    Args:
        start_path: The directory to start searching from.

    Returns:
        Path to the .env file if found, otherwise None.
    """
    try:
        start_path = __file__
    except NameError:
        #print("Notebook probably")
        start_path = os.getcwd()  
        #print(start_path)       

    current_path = Path(start_path)
    for _ in range(10):  # Limit search depth to 5 levels
        dotenv_path = current_path / '.env'
        if dotenv_path.exists():
            return dotenv_path
        current_path = current_path.parent
    return None

ENV_FILE = find_dotenv()

#NALOADUJEME DOTENV ENV VARIABLES
if load_dotenv(ENV_FILE, verbose=True) is False:
    print(f"TTOOLS: Error loading.env file {ENV_FILE}. Now depending on ENV VARIABLES set externally.")
else:
    print(f"TTOOLS: Loaded env variables from file {ENV_FILE}")

ACCOUNT1_LIVE_API_KEY = os.environ.get('ACCOUNT1_LIVE_API_KEY')
ACCOUNT1_LIVE_SECRET_KEY = os.environ.get('ACCOUNT1_LIVE_SECRET_KEY')
DATA_DIR_NAME = os.environ.get('DATA_DIR_NAME', "ttools") #folder in datadir

DATA_DIR = user_data_dir(DATA_DIR_NAME, False)
TRADE_CACHE = Path(DATA_DIR)/"tradecache"
AGG_CACHE = Path(DATA_DIR)/"aggcache"
zoneNY = pytz.timezone('US/Eastern')

#AGG conditions -defaults
EXCLUDE_CONDITIONS = ['C','O','4','B','7','V','P','W','U','Z','F','9','M','6']
#added 9 - correction, M - Official Close, T- extended hours, 6 - Cancel Trade
MINSIZE = 100