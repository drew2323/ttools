
from dotenv import load_dotenv
from appdirs import user_data_dir
from ttools.utils import find_dotenv
import os
import pytz
import vectorbtpro as vbt
import pytz
from pathlib import Path
from dotenv import load_dotenv
import os

ENV_FILE = find_dotenv()

#NALOADUJEME DOTENV ENV VARIABLES
if load_dotenv(ENV_FILE, verbose=True) is False:
    print(f"Error loading.env file {ENV_FILE}. Now depending on ENV VARIABLES set externally.")
else:
    print(f"Loaded env variables from file {ENV_FILE}")

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