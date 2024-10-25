
from dotenv import load_dotenv
from appdirs import user_data_dir
from ttools.utils import find_dotenv, AGG_TYPE, RecordType, StartBarAlign, zoneNY, zonePRG, zoneUTC
import os
import pytz

#Trade can be shared with v2realbot, agg cache not (we use df, but v2realbot uses Queue - will be changed in the future, when vectorized agg is added to v2realbot)
DATA_DIR = user_data_dir("v2realbot", False) # or any subfolder, if not sharing cache with v2realbot
LOCAL_TRADE_CACHE = DATA_DIR + "/tradecache:new/" # +daily_file
LOCAL_AGG_CACHE = DATA_DIR + "/aggcache_new/" #+ cache_file
RECTYPE = "BAR"
#AGG conditions -defaults
EXCLUDE_CONDITIONS = ['C','O','4','B','7','V','P','W','U','Z','F']
MINSIZE = 100
RECORD_TYPE = RecordType.BAR #loader supports only BAR type (no cbars)

#Load env variables
ENV_FILE = find_dotenv(__file__)
print(ENV_FILE)
if load_dotenv(ENV_FILE, verbose=True) is False:
    print(f"Error loading.env file {ENV_FILE}. Now depending on ENV VARIABLES set externally.")
else:
    print(f"Loaded env variables from file {ENV_FILE}")

#Alpaca accounts
ACCOUNT1_LIVE_API_KEY = os.getenv('ACCOUNT1_LIVE_API_KEY')
ACCOUNT1_LIVE_SECRET_KEY = os.getenv('ACCOUNT1_LIVE_SECRET_KEY')
ACCOUNT1_PAPER_API_KEY = os.getenv('ACCOUNT1_PAPER_API_KEY')
ACCOUNT1_PAPER_SECRET_KEY = os.getenv('ACCOUNT1_PAPER_SECRET_KEY')