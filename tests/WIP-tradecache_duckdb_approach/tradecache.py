#this goes to the main direcotry


from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Set, Dict, Tuple
import pandas as pd
import duckdb
import pandas_market_calendars as mcal
from abc import ABC, abstractmethod
import logging
from ttools.utils import zoneNY
from concurrent.futures import ThreadPoolExecutor
from ttools.loaders import fetch_daily_stock_trades
import time
logger = logging.getLogger(__name__)

class TradeCache:
    def __init__(
        self,
        base_path: Path,
        market: str = 'NYSE',
        max_workers: int = 4,
        cleanup_after_days: int = 7
    ):
        """
        Initialize TradeCache with monthly partitions and temp storage
        
        Args:
            base_path: Base directory for cache
            market: Market calendar to use
            max_workers: Max parallel fetches
            cleanup_after_days: Days after which to clean temp files
        """
        """Initialize TradeCache with the same parameters but optimized for the new schema"""
        self.base_path = Path(base_path)
        self.temp_path = self.base_path / "temp"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        self.calendar = mcal.get_calendar(market)
        self.max_workers = max_workers
        self.cleanup_after_days = cleanup_after_days
        
        # Initialize DuckDB with schema-specific optimizations
        self.con = duckdb.connect()
        self.con.execute("SET memory_limit='16GB'")
        self.con.execute("SET threads TO 8")
        
        # Create the schema for our tables
        self.schema = """
            x VARCHAR,
            p DOUBLE,
            s BIGINT,
            i BIGINT,
            c VARCHAR[],
            z VARCHAR,
            t TIMESTAMP WITH TIME ZONE
        """
        
        self._trading_days_cache: Dict[Tuple[date, date], List[date]] = {}
        
    def get_partition_path(self, symbol: str, year: int, month: int) -> Path:
        """Get path for a specific partition"""
        return self.base_path / f"symbol={symbol}/year={year}/month={month}"
    
    def get_temp_path(self, symbol: str, day: date) -> Path:
        """Get temporary file path for a day"""
        return self.temp_path / f"{symbol}_{day:%Y%m%d}.parquet"
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[date]:
        """Get trading days with caching"""
        key = (start_date.date(), end_date.date())
        if key not in self._trading_days_cache:
            schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
            self._trading_days_cache[key] = [d.date() for d in schedule.index]
        return self._trading_days_cache[key]
    
    def cleanup_temp_files(self):
        """Clean up old temp files"""
        cutoff = datetime.now() - timedelta(days=self.cleanup_after_days)
        for file in self.temp_path.glob("*.parquet"):
            try:
                # Extract date from filename
                date_str = file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                if file_date < cutoff:
                    file.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning up {file}: {e}")
    

    def consolidate_month(self, symbol: str, year: int, month: int) -> bool:
        """
        Consolidate daily files into monthly partition only if we have complete month
        Returns True if consolidation was successful
        """
        # Get all temp files for this symbol and month
        temp_files = list(self.temp_path.glob(f"{symbol}_{year:04d}{month:02d}*.parquet"))
        
        if not temp_files:
            return False
            
        try:
            # Get expected trading days for this month
            start_date = zoneNY.localize(datetime(year, month, 1))
            if month == 12:
                end_date = zoneNY.localize(datetime(year + 1, 1, 1)) - timedelta(days=1)
            else:
                end_date = zoneNY.localize(datetime(year, month + 1, 1)) - timedelta(days=1)
                
            trading_days = self.get_trading_days(start_date, end_date)
            
            # Check if we have data for all trading days
            temp_dates = set(datetime.strptime(f.stem.split('_')[1], '%Y%m%d').date() 
                            for f in temp_files)
            missing_days = set(trading_days) - temp_dates
            
            # Only consolidate if we have all trading days
            if missing_days:
                logger.info(f"Skipping consolidation for {symbol} {year}-{month}: "
                        f"missing {len(missing_days)} trading days")
                return False
                
            # Proceed with consolidation since we have complete month
            partition_path = self.get_partition_path(symbol, year, month)
            partition_path.mkdir(parents=True, exist_ok=True)
            file_path = partition_path / "data.parquet"
            
            files_str = ', '.join(f"'{f}'" for f in temp_files)
            
            # Modified query to handle the new schema
            self.con.execute(f"""
                COPY (
                    SELECT x, p, s, i, c, z, t
                    FROM read_parquet([{files_str}])
                    ORDER BY t
                )
                TO '{file_path}'
                (FORMAT PARQUET, COMPRESSION 'ZSTD')
            """)
            
            # Remove temp files only after successful write
            for f in temp_files:
                f.unlink()
                
            logger.info(f"Successfully consolidated {symbol} {year}-{month} "
                    f"({len(temp_files)} files)")
            return True
            
        except Exception as e:
            logger.error(f"Error consolidating {symbol} {year}-{month}: {e}")
            return False
    
    def fetch_remote_day(self, symbol: str, day: date) -> pd.DataFrame:
        """Implement this to fetch single day of data"""
        min_datetime = zoneNY.localize(datetime.combine(day, datetime.min.time()))
        max_datetime = zoneNY.localize(datetime.combine(day, datetime.max.time()))
        return fetch_daily_stock_trades(symbol, min_datetime, max_datetime)
    
    def _fetch_and_save_day(self, symbol: str, day: date) -> Optional[Path]:
        """Fetch and save a single day, returns file path if successful"""
        try:
            df_day = self.fetch_remote_day(symbol, day)
            if df_day.empty:
                return None
                
            temp_file = self.get_temp_path(symbol, day)
            df_day.to_parquet(temp_file, compression='ZSTD')
            return temp_file
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} for {day}: {e}")
            return None
    
    def load_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        columns: Optional[List[str]] = None,
        consolidate: bool = False
        ) -> pd.DataFrame:
        """Load data for date range, consolidating when complete months are detected"""
        #self.cleanup_temp_files()
        
        trading_days = self.get_trading_days(start_date, end_date)
        
        # Modify column selection for new schema
        col_str = '*' if not columns else ', '.join(columns)
        
        if consolidate:
            # First check temp files for complete months
            temp_files = list(self.temp_path.glob(f"{symbol}_*.parquet"))
            if temp_files:
                # Group temp files by month
                monthly_temps: Dict[Tuple[int, int], Set[date]] = {}
                for file in temp_files:
                    try:
                        # Extract date from filename
                        date_str = file.stem.split('_')[1]
                        file_date = datetime.strptime(date_str, '%Y%m%d').date()
                        key = (file_date.year, file_date.month)
                        if key not in monthly_temps:
                            monthly_temps[key] = set()
                        monthly_temps[key].add(file_date)
                    except Exception as e:
                        logger.warning(f"Error parsing temp file date {file}: {e}")
                        continue

                # Check each month for completeness and consolidate if complete
                for (year, month), dates in monthly_temps.items():
                    # Get trading days for this month
                    month_start = zoneNY.localize(datetime(year, month, 1))
                    if month == 12:
                        month_end = zoneNY.localize(datetime(year + 1, 1, 1)) - timedelta(days=1)
                    else:
                        month_end = zoneNY.localize(datetime(year, month + 1, 1)) - timedelta(days=1)
                    
                    month_trading_days = set(self.get_trading_days(month_start, month_end))
                    
                    # If we have all trading days for the month, consolidate
                    if month_trading_days.issubset(dates):
                        logger.info(f"Found complete month in temp files for {symbol} {year}-{month}")
                        self.consolidate_month(symbol, year, month)

        #timing the load
        time_start = time.time()
        print("Start loading data...", time_start)
        # Now load data from both consolidated and temp files
        query = f"""
            WITH monthly_data AS (
                SELECT {col_str}
                FROM read_parquet(
                    '{self.base_path}/*/*.parquet',
                    hive_partitioning=1,
                    union_by_name=true
                )
                WHERE t BETWEEN '{start_date}' AND '{end_date}'
            ),
            temp_data AS (
                SELECT {col_str}
                FROM read_parquet(
                    '{self.temp_path}/{symbol}_*.parquet',
                    union_by_name=true
                )
                WHERE t BETWEEN '{start_date}' AND '{end_date}'
            )
            SELECT * FROM (
                SELECT * FROM monthly_data
                UNION ALL
                SELECT * FROM temp_data
            )
            ORDER BY t
        """
        
        try:
            df_cached = self.con.execute(query).df()
        except Exception as e:
            logger.warning(f"Error reading cached data: {e}")
            df_cached = pd.DataFrame()
        
        print("fetched parquet", time_start - time.time())
        if not df_cached.empty:
            cached_days = set(df_cached['t'].dt.date)
            missing_days = [d for d in trading_days if d not in cached_days]
        else:
            missing_days = trading_days
        
        # Fetch missing days in parallel
        if missing_days:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_day = {
                    executor.submit(self._fetch_and_save_day, symbol, day): day 
                    for day in missing_days
                }
                
                for future in future_to_day:
                    day = future_to_day[future]
                    try:
                        temp_file = future.result()
                        if temp_file:
                            logger.debug(f"Successfully fetched {symbol} for {day}")
                    except Exception as e:
                        logger.error(f"Error processing {symbol} for {day}: {e}")
            
            # Check again for complete months after fetching new data
            temp_files = list(self.temp_path.glob(f"{symbol}_*.parquet"))
            if temp_files:
                monthly_temps = {}
                for file in temp_files:
                    try:
                        date_str = file.stem.split('_')[1]
                        file_date = datetime.strptime(date_str, '%Y%m%d').date()
                        key = (file_date.year, file_date.month)
                        if key not in monthly_temps:
                            monthly_temps[key] = set()
                        monthly_temps[key].add(file_date)
                    except Exception as e:
                        logger.warning(f"Error parsing temp file date {file}: {e}")
                        continue

                # Check for complete months again
                for (year, month), dates in monthly_temps.items():
                    month_start = zoneNY.localize(datetime(year, month, 1))
                    if month == 12:
                        month_end = zoneNY.localize(datetime(year + 1, 1, 1)) - timedelta(days=1)
                    else:
                        month_end = zoneNY.localize(datetime(year, month + 1, 1)) - timedelta(days=1)
                    
                    month_trading_days = set(self.get_trading_days(month_start, month_end))
                    
                    if month_trading_days.issubset(dates):
                        logger.info(f"Found complete month after fetching for {symbol} {year}-{month}")
                        self.consolidate_month(symbol, year, month)
            
            # Load final data including any new fetches
            try:
                df_cached = self.con.execute(query).df()
            except Exception as e:
                logger.warning(f"Error reading final data: {e}")
                df_cached = pd.DataFrame()
        
        return df_cached.sort_values('t')