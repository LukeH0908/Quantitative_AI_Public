import os
from dataclasses import dataclass, field

from urllib.parse import urlencode
import requests


from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List
import datetime
# ─── DB SETUP ──────────────────────────────────────────────────────────────────
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

@dataclass
class av_client:
    session: requests.Session = field(init=False)

    def __post_init__(self):
        self.session = requests.Session()


    def symbol_search(self, symbol: str) -> dict:
        """Call SYMBOL_SEARCH and return JSON."""
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        resp = self.session.get("https://www.alphavantage.co/query?", params=params)
        resp.raise_for_status()
        return resp.json().get("bestMatches", [])
    
    def _fetch(self, params):
        params["apikey"] = ALPHA_VANTAGE_API_KEY
        url = "https://www.alphavantage.co/query?" + urlencode(params)
        r = self.session.get(url)
        return r.json()

    def fetch_intraday(self, symbol: str, month: str, interval: str = "1min", outputsize: str= "compact", entitlement: str ="delayed", extended_hours: str ="false") -> dict:
        """
        month: 'YYYY-MM' string for Alpha Vantage month parameter
        """
        params = {
            "function":   "TIME_SERIES_INTRADAY",
            "symbol":     symbol,
            "month":      month,
            "interval":   interval,
            "outputsize": outputsize,
            "entitlement": entitlement,
            "extended_hours": extended_hours,
            "apikey":     ALPHA_VANTAGE_API_KEY,
        }
        r = self.session.get("https://www.alphavantage.co/query?", params=params)
        print(f"Accessing URL: {r.url}")
        r.raise_for_status()
        key = f"Time Series ({interval})"
        json_resp = r.json()
        if key not in json_resp:
            print(f"[ERROR] Could not find '{key}' in response. Response keys: {list(json_resp.keys())}")
        return json_resp.get(key, {})


    def fetch_REALTIME_BULK_QUOTES(self, symbols: str, extended_hours: bool) -> dict:
        """
        month: 'YYYY-MM' string for Alpha Vantage month parameter
        """
        params = {
            "function":   "REALTIME_BULK_QUOTES",
            "symbol":     symbols,
            "entitlement": "realtime",
            "apikey":     ALPHA_VANTAGE_API_KEY,
            "extended_hours": extended_hours,
        }
        r = self.session.get("https://www.alphavantage.co/query?", params=params)
        # print(f"Accessing URL: {r.url}")
        r.raise_for_status()
        return r.json().get(f"data", {})
    

    def fetch_indicator_naive(self, symbol, function, interval, month, time_period, series_type):
        params = {
            "function":      function,
            "symbol":        symbol,
            "interval":      interval,
            "time_period":   time_period,
            "series_type":   series_type,
            "outputsize":    "full",
            "month":         month,
            "apikey":       ALPHA_VANTAGE_API_KEY,
            "entitlement": "realtime",
            "extended_hours": "false",
        }
        r = self.session.get("https://www.alphavantage.co/query?", params=params)
        print("Request URL:", r.url)
        r.raise_for_status()
            # 2) Parse the JSON into a DataFrame
        key = f"Technical Analysis: {function}"
        raw = r.json().get(key, {})
        df = (
            pd.DataFrame.from_dict(raw, orient="index")
            .rename(columns={function: f"api_{function.lower()}"})
            .astype(float)
            .sort_index()
        )

        # 3) Convert the *string* index to EST datetime, floor it, then turn to UTC‐seconds
        est_dt = (
            pd.to_datetime(df.index, format="%Y-%m-%d %H:%M")
            .tz_localize("US/Eastern")
            .floor("min")
        )
        utc_seconds = (est_dt.tz_convert("UTC").view(int) // 10**9)

        # 4) Debug presence of your target timestamp
        if 1747425540 in utc_seconds:
            print("✅ API returned 2025‑05‑16 15:59:00 EST (uts=1747425540)")
        else:
            print("❌ API is missing 15:59 bar")

        # 5) Finally re‑index the DataFrame
        df.index = utc_seconds
        return df

    def fetch_indicator(self, input_dict, output_keys):
            params = {
                "function":      input_dict.get("function"),
                "symbol":        input_dict.get("symbol"),
                "interval":      input_dict.get("interval"),
                "time_period":   input_dict.get("time_period"),
                "series_type":   input_dict.get("series_type"),
                "fastlimit":   input_dict.get("fastlimit"),  # mama
                "slowlimit":   input_dict.get("slowlimit"), # mama
                "fastperiod":   input_dict.get("fastperiod"), #macd
                "slowperiod":   input_dict.get("slowperiod"), #macd
                "signalperiod":   input_dict.get("signalperiod"), #macd
                "fastmatype":   input_dict.get( "fastmatype"), #macdext {0...8} 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). 
                "slowmatype":   input_dict.get("slowmatype"), #macdext {0...8} 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). 
                "fastkperiod":   input_dict.get("fastkperiod"),
                "slowkperiod":   input_dict.get("slowkperiod"),

                "slowdperiod":   input_dict.get("slowdperiod"),


                "slowkmatype":   input_dict.get("slowkmatype"), #stoch {0...8} 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). 
                "fastdmatype":   input_dict.get("fastdmatype"), #stoch {0...8} 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA). 
                
                "timeperiod1":   input_dict.get( "timeperiod1"), #utlosc
                "timeperiod2":   input_dict.get("timeperiod2"), #utlosc
                "timeperiod3":   input_dict.get("timeperiod3"), #utlosc

                "nbdevup":   input_dict.get("nbdevup"), # bbands
                "nbdevdn":   input_dict.get("nbdevdn"), # bbands
                "matype":   input_dict.get("matype"), # bbands
                "acceleration":   input_dict.get("acceleration"), #sar
                "maximum":   input_dict.get("maximum"), #sar
                "outputsize":    "full",
                "month":         input_dict.get("month"),
                "apikey":        ALPHA_VANTAGE_API_KEY,
                "extended_hours":"false",
            }
            # Remove any keys where the value is None
            params = {k: v for k, v in params.items() if v is not None}

            r = self.session.get("https://www.alphavantage.co/query?", params=params)
            print("Request URL:", r.url)
            r.raise_for_status()
            fn = "Chaikin A/D" if input_dict.get("function") == "AD" else input_dict.get("function")
            key = f"Technical Analysis: {fn}"
            raw = r.json().get(key, {})

            result = {}
            for timestamp, values in raw.items():
                result[timestamp] = {k: float(values.get(k, float('nan'))) for k in output_keys}
            return result
        




def process_REALTIME_BULK_QUOTES(av: av_client, tickers: List[str], extended_hours:bool) -> Dict[str, Dict]:
    # Initialize the dictionary to store the final organized data
    processed_data = {}

    # print(f"Fetching bulk quotes for {len(tickers)} ")
    passed_arg = ",".join(tickers)
    print(passed_arg)
    # This is your API call to get the raw data
    raw_items = av.fetch_REALTIME_BULK_QUOTES(passed_arg, extended_hours)

    # Check if the API returned any data
    if not raw_items:
        print("Warning: API call returned no items.")
        return processed_data

    # Loop through each item (quote) returned by the API
    for item in raw_items:
        # Safely get the symbol from the item dictionary
        symbol = item.get("symbol")
        quote_details = {
            "symbol": symbol,
            "timestamp": datetime.datetime.strptime(item.get("timestamp"), "%Y-%m-%d %H:%M:%S.%f"),
            "open": float(item.get("open", 0.0)),
            "high": float(item.get("high", 0.0)),
            "low": float(item.get("low", 0.0)),
            "close": float(item.get("close", 0.0)),
            "volume": int(item.get("volume", 0)),
            "previous_close": float(item.get("previous_close", 0.0)),
            "change": float(item.get("change", 0.0)),
            "change_percent": item.get("change_percent"),  # Keep as string (e.g., "1.5%")
        }
        
        # Add the cleaned data to our main dictionary, keyed by the symbol
        processed_data[symbol] = quote_details

    # print(f"Successfully processed data for {len(processed_data)} symbols.")
    return processed_data




