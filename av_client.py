import os
from datetime import datetime, timedelta
import pytz
import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlencode
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List, Tuple, Union
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
        



def outer_context_loop(client, tuning_company, observing_companies):
    start_time = datetime.now()
    jobs = [
            (f"{tuning_company}_uncomparable_bars",    lambda: client.fetch_stock_data(symbol=tuning_company),    ["1. open","4. close","3. low","2. high","5. volume"]),
            (f"{tuning_company}_uncomparable_sma15H",  lambda: client.fetch_sma(tuning_company,15,"high"),  ["SMA"]),
            (f"{tuning_company}_uncomparable_sma60H",  lambda: client.fetch_sma(tuning_company,60,"high"),  ["SMA"]),
            (f"{tuning_company}_uncomparable_sma200H",  lambda: client.fetch_sma(tuning_company,200,"high"),  ["SMA"]),


            (f"{tuning_company}_uncomparable_sma15L",  lambda: client.fetch_sma(tuning_company,15,"low"),  ["SMA"]),
            (f"{tuning_company}_uncomparable_sma60L",  lambda: client.fetch_sma(tuning_company,60,"low"),  ["SMA"]),
            (f"{tuning_company}_uncomparable_sma200L",  lambda: client.fetch_sma(tuning_company,200,"low"),  ["SMA"]),
            

            (f"{tuning_company}_uncomparable_sma15C",  lambda: client.fetch_sma(tuning_company,15,"close"),  ["SMA"]),
            (f"{tuning_company}_uncomparable_sma60C",  lambda: client.fetch_sma(tuning_company,60,"close"),  ["SMA"]),
            (f"{tuning_company}_uncomparable_sma200C",  lambda: client.fetch_sma(tuning_company,200,"close"),  ["SMA"]),


            (f"{tuning_company}_uncomparable_vwap",   lambda: client.fetch_vwap(tuning_company),          ["VWAP"]),


            (f"{tuning_company}_uncomparable_bbands",   lambda: client.fetch_bbands_data(tuning_company,15,"close"),  ['Real Upper Band','Real Middle Band','Real Lower Band']),
            
        
            (f"{tuning_company}_uncomparable_sar",   lambda: client.fetch_sar_data(tuning_company, .01, .2),  ["SAR"]),
            
            (f"{tuning_company}_uncomparable_ht_tl_h",   lambda: client.fetch_ht_trendline_data(tuning_company,"high"), ['HT_TRENDLINE']),
            (f"{tuning_company}_uncomparable_ht_tl_l",   lambda: client.fetch_ht_trendline_data(tuning_company,"low"),  ['HT_TRENDLINE']),
            (f"{tuning_company}_uncomparable_ht_tl_c",   lambda: client.fetch_ht_trendline_data(tuning_company,"close"),  ['HT_TRENDLINE']),


            ]
            # Begin comparables
    print(len(observing_companies))
    for company in observing_companies:
        jobs.append(
        (f"{company}_comparable_natr",   lambda c=company: client.fetch_natr_data(c,15),  ["NATR"]),
        )    
        jobs.append(
            (f"{company}_comparable_macd", lambda c=company: client.fetch_macd(c, "close"), ["MACD", "MACD_Hist", "MACD_Signal"])
        )
        jobs.append(
        (f"{company}_comparable_stoch",   lambda c=company: client.fetch_stoch_data(c),  ['SlowD','SlowK'])
        )
        jobs.append(
        (f"{company}_comparable_rsi",   lambda c=company: client.fetch_rsi_data(c,15,"close"),  ["RSI"]),
        )
        jobs.append(
        (f"{company}_comparable_apo",   lambda c=company: client.fetch_apo_data(c,12,26,"high"),  ["APO"]),
        )
        jobs.append(
        (f"{company}_comparable_ppo",   lambda c=company:  client.fetch_ppo_data(c,12,26,"high"),  ["PPO"]),
        )
        jobs.append(
        (f"{company}_comparable_mom",   lambda c=company: client.fetch_mom_data(c,15,"high"),  ["MOM"]),
        )
        jobs.append(
        (f"{company}_comparable_cci",   lambda c=company: client.fetch_cci_data(c,15,"high"),  ["CCI"]),
        )
        jobs.append(
        (f"{company}_comparable_roc",   lambda c=company: client.fetch_roc_data(c,15,"high"),  ["ROC"]),
        )
        jobs.append(
        (f"{company}_comparable_rocr",   lambda c=company: client.fetch_rocr_data(c,15,"high"),  ["ROCR"]),
        )
        jobs.append(
        (f"{company}_comparable_mfi",   lambda c=company: client.fetch_mfi_data(c,15),  ["MFI"]),
        )
        jobs.append(
        (f"{company}_comparable_ultosc",   lambda c=company: client.fetch_ultosc_data(c,7,14,28),  ["ULTOSC"]),
        )
        jobs.append(
        (f"{company}_comparable_ht_sine_h",   lambda c=company:  client.fetch_ht_sine_data(c,"high"), ['SINE','LEAD SINE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_tm_h",   lambda c=company: client.fetch_ht_trendmode_data(c,"high"),  ['TRENDMODE']), 
        )
        jobs.append(
        (f"{company}_comparable_ht_dcper_h",   lambda c=company:  client.fetch_ht_dcperiod_data(c,"high"),  ['DCPERIOD']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcphase_h",   lambda c=company: client.fetch_ht_dcphase_data(c,"high"),  ['HT_DCPHASE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcphasor_h",   lambda c=company: client.fetch_ht_phasor_data(c,"high"),  ['PHASE','QUADRATURE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_sine_l",   lambda c=company:  client.fetch_ht_sine_data(c,"low"),  ['SINE','LEAD SINE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_tm_l",   lambda c=company:  client.fetch_ht_trendmode_data(c,"low"),  ['TRENDMODE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcper_l",   lambda c=company: client.fetch_ht_dcperiod_data(c,"low"),  ['DCPERIOD']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcphase_l",   lambda c=company: client.fetch_ht_dcphase_data(c,"low"),  ['HT_DCPHASE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcphasor_l",   lambda c=company: client.fetch_ht_phasor_data(c,"low"),  ['PHASE','QUADRATURE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_sine_c",   lambda c=company:  client.fetch_ht_sine_data(c,"close"),  ['SINE','LEAD SINE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_tm_c",   lambda c=company: client.fetch_ht_trendmode_data(c,"close"), ['TRENDMODE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcper_c",   lambda c=company: client.fetch_ht_dcperiod_data(c,"close"),  ['DCPERIOD']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcphase_c",   lambda c=company: client.fetch_ht_dcphase_data(c,"close"), ['HT_DCPHASE']),
        )
        jobs.append(
        (f"{company}_comparable_ht_dcphasor_c",   lambda c=company: client.fetch_ht_phasor_data(c,"close"),  ['PHASE','QUADRATURE']),
    )

        # # Print the total number of API calls
    print(f"Total API calls: {len(jobs)}")
    lookback = 10
    results = {}



    with ThreadPoolExecutor(max_workers=8) as executor:
        # submit in order, capturing futures in a list
        futures = [executor.submit(fn) for name, fn, _ in jobs]

        # now iterate futures in the same order you built them
        for i, (future, (name, _, _)) in enumerate(zip(futures, jobs)):
            try:
                results[name] = future.result()
                # print(f"Iteration: {i}")
            except Exception as e:
                print(f"Job {name} failed with exception: {e}")

    end_time = datetime.now()
    print(f"Outer loop took {end_time - start_time}")
    return jobs, results

def fetch_and_build_vectors(client, tuning_company, results, jobs, lookback=10):

  # 2) get current price & time
    now = datetime.now(pytz.utc)
    curr_price = client.fetch_stock_data(symbol=tuning_company)[0]
    day, hour, minute = now.weekday(), now.hour, now.minute

    # 3) build the feature vectors and index_map
    vectors  = []
    index_map = {}
    pos = 0

    for name, _, _ in jobs:
        raw = results.get(name, [])
        # 1) decide which part of `raw` to use
        try:
            if "_comparable_" in name:
                if "macd" in name:
                    data = raw[:3]
                elif "stoch" in name:
                    data = raw[:2]
                elif "sine" in name:
                    data = raw[:2]
                elif "dcphasor" in name:
                    data = raw[:2]
                else:
                    data = raw[:1]
            else:
                data = raw / curr_price
        except Exception as e:
            print(f"[ERROR] Failed to process {name} with exception: {e} → replacing with 0")
            data = 0

        # 2) turn into a 1‑D numpy array, padding empty→[1]
        arr = np.array(data).flatten()
        if arr.size == 0:
            print(f"[WARN] {name} returned no data → padding with 0.0")
            if "_uncomparable_" in name:
                if "_bars_" in name: 
                    arr =np.array([1]*50) 
                elif "bbands" in name:
                    arr = np.array([1]*30)
                else:
                    arr = np.array([1]*10)
            elif "_comparable_" in name: 
                if "macd" in name:
                    arr = np.array([1]*3)
                elif "stoch" in name:
                    arr = np.array([1]*2)
                elif "sine" in name:
                    arr = np.array([1]*2)
                elif "dcphasor" in name:
                    arr = np.array([1]*2)
                else:
                    arr = np.array([1])

        # 3) record
        length = arr.shape[0]
        vectors.append(arr)
        index_map[name] = (pos, pos + length)
        pos += length


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
            "timestamp": datetime.strptime(item.get("timestamp"), "%Y-%m-%d %H:%M:%S.%f"),
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


# TODO:

# def process_REALTIME_Intraday_QUOTES(av: av_client, ticker : str, extended_hours:bool) -> Dict[str, Dict]:
#     # Initialize the dictionary to store the final organized data
#     processed_data = {}

#     def fetch_intraday(self, symbol: str, month: str, interval: str = "1min", outputsize: str= "compact", entitlement: str ="delayed", extended_hours: str ="false") -> dict:
    
#     # This is your API call to get the raw data
#     raw_items = av.fetch_intraday(symbol=ticker, month: str, interval: str = "1min", outputsize: str= "compact", entitlement: str ="delayed", extended_hours: str ="false"))

#     # Check if the API returned any data
#     if not raw_items:
#         print("Warning: API call returned no items.")
#         return processed_data

#     # Loop through each item (quote) returned by the API
#     for item in raw_items:
#         # Safely get the symbol from the item dictionary
#         symbol = item.get("symbol")
#         quote_details = {
#             "symbol": symbol,
#             "timestamp": datetime.strptime(item.get("timestamp"), "%Y-%m-%d %H:%M:%S.%f"),
#             "open": float(item.get("open", 0.0)),
#             "high": float(item.get("high", 0.0)),
#             "low": float(item.get("low", 0.0)),
#             "close": float(item.get("close", 0.0)),
#             "volume": int(item.get("volume", 0)),
#             "previous_close": float(item.get("previous_close", 0.0)),
#             "change": float(item.get("change", 0.0)),
#             "change_percent": item.get("change_percent"),  # Keep as string (e.g., "1.5%")
#         }
        
#         # Add the cleaned data to our main dictionary, keyed by the symbol
#         processed_data[symbol] = quote_details

#     # print(f"Successfully processed data for {len(processed_data)} symbols.")
#     return processed_data




