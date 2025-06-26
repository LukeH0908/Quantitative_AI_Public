import datetime
from collections import deque, defaultdict
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from models import Conv2DMultiBinary

from indicators import ( LOCAL_FUNCS, AGAUSSIAN, PERCENTILE,
    HUGE, CONSTANT, parse_param_to_inputs, load_indicator_input, load_indicator_output
)

from av_client import av_client, process_REALTIME_BULK_QUOTES

import torch
from urllib.parse import urlencode, parse_qs
import copy


import matplotlib.pyplot as plt
import seaborn as sns



class RealTimeInferenceEngine:
    def __init__(self, tickers,  profit_model_path, accuracy_model_path, context_depth, params, device='cuda'):
        self.tickers = tickers
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # --- Load Both Models ---
        print("Loading dual models...")
        num_outputs = len(self.tickers)
        
        # Load Profit Model
        self.profit_model = Conv2DMultiBinary(in_channels=1, base_filters=32, num_outputs=num_outputs).to(self.device)
        self.profit_model.load_state_dict(torch.load(profit_model_path, map_location=self.device))
        self.profit_model.eval()
        print(f"✅ Profit Model loaded from {profit_model_path}")

        # Load Accuracy Model
        self.accuracy_model = Conv2DMultiBinary(in_channels=1, base_filters=32, num_outputs=num_outputs).to(self.device)
        self.accuracy_model.load_state_dict(torch.load(accuracy_model_path, map_location=self.device))
        self.accuracy_model.eval()
        print(f"✅ Accuracy Model loaded from {accuracy_model_path}")



        ochlv = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        self.context_depth = context_depth
        self.params  = sorted(params)
        self.hot_raw_ohlcv_data_buffer = {
            f"ticker={ticker}&function={func}": deque(maxlen=250)
            for ticker in self.tickers
            for func in ochlv
        }
        
        self.hot_params_tensor = {
            ticker: np.zeros((self.context_depth, len(self.params)), dtype=np.float32)
            for ticker in self.tickers

        } 


        self.param_to_idx = {param: i for i, param in enumerate(self.params)}
        
        print(f"Engine initialized for {len(self.tickers)} tickers.")


    def cold_start(self):
        print("--- Starting Cold Start via API ---")
        av = av_client()

        # This will hold the final, processed DataFrame for each ticker.
        raw_ohlcv_data = {}
        # you might run into some problems if the trading day occurs on the first of the month, in that case, you would just call the month before
        burn_in_period = 600
        current_date = datetime.datetime.now()
        current_month_yyyy_mm = current_date.strftime("%Y-%m")

        for ticker in self.tickers:

            print(f"Fetching historical data for {ticker}...")
            try:
                # 1. FETCH DATA using the API client
                # We fetch a full month to ensure we have enough data even after market holidays.
                series_data = av.fetch_intraday(
                    symbol=ticker, 
                    month=current_month_yyyy_mm, # You can make this dynamic if needed
                    interval="1min",
                    outputsize="full",
                    extended_hours=True
                )

                if not series_data:
                    print(f"⚠️ Warning: No data returned from API for {ticker}.")
                    continue
            
                # 2. PARSE AND CONVERT TO PANDAS DATAFRAME
                df = pd.DataFrame.from_dict(series_data, orient='index')
                df = df.rename(columns={
                    '1. open': 'open', '2. high': 'high', 
                    '3. low': 'low', '4. close': 'close', '5. volume': 'volume'
                })
                
                # Convert index to datetime objects and data to numbers
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)

                # 3. SORT DATA into ascending chronological order (oldest to newest)
                df = df.sort_index(ascending=True)
                df = df.resample('1min').ffill()

                df_recent = df.tail(burn_in_period)
                # In your cold_start function, after creating df_recent
                if not df_recent.empty:
                    print(f"Processed time range for {ticker}: {df_recent.index[0]} to {df_recent.index[-1]}")
                raw_ohlcv_data[ticker] = df_recent
                print(f"✅ Processed {len(df_recent)} data points for {ticker}.")

            except Exception as e:
                print(f"❌ Failed to process data for {ticker}. Error: {e}")


        all_indicator_series = {}
        params_by_ticker = defaultdict(list)
        for p_str in self.params:
            tkr = parse_qs(p_str).get("ticker", [None])[0]
            if tkr:
                params_by_ticker[tkr].append(p_str)

        for ticker, param_list in params_by_ticker.items():
            df_full = raw_ohlcv_data.get(ticker)
            if df_full is None or df_full.empty: continue
            
            for p_str in param_list:
                try:
                    input_dict, key = parse_param_to_inputs(p_str)
                    fn = LOCAL_FUNCS[key[0]]
                    result_series = fn(df_full, input_dict, key)
                    all_indicator_series[p_str] = result_series
                except Exception as e:
                    print(f"Warning: Failed to calculate {p_str} for {ticker}. Error: {e}")

        for ticker, df in raw_ohlcv_data.items():
            recent_data = df.tail(250) # Use the maxlen of your deque
            
            # Loop through each OHLCV component
            for component in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
                # Construct the unique key for the buffer
                buffer_key = f"ticker={ticker}&function={component.upper()}"
                
                # Check if this buffer should exist
                if buffer_key in self.hot_raw_ohlcv_data_buffer:
                    # Get the column name (e.g., 'open', 'high')
                    column_name = component.lower()
                    if column_name in recent_data.columns:
                        # Get the data from the DataFrame column
                        values = recent_data[column_name].tolist()
                        # Extend the deque with the historical values
                        self.hot_raw_ohlcv_data_buffer[buffer_key].extend(values)

        print("✅ Cold start complete. Tensors and buffers are ready.")


    def _calculate_param_slice(self, param_str: str, local_hot_raw_ohlcv_data_buffer: Dict[str, deque]) -> pd.Series:
        ticker = param_str.split("ticker=")[1].split("&")[0]
        
        # Reconstruct the DataFrame from the provided buffer
        reconstructed_data = {}
        components = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        for component in components:
            buffer_key = f"ticker={ticker}&function={component.upper()}"
            reconstructed_data[component.lower()] = list(local_hot_raw_ohlcv_data_buffer.get(buffer_key, []))
        
        df = pd.DataFrame(reconstructed_data)
        
        if df.empty:
            return pd.Series(dtype=np.float64) # Return empty series if no data

        # Calculate the specific indicator requested by param_str
        try:
            input_dict, key = parse_param_to_inputs(param_str)
            fn = LOCAL_FUNCS[key[0]]
            result_series = fn(df, input_dict, key)
            return result_series
        except Exception as e:
            # print(f"Could not calculate {param_str}: {e}")
            return pd.Series(dtype=np.float64)

    def update_hot_tensor_and_return_inference_item(self, new_quote_data: Dict[str, Dict], update: bool) -> np.ndarray:
        local_buffer = copy.deepcopy(self.hot_raw_ohlcv_data_buffer)

        # Step 2: Update the top layer of the *local* buffer with new quotes.
        for ticker, quote in new_quote_data.items():
            if ticker in self.tickers:
                for component in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
                    buffer_key = f"ticker={ticker}&function={component.upper()}"
                    if component.lower() in quote and buffer_key in local_buffer:
                        local_buffer[buffer_key].append(quote[component.lower()])


        if update:
            print("--- Starting Hot Update ---")
      
            # This correctly replaces the old buffer with the new one
            self.hot_raw_ohlcv_data_buffer = local_buffer 


        # Step 3: Loop through each parameter and calculate its full time-series slice.
        # print(f"Calculating new values for {len(self.params)} parameters...")
        all_new_indicator_values = {}
        for param_str in self.params:
            result_series = self._calculate_param_slice(param_str, local_buffer)

            unnormalized_series = result_series[-self.context_depth:]

            fn_name = param_str.split("function=")[1].split("&")[0] 
            last_value = unnormalized_series.iloc[-1]

            if fn_name in AGAUSSIAN:
                normalized_series = unnormalized_series - last_value
            elif fn_name in PERCENTILE:
                normalized_series = (unnormalized_series - last_value) / 100.0
            elif fn_name in CONSTANT:
                normalized_series = unnormalized_series - last_value
            else: # HUGE functions
                normalized_series = (unnormalized_series - last_value) / 10000000.0
            all_new_indicator_values[param_str] = normalized_series

        ### --- Step 4: Order Slices and Create Final NumPy Tensor --- ###
        ordered_series_list = []
        for param_str in self.params:
            series = all_new_indicator_values.get(param_str, pd.Series(dtype=np.float64))
            ordered_series_list.append(series)
            
        # Concatenate all series into a single DataFrame. `axis=1` makes them columns.
        inference_df = pd.concat(ordered_series_list, axis=1)
        
        # Set column names for clarity (optional but good practice)
        inference_df.columns = self.params
        
        # Convert the final DataFrame to a NumPy array. This is our inference item.
        inference_item = inference_df.values.astype(np.float32)
        return inference_item

    def infer(self, inference_item: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_tensor = torch.from_numpy(inference_item)
            x = x_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

            profit_logits = self.profit_model(x).squeeze()
            accuracy_logits = self.accuracy_model(x).squeeze()
            
            return profit_logits, accuracy_logits
        
    
    def visualize_inference_tensor(self, inference_item: np.ndarray, file_name: str):
        """
        Generates and saves a discrete heatmap of the feature tensor.

        Args:
            inference_item (np.ndarray): The (time_steps, num_features) tensor to plot.
            file_name (str): The name of the image file to save (e.g., 'cold_start_heatmap.png').
        """
        print(f"\nGenerating heatmap of the feature tensor, shape: {inference_item.shape}...")
        
        # We transpose the matrix so features are on the Y-axis and time is on the X-axis
        data_to_plot = inference_item.T  # Shape becomes (num_features, time_steps)

        # Set a dynamic figure size. Height depends on the number of features.
        # Adjust the numbers as needed for your screen/preference.
        height = max(10, len(self.params) / 4) # at least 10 inches tall
        width = 15
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Use seaborn to create the heatmap.
        # 'coolwarm' is great for data centered around zero, like your normalized features.
        sns.heatmap(data_to_plot, 
                    ax=ax,
                    yticklabels=self.params, # Use feature names as Y-axis labels
                    xticklabels=False,      # Hide the X-axis labels (too many time steps)
                    cmap='coolwarm',        # Color scheme
                    cbar=True)              # Show the color bar legend

        ax.set_title('Feature Tensor Heatmap at Cold Start', fontsize=16)
        ax.set_xlabel(f'Time Steps ({self.context_depth} mins)')
        ax.set_ylabel('Features')
        
        # Ensure labels aren't cut off
        plt.tight_layout()
        
        try:
            plt.savefig(file_name, dpi=150) # Save the plot to a file
            print(f"✅ Heatmap saved successfully to '{file_name}'")
        except Exception as e:
            print(f"❌ Failed to save heatmap. Error: {e}")
        
        plt.close() # Close the plot to free up memory
    # Add this new method inside your RealTimeInferenceEngine class

    def visualize_ohlcv_buffers(self, file_name: str = 'ohlcv_buffers_visualization.png'):
        """
        Generates and saves line plots of the raw OHLCV data stored in the
        engine's buffers after cold start.
        """
        print(f"\nGenerating visualization of the OHLCV data buffers...")

        num_tickers = len(self.tickers)
        if num_tickers == 0:
            print("No tickers to visualize.")
            return

        # Create a tall figure with one subplot for each ticker
        fig, axes = plt.subplots(
            nrows=num_tickers,
            ncols=1,
            figsize=(15, 6 * num_tickers), # 6 inches of height per ticker
            squeeze=False # Ensures axes is always a 2D array
        )
        axes = axes.flatten() # Flatten to a 1D array for easy iteration

        for i, ticker in enumerate(self.tickers):
            ax = axes[i]
            
            # --- Extract Data for this Ticker ---
            # Convert deques to lists for plotting
            try:
                close_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=CLOSE"])
                open_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=OPEN"])
                high_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=HIGH"])
                low_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=LOW"])
                volume_data = list(self.hot_raw_ohlcv_data_buffer[f"ticker={ticker}&function=VOLUME"])
            except KeyError as e:
                print(f"Data not found for {ticker} in buffer, skipping plot. Error: {e}")
                continue

            if not close_data:
                ax.set_title(f'{ticker} OHLCV Buffer Data - NO DATA FOUND', color='red')
                continue
                
            # --- Plot Price Data (OHLC) ---
            ax.plot(close_data, label='Close', color='blue', linewidth=2)
            ax.plot(open_data, label='Open', color='black', linestyle='--', alpha=0.7)
            ax.plot(high_data, label='High', color='green', linestyle=':', alpha=0.6)
            ax.plot(low_data, label='Low', color='red', linestyle=':', alpha=0.6)
            
            ax.set_title(f'{ticker} OHLCV Buffer Data', fontsize=16)
            ax.set_ylabel('Price ($)', color='blue')
            ax.grid(True, linestyle='--', alpha=0.6)

            # --- Plot Volume Data on a Second Y-Axis ---
            ax2 = ax.twinx() # Create a second y-axis that shares the same x-axis
            ax2.bar(range(len(volume_data)), volume_data, label='Volume', color='grey', alpha=0.2)
            ax2.set_ylabel('Volume', color='grey')
            
            # --- Create a single, combined legend for both axes ---
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')

        # Adjust layout to prevent titles and labels from overlapping
        plt.tight_layout(pad=3.0)
        
        try:
            plt.savefig(file_name, dpi=150)
            print(f"✅ OHLCV buffer visualization saved successfully to '{file_name}'")
        except Exception as e:
            print(f"❌ Failed to save OHLCV visualization. Error: {e}")
            
        plt.close() # Close the plot to free up memory



