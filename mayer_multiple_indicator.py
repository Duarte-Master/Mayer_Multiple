import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # fetch historical price data from Yahoo Finance

# Matplotlib is replaced by Plotly for Streamlit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Keep for color mapping logic

# --- Custom SMA Function (Unchanged) ---
def sma(series, period):
    """
    Calculates the Simple Moving Average (SMA) for a given series.
    Uses min_periods=period to return a value only when the full window is available.
    """
    # Use min_periods=period for standard indicator calculation.
    return series.rolling(window=period, min_periods=period).mean()
# --- End Custom SMA Function ---


# MODIFIED: Removed date filters as Plotly's rangeslider handles the view
@st.cache_data(show_spinner=False)
def fetch_price_data(start_date="2013-01-01"):
    """Fetches historical Bitcoin price data using Yahoo Finance via yfinance.

    The function uses Streamlit's cache to avoid repeated network calls. Data is
    trimmed to begin at **start_date** (default January 1st 2013). The returned
    DataFrame has a datetime index and a single column named **Price**.
    """
    try:
        # yfinance returns a DataFrame indexed by datetime
        df = yf.download("BTC-USD", start=start_date, interval="1d", progress=False)
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame(columns=["Price"])

    if df.empty:
        st.error("No data returned from Yahoo Finance.")
        return pd.DataFrame(columns=["Price"])

    # note: Yahoo Finance only has BTC history starting around September 2014; if
    # the caller requested an earlier start_date we cannot fulfil it. We'll still
    # return whatever data is available and display a message later in the app.

    # Use the closing price as the price series
    if "Close" not in df.columns:
        st.error("Expected 'Close' column in data from Yahoo Finance.")
        return pd.DataFrame(columns=["Price"])

    # when yfinance returns a MultiIndex column frame (Price, Ticker) as it does
    # when a single ticker is requested, selecting ['Close'] still yields a frame with
    # two levels.  We'll convert to a simple Series/DataFrame with one column named
    # 'Price' so the downstream code can index it normally.
    price_df = df[["Close"]].rename(columns={"Close": "Price"})
    price_df.index = pd.to_datetime(price_df.index)

    # flatten any multi‑level columns produced by yfinance
    if isinstance(price_df.columns, pd.MultiIndex):
        # keep only the first level ("Price") since ticker is redundant
        price_df.columns = price_df.columns.get_level_values(0)

    # Filter start date just in case
    price_df = price_df[price_df.index >= pd.to_datetime(start_date)]

    return price_df


def plot_timeseries_data(df=None):
    """
    Calculates indicators and plots the result using Plotly.

    If *df* is omitted the function will automatically fetch data via API.
    """

    st.write(f"Created by Gonçalo Duarte")

    # --- Data Loading and Preprocessing (Updated) ---
    if df is None:
        df = fetch_price_data()
    # protect against a None return value (caching glitch or network error)
    if df is None:
        st.error("Data fetch failed; received None from fetch_price_data.")
        return
    if df.empty:
        st.error("No data available to plot.")
        return

    # display the range of available data for user awareness
    st.write(f"Data from {df.index.min().date()} to {df.index.max().date()}")

    # Ensure expected columns exist
    if "Price" not in df.columns:
        st.error("Error: Dataframe must contain a 'Price' column.")
        return

    # Already indexed by date from fetch function; convert if necessary
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)

    price_data = pd.to_numeric(df["Price"], errors="coerce")
    price_data.dropna(inplace=True)

    # CRITICAL FIX: Reindex to fill calendar gaps in the dataset
    if not price_data.empty:
        start_date = price_data.index.min()
        end_date = price_data.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        price_data = price_data.reindex(full_date_range).ffill()
    price_data.dropna(inplace=True)
    # ----------------------------------------------------

    # 🚀 --- Mayer Multiple Z-Score Calculation (Unchanged) --- 🚀

    # Pine Script Inputs
    psma_length = 200
    overvalued = 2.0
    undervalue = 0.5

    # Calculations
    ma = sma(price_data, period=psma_length)
    multiple = price_data / ma

    top_bands = overvalued
    bottom_bands = undervalue

    mean = (top_bands + bottom_bands) / 2
    bands_range = top_bands - bottom_bands

    # Calculation of StdDev and Z-Score
    stdDev = bands_range / 6 if bands_range != 0 else 0

    if stdDev != 0:
        zScore = (multiple - mean) / stdDev
    else:
        zScore = pd.Series(0, index=multiple.index)

    zScore_valid = zScore.dropna()
    # multiple_valid = multiple.dropna() # Not needed for plotting

    # --- Plotting Setup using Plotly ---

    # 1. Setup figure with two subplots: Price (row 1) and Z-Score (row 2)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
        row_heights=[3, 1],
        subplot_titles=(
            f'BTC/USD Price Time Series (Log Scale) and 200 SMA',
            "Mayer Multiple Z-Score"
        )
    )

    # 📊 --- Price Chart (Row 1) ---

    # 2. Add Price Data (use a contrasting color & thicker line so it's always visible)
    fig.add_trace(
        go.Scatter(
            x=price_data.index, y=price_data.values,
            mode='lines', name='Price',
            line=dict(color='orange', width=2),
            # Plotly handles hover tooltips automatically
            hovertemplate='Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 3. Add 200-Day SMA
    if not ma.empty:
        latest_ma = ma.iloc[-1]
    else:
        latest_ma = np.nan

    fig.add_trace(
        go.Scatter(
            x=ma.index, y=ma.values,
            mode='lines', name=f'200 SMA: {latest_ma:.2f}',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>200 SMA: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Configure main plot aesthetics (Row 1)
    fig.update_yaxes(type="log", row=1, col=1, title_text="Price (USD) [Log Scale]", showgrid=True)


    # 📊 --- Z-Score Subplot (Row 2) ---

    # 4. Add Z-Score Line
    if not zScore_valid.empty:
        latest_z_score = zScore_valid.iloc[-1]
    else:
        latest_z_score = np.nan

    fig.add_trace(
        go.Scatter(
            x=zScore_valid.index, y=zScore_valid.values,
            mode='lines', name=f'Z-Score (Last): {latest_z_score:.2f}',
            line=dict(color='white', width=2),
            hovertemplate='Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # 5. Z-Score Horizontal Lines (using Plotly's HLine)

    # Neutral/Zero Line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

    # Positive Lines (Red/Overvalued) - Using RGBA to mimic Matplotlib's alpha
    fig.add_hline(y=1, line_dash="dash", line_color="rgba(255, 0, 0, 0.33)", line_width=1, row=2, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.66)", line_width=1, row=2, col=1)
    fig.add_hline(y=3, line_dash="dash", line_color="rgba(255, 0, 0, 1.00)", line_width=1, row=2, col=1)

    # Negative Lines (Green/Undervalued)
    fig.add_hline(y=-1, line_dash="dash", line_color="rgba(0, 255, 0, 0.33)", line_width=1, row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="rgba(0, 255, 0, 0.66)", line_width=1, row=2, col=1)
    fig.add_hline(y=-3, line_dash="dash", line_color="rgba(0, 255, 0, 1.00)", line_width=1, row=2, col=1)

    # 6. Color Bar Background (Approximation using Scatter and color mapping)
    # Get Colormap from Matplotlib and convert to a Plotly colorscale
    # Matplotlib 3.7+ deprecates plt.cm.get_cmap
    cmap = plt.colormaps.get('RdYlGn_r')
    norm = mcolors.Normalize(vmin=-3, vmax=3)
    # Create the Plotly color scale (list of [relative value, color])
    plotly_colorscale = []
    for i in np.linspace(0, 1, 11): # Sample 11 points for gradient
        rgb = cmap(i)[:3]
        hex_color = mcolors.to_hex(rgb)
        plotly_colorscale.append([i, hex_color])

    # Plotly Scatter for gradient coloring (similar to your Matplotlib scatter)
    if not zScore_valid.empty:
        fig.add_trace(
            go.Scatter(
                x=zScore_valid.index,
                y=zScore_valid.values,
                mode='markers',
                marker=dict(
                    color=zScore_valid.values,
                    colorscale=plotly_colorscale,
                    cmin=-3,
                    cmax=3,
                    size=3, # Smaller size for a line-like appearance
                    showscale=False
                ),
                name='Z-Score Gradient',
                showlegend=False
            ),
            row=2, col=1
        )

    # Z-Score Y-axis limits
    fig.update_yaxes(
        title_text="Z-Score",
        range=[-4.5, 6.0], # Fixed range for better visualization of bands
        row=2, col=1,
        showgrid=True
    )

    # --- Global Layout Customization (Plotly Range Slider) ---
    fig.update_layout(
        title_text=f'Bitcoin Mayer Multiple Z-Score Dashboard',
       height=820,
        template="plotly_dark", 
        hovermode="x unified", 
        
        # CRITICAL Change: increase bottom margin to push the subplot up.
        margin=dict(b=150), 
        
        # Rangeslider is applied to the shared X-axis (the bottom one)
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.04, # Keep thickness minimal
            ),
            type="date",
            title_text="Date"
        ),
        
        # We no longer need this x-axis filter since the Plotly slider controls the view.
        xaxis2=dict(
            range=None,
        )
    )

    # 7. Display the interactive plot in Streamlit
    # `use_container_width` is deprecated; use width='stretch' for full width
    st.plotly_chart(fig, width='stretch')


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Bitcoin Mayer Multiple Z-Score Dashboard")

    # Automatically fetch price history via API; no local CSV required.
    plot_timeseries_data()  # the function will call fetch_price_data internally
