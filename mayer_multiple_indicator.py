import streamlit as st
import numpy as np
import pandas as pd
# Matplotlib is replaced by Plotly for Streamlit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
def plot_timeseries_data(filepath):
    """
    Loads data, calculates indicators, and plots the result using Plotly.
    """

    st.write(f"Created by Gonçalo Duarte")
    st.info(f"Loading data for Mayer Multiple Z-Score from: {filepath}...")

    # --- Data Loading and Preprocessing (Largely Unchanged) ---
    try:
        df = pd.read_csv(filepath, thousands=',')
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please check the path.")
        return
    except Exception as e:
        st.error(f"An error occurred while reading the CSV: {e}")
        return

    date_column_name = 'Date'
    price_column_name = 'Price'

    if date_column_name not in df.columns or price_column_name not in df.columns:
        missing = [col for col in [date_column_name, price_column_name] if col not in df.columns]
        st.error(f"Error: Missing required columns: {', '.join(missing)}. Available columns: {df.columns.tolist()}")
        return

    try:
        df[date_column_name] = pd.to_datetime(df[date_column_name], format='%m/%d/%Y')
    except ValueError as e:
        st.error(f"Error parsing dates: {e}. Check if the date format is strictly 'MM/DD/YYYY'.")
        return

    df.set_index(date_column_name, inplace=True)
    price_data = df[price_column_name]
    price_data = pd.to_numeric(price_data, errors='coerce')
    price_data.dropna(inplace=True)

    # CRITICAL FIX: Reindex to fill calendar gaps
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
        vertical_spacing=0.1,
        row_heights=[3, 1],
        subplot_titles=(
            f'BTC/USD Price Time Series (Log Scale) and 200 SMA',
            "Mayer Multiple Z-Score"
        )
    )

    # 📊 --- Price Chart (Row 1) ---

    # 2. Add Price Data
    fig.add_trace(
        go.Scatter(
            x=price_data.index, y=price_data.values,
            mode='lines', name=price_column_name,
            line=dict(color='white', width=1),
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
    cmap = plt.cm.get_cmap('RdYlGn_r')
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
        range=[-4.5, 4.5], # Fixed range for better visualization of bands
        row=2, col=1,
        showgrid=True
    )

    # --- Global Layout Customization (Plotly Range Slider) ---
    fig.update_layout(
        title_text=f'**{price_column_name}** Mayer Multiple Z-Score Dashboard',
        height=820,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(b=100, t=100), # Adjust margins
        xaxis=dict(
            rangeslider=dict(
                visible=True, # This replaces your Matplotlib RangeSlider
                thickness=0.04,
            ),
            type="date",
            title_text="Date"
        ),
    )

    # 7. Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("💰 BTC Mayer Multiple Z-Score Dashboard")

    # Define the file path (must be accessible to the Streamlit app)
    # The user must upload the file or ensure it exists in the app's directory.
    file_to_plot = 'Bitcoin Historical Data_test.csv'

    # Streamlit file uploader for production use (optional, but good practice)
    # uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # For local testing, we'll use the hardcoded path
    plot_timeseries_data(file_to_plot)