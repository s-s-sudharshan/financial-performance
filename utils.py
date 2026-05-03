import pandas as pd
import yfinance as yf
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from langchain_openai import ChatOpenAI

# Function to retrieve the FX-Rate
def get_fx_rate(from_currency: str, to_currency: str = "EUR") -> float:
  """
  Get the live FX rate from Yahoo. Returns 1.0 if the same currency is used.
  """
  if from_currency == to_currency:
    return 1.0
  pair = f"{from_currency}{to_currency}=X"
  try:
    rate = yf.Ticker(pair).history(period="1d")
    return rate["Close"].iloc[-1]
  except Exception:
    return np.nan
  
# Build a function to fetch prices
def get_price_local(row, fx_cache=None) -> float:
  """
  Fetch the yahoo price in native currency and convert to local currency
  """
  try:
    price_native = (
        yf.Ticker(row['Ticker']).history(period = "1d")['Close'].iloc[-1]
    )
    
    # Use fx_cache if provided, otherwise get rate directly
    if fx_cache is not None and row['Currency Yahoo'] in fx_cache:
        fx_rate = fx_cache[row['Currency Yahoo']]
    else:
        fx_rate = get_fx_rate(row['Currency Yahoo'])
    
    return price_native * fx_rate
  except Exception as e:
    print(f"Error fetching {row['Ticker']}: {str(e)}")
    return np.nan
  
# Build a function that retrieves prices and info
def get_history(ticker, period = "1y", interval = "1d"):
  """ Fetch historical data for a ticker
  """
  try:
    price_history = yf.Ticker(ticker).history(period = period, interval = interval)
    info = yf.Ticker(ticker).info
    return price_history, info
  except Exception:
    print(f"Error fetching {ticker}")
    return None
  
# Compute the moving averages for the 50/100/200 days and display them
def compute_moving_averages(ticker):
  """
  Compute the moving averages for the 50/100/200 days and display them
  """
  # Compute the Moving averages
  price_history, _ = get_history(ticker)
  price_history["MA50"] = price_history["Close"].rolling(50).mean()
  price_history["MA100"] = price_history["Close"].rolling(100).mean()
  price_history["MA200"] = price_history["Close"].rolling(200).mean()

  # Retrieve the latest price and moving averages
  latest_price = price_history["Close"].iloc[-1]
  ma50 = price_history["MA50"].iloc[-1]
  ma100 = price_history["MA100"].iloc[-1]
  ma200 = price_history["MA200"].iloc[-1]

  # Return the results instead of displaying them
  return price_history, {
    "ticker": ticker,
    "latest_price": latest_price,
    "ma50": ma50,
    "ma100": ma100,
    "ma200": ma200
  }

# Build a function that plots the prices and moving averages
def plot_moving_averages(price_history, ticker):
  """
  Plot the prices and moving averages
  """
  plt.figure(figsize=(12, 4))
  plt.plot(price_history.index, price_history["Close"], label="Close Price")
  plt.plot(price_history.index, price_history["MA50"], label="MA50")
  plt.plot(price_history.index, price_history["MA100"], label="MA100")
  plt.plot(price_history.index, price_history["MA200"], label="MA200")

  # Customize the plots
  plt.title(f"{ticker} Price and Moving Averages")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  plt.grid(True)
  st.pyplot(plt.gcf())

# Build a function to compute the ticker volatility
def compute_volatility(ticker, window = 50):
  """
  Compute the volatility of a ticker
  """
  # Compute the volatility
  price_history, _ = get_history(ticker)
  returns = price_history["Close"].pct_change().dropna()
  # Compute the rolling volatility
  rolling_volatility = returns.rolling(window = 100).std() * 100 * np.sqrt(250)

  # Retrieve and Display the latest volatility
  latest_volatility = rolling_volatility.iloc[-1]
  return latest_volatility

# Build a function to plot the volatilities as a bar chart for my portfolio
def plot_volatility(volatilities):
  """
  Function that retrieves the volatitiles for all my portfolio tickers and plots them as a bar chart
  """
  # Plot them as a bar chart
  plt.figure(figsize=(10, 4))
  plt.bar(volatilities.keys(), volatilities.values(), color = "skyblue")
  plt.title("Portfolio Tickers Volatility")
  plt.xlabel("Ticker")
  plt.ylabel("Volatility (%)")
  plt.xticks(rotation=45)
  plt.grid(axis = "y", linestyle = "--", alpha = 0.6)
  st.pyplot(plt.gcf())

# Build a function to retrieve the P/E ratio for a ticker
def pe_ratio(ticker):
  """
  Retrieve the P/E ratio for a ticker
  """
  try:
    _, info = get_history(ticker)
    pe = info.get('trailingPE', np.nan)
    return pe
  except Exception:
    return np.nan
  
# Build a function to plot the PE ratios
def plot_pe_ratios(pe_ratios):
  """
  Plot the PE ratios for all tickers in the portfolio
  """
  # Plot the ratios
  plt.figure(figsize=(10, 4))
  plt.bar(pe_ratios.keys(), pe_ratios.values(), color = "red")
  plt.title("Portfolio Tickers PE Ratio")
  plt.xlabel("Ticker")
  plt.ylabel("PE Ratio")
  plt.xticks(rotation=45)
  plt.grid(axis = "y", linestyle = "--", alpha = 0.6)
  st.pyplot(plt.gcf())

# Defining a function to retrieve the Beta
def beta_values(ticker):
  """ Function that retrieves the beta of stocks, if available"""
  try:
    _, info = get_history(ticker)
    beta = info.get('beta', np.nan)
    return beta
  except Exception:
    return np.nan

# Build a function to plot the Betas
def plot_betas(betas):
  """ Function that blots with a bar chart the betas
  """

  # Plot the Bar chart
  plt.figure(figsize=(10, 4))
  plt.bar(betas.keys(), betas.values(), color = "green")
  plt.title("Portfolio Tickers Beta")
  plt.xlabel("Ticker")
  plt.ylabel("Beta")
  plt.xticks(rotation=45)
  plt.grid(axis = "y", linestyle = "--", alpha = 0.6)
  st.pyplot(plt.gcf())


# Build a function to compute the Sharpe Ratio
def sharpe_ratio(ticker, risk_free_rate = 0.04):
  """
  Compute the Sharpe Ratio for a ticker
  """
  # Compute the Sharpe Ratio
  price_history, _ = get_history(ticker)

  # Compute the daily returns
  returns = price_history["Close"].pct_change().dropna()

  # Compute the annualized return
  avg_daily_return = returns.mean()
  annualized_return = avg_daily_return * 252
  volatility = returns.std() * np.sqrt(252)

  # Compute the Sharpe Ratio
  sharpe = (annualized_return - risk_free_rate) / volatility

  return sharpe


# Build a function to compute the RSI for a ticker
def rsi(ticker, window = 14):
  """ Computes the RSI for a ticker. last date only"""
  price_history, _ = get_history(ticker)

  delta = price_history["Close"].diff()

  #See how many days with gains and how many with losses
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)

  # compute the rolling average gain and loss
  avg_gain = gain.rolling(window = window).mean()
  avg_loss = loss.rolling(window = window).mean()

  # Compute the RSI
  rs = avg_gain / avg_loss
  rsi = 100 - (100 / (1 + rs))

  return rsi.iloc[-1]



# Build a function to visualize the RSI of each portfolio Ticker
def plot_rsi(rsi_values):
  """ Plot the RSI for all tickers in the portfolio"""


  # Plot the RSI with a bar chart
  plt.figure(figsize=(10, 4))
  plt.bar(rsi_values.keys(), rsi_values.values(), color = "orange")
  plt.axhline(y = 30, color = "g", linestyle = "--", label = "Oversold")
  plt.axhline(y = 70, color = "r", linestyle = "--", label = "Overbought")
  plt.title("Portfolio Tickers RSI (14-day)")
  plt.xlabel("Ticker")
  plt.ylabel("RSI")
  plt.xticks(rotation=45)
  plt.grid(axis = "y", linestyle = "--", alpha = 0.6)
  plt.legend()
  st.pyplot(plt.gcf())

"""## MACD Crossover"""

# Build a function to compute the MACD
def compute_macd(ticker, short_window=12, long_window=26, signal_window=9):
  """
  Compute the MACD for a ticker
  """
  price_history, _ = get_history(ticker)
  close = price_history["Close"]

  # Compute the moving averages
  short_ema = close.ewm(span=short_window, adjust=False).mean()
  long_ema = close.ewm(span=long_window, adjust=False).mean()

  # Compute the MACD and Signal Lines
  price_history['MACD'] = short_ema - long_ema
  price_history['Signal'] = price_history['MACD'].ewm(span=signal_window, adjust=False).mean()

  # Latest Crossover
  macd_val = price_history['MACD'].iloc[-1]
  signal_val = price_history['Signal'].iloc[-1]
  if macd_val > signal_val:
    status = "Bullish"
  elif macd_val < signal_val:
    status = "Bearish"
  else:
    status = "Neutral"

  return price_history, {
    "ticker": ticker,
    "macd": macd_val,
    "signal": signal_val,
    "status": status
  }

# Build a function to visualize the MACD and Signal
def plot_macd(price_history, ticker):
  """
  Plot the MACD and Signal
  """
  plt.figure(figsize=(12, 4))
  plt.plot(price_history.index, price_history['MACD'], label='MACD', color='blue')
  plt.plot(price_history.index, price_history['Signal'], label='Signal', color='red')
  plt.axhline(y=0, color='black', linestyle='--')
  plt.title(f"{ticker} MACD and Signal")
  plt.xlabel("Date")
  plt.ylabel("MACD/Signal")
  plt.legend()
  plt.grid(True)
  st.pyplot(plt.gcf())
