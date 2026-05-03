import pandas as pd
import yfinance as yf
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from utils import (
    get_fx_rate, 
    get_price_local, get_history, 
    compute_moving_averages, plot_moving_averages,
    compute_volatility, plot_volatility,
    pe_ratio, plot_pe_ratios,
    beta_values, plot_betas,
    sharpe_ratio,
    rsi, plot_rsi,
    compute_macd, plot_macd
)

#Set Page title
st.set_page_config(page_title="Stock Analysis",
                   page_icon=":chart_with_upwards_trend:")

# Set Page header
st.title("📊 Financial Performance Analysis")

# Explain what the app does
st.markdown("This app allows you to upload a CSV with your financial assets and view agains, stock performance, and get AI-based recommendations.")

# Upload CSV file
st.markdown("### 📁 Upload your CSV file")
uploaded_file = st.file_uploader("Upload your portfolio CSV 📄", type=["csv"])

# Set up the connection with OpenAI
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if a file is uploaded
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Save the uploaded file as at and continue the analysis
if uploaded_file and st.session_state.data_loaded == False:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.data_loaded = True

# Flag to start analyzing / downloading CSV
if 'flag' not in st.session_state:
    st.session_state.flag = False

if st.session_state.data_loaded == True:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Gains 📈", "Stock updates 🛠️", "Export Data ⬇️", "Stock Analysis 🔎", "AI Recommendation ✨", "Investment Possibilities ➕"])

    ## Tab 1: Gains
    with tab1:
        # Cache the FX rates
        fx_cache = {}
        for fx in st.session_state.df['Currency Yahoo'].unique():
            if fx == "INR":
                fx_cache[fx] = 1.0
            else:
                fx_cache[fx] = get_fx_rate(fx)

        # Apply function
        st.session_state.df.apply(lambda row: get_price_local(row, fx_cache), axis=1)

        # Compute gains since the last update
        st.session_state.df["Price Today (INR)"] = st.session_state.df.apply(lambda row: get_price_local(row, fx_cache), axis=1)
        st.session_state.df["Value Today (INR)"] = st.session_state.df["Price Today (INR)"] * st.session_state.df["Units"]

        # Compute the Gains in INR since last update and Purchase
        st.session_state.df["Gain since Last Update (INR)"] = st.session_state.df["Value Today (INR)"] - st.session_state.df["Value Last Update"]
        st.session_state.df["Gain since Purchase (INR)"] = st.session_state.df["Value Today (INR)"] - st.session_state.df["Units"] * st.session_state.df["Purchase Price"]

        # Compute the Relative gains since last update and purchase
        st.session_state.df["Gain since Last Update (%)"] = st.session_state.df["Gain since Last Update (INR)"] / st.session_state.df["Value Last Update"] * 100
        st.session_state.df["Gain since Purchase (%)"] = st.session_state.df["Gain since Purchase (INR)"] / (st.session_state.df["Units"] * st.session_state.df["Purchase Price"]) * 100

        # Compute the totals
        total_value_last_update = st.session_state.df["Value Last Update"].sum()
        total_value_purchase = (st.session_state.df["Units"] * st.session_state.df["Purchase Price"]).sum()

        totals = {
            "Asset": "Total",
			"Ticker": "",
			"Gain since Last Update (INR)": st.session_state.df["Gain since Last Update (INR)"].sum(),
			"Gain since Last Update (%)": (st.session_state.df["Gain since Last Update (INR)"].sum() / total_value_last_update * 100) if total_value_last_update != 0 else 0,
			"Gain since Purchase (INR)": st.session_state.df["Gain since Purchase (INR)"].sum(),
			"Gain since Purchase (%)": (st.session_state.df["Gain since Purchase (INR)"].sum() / total_value_purchase * 100) if total_value_purchase != 0 else 0
        }

        # Columns to show
        columns = ["Asset", "Ticker", "Gain since Last Update (INR)", "Gain since Last Update (%)", "Gain since Purchase (INR)", "Gain since Purchase (%)"]
        report = pd.concat([st.session_state.df[columns], pd.DataFrame([totals])], ignore_index=True)

        # Display the report with just 2 float decimals
        today = date.today()
        st.markdown(f"### Snapshot of the Financial Performance {today} 📌")
        st.dataframe(report.style.format(precision=2))

		# If the data is loaded, set the flag to True
        if st.session_state.data_loaded == True:
            st.session_state.flag = True

    ## Tab 2: Stock updates
    with tab2:
        # Ask if there is any update
        update = st.radio("Do you have any update for your portfolio? ✏️", ("No", "Yes"), horizontal=True)

        if update == "Yes":
            st.markdown("### Stock Asset Details")
            selected_asset = st.selectbox("Select an asset to update", st.session_state.df["Asset"].tolist())
            changed_units = st.number_input("How many units were bought (2) /sold (-2): ",
                                            min_value=None,
                                            step=0.00001)
            new_purchase_price = st.number_input("What was the purchase price per unit (INR): ",
                                                min_value=None,
                                                step=0.01)

            if st.button("Update Asset"):
                if selected_asset and changed_units != 0 and new_purchase_price > 0:
                    # Update units average purchase price
                    idx = st.session_state.df[st.session_state.df["Asset"] == selected_asset].index[0]
                    old_units = st.session_state.df.at[idx, "Units"]
                    st.session_state.df.at[idx, "Units"] = old_units + changed_units

                    # Update the average purchasing price
                    old_purchase_price = st.session_state.df.at[idx, "Purchase Price"]
                    st.session_state.df.at[idx, "Purchase Price"] = (old_purchase_price * old_units + changed_units * new_purchase_price) / (old_units + changed_units)

                    st.success(f"Updated {selected_asset}: {old_units + changed_units} units @ {st.session_state.df.at[idx, 'Purchase Price']} ✅")
                else:
                    st.error("Please fill all fields with valid values. ⚠️")
        # Add new assets to the portfolio
        new_asset = st.radio("Did you add any new assets (not in the current portfolio)?",
                                  ("No", "Yes"),
                                  horizontal=True)
        # if yes
        if new_asset == "Yes":
            st.markdown("### Add New Asset Details ➕")

            # Input the fields
            asset_name = st.text_input("Asset name: ")
            ticker = st.text_input("Ticker: ")
            currency = st.selectbox("Currency Yahoo: ", ["INR", "USD"])
            units = st.number_input("Units: ", min_value=0.00001, step=0.00001)
            purchase_price = st.number_input("Purchase Price: ", min_value=0.0, step=0.01)
            if st.button("Add Asset"):
                if asset_name and ticker and currency and units > 0 and purchase_price > 0:
                    # Check if ticker is valid
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    if "shortName" in info:
                        # Create a new row
                        new_row = {
                            "Asset": asset_name,
                            "Ticker": ticker,
                            "Currency Yahoo": currency,
                            "Units": units,
                            "Purchase Price": purchase_price,
                            "Currency Purchase": "INR",
                            "Price Last Update": np.nan,
                            "Date Last Update": np.nan,
                            "Value Last Update": np.nan,
                            "Profit Last Update": np.nan,
                        }

                        # Append the row to session dataframe
                        st.session_state.df = pd.concat(
                            [st.session_state.df, pd.DataFrame([new_row])],
                            ignore_index=True
                        )

                        st.success(f"Added new asset: {asset_name} to portfolio. ✅")
                    else:
                        st.error("Ticker not found. ⚠️")
                else:
                    st.error("Please fill all fields with valid values. ⚠️")
        
        # If the analyze button is clicked, we can do the stock analysis / AI recommendations
        if st.button("Analyze 🔍"):
            st.session_state.flag = True

	## Tab 3: Export data
    with tab3:
        if st.session_state.flag == True:
            # Update the columns
            st.session_state.df["Price Last Update"] = st.session_state.df["Price Today (INR)"]
            st.session_state.df["Date Last Update"] = today

			# Show the selected data
            selected_data = st.session_state.df.iloc[:, :10]
            st.dataframe(selected_data.style.format(precision=2))

            # Isolating first 10 rows
            csv = selected_data.to_csv(index=False).encode('utf-8')

			# Export CSV
            st.download_button(
                label="Download updated CSV ⬇️",
                data=csv,
                file_name=f"assets_{today}.csv",
                mime="text/csv"
			)
            st.session_state.flag = True

	# Tab 4: Stock Analysis
    with tab4:
        # Create subtabs
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6, subtab7 = st.tabs([
            "Moving Averages", 
            "Volatility", 
            "P/E Ratio",
            "Beta",
            "Sharpe Ratio",
            "RSI",
            "MACD"
        ])

        if st.session_state.flag == True:
            ## Moving Averages
            with subtab1:
                st.markdown("""
                - **50-day Moving Average (MA50):** Short-term trend indicator. If the price is above MA50, it suggests short-term bullishness; below indicates short-term bearishness. 📈
                - **100-day Moving Average (MA100):** Medium-term trend indicator. Used to confirm trend direction and filter out short-term noise. 📊
                - **200-day Moving Average (MA200):** Long-term trend indicator. Price above MA200 is generally considered bullish; below is bearish. 🏦
                - **Crossovers:**
                    - When a shorter MA (e.g., MA50) crosses above a longer MA (e.g., MA200), it's called a "Golden Cross" and is a bullish signal. 🟢
                    - When a shorter MA crosses below a longer MA, it's called a "Death Cross" and is a bearish signal. 🔴
                - **Multiple Crosses:**
                    - If MA50 > MA100 > MA200, the trend is strongly bullish. 🚀
                    - If MA50 < MA100 < MA200, the trend is strongly bearish. 📉
                """)

                # Plot for all tickers in the portfolio
                for ticker in st.session_state.df["Ticker"]:
                    price_history, ma_data = compute_moving_averages(ticker)
                    st.markdown(f"**{ticker}:** Price: {ma_data['latest_price']:.2f} | MA50: {ma_data['ma50']:.2f} | MA100: {ma_data['ma100']:.2f} | MA200: {ma_data['ma200']:.2f}")
                    plot_moving_averages(price_history, ticker)

            ## Volatility
            with subtab2:
                st.markdown("""
                - **Volatility** measures how much a stock's price fluctuates. 📈
                - Higher volatility = higher risk and potential reward. ⚡
                - Lower volatility = more stable, less dramatic moves. 🛡️
                - Compare volatility across stocks to assess risk. 🔍
                - Use rolling volatility to spot changes in market behavior. 🔄
                """)

                # Compute the volatilities
                volatilities = {}
                for ticker in st.session_state.df["Ticker"]:
                    vol = compute_volatility(ticker)
                    if vol is not None: # Check if volatility was computed successfully
                        volatilities[ticker] = vol
                        st.markdown(f"**{ticker}:** Volatility: {vol:.1f}%")
                plot_volatility(volatilities)
            
            ## P/E Ratio
            with subtab3:
                st.markdown("""
                - **P/E Ratio** measures stock price relative to earnings. 📊
                - **High P/E (>25):** May be overvalued or high-growth expectations. ⚠️
                - **Low P/E (<15):** May be undervalued or poor growth prospects. 🔍
                - **Compare** P/E ratios within the same industry for context. 📈
                - **Watch out for:** Negative P/E (losses) or extremely high P/E (>50). 🚨
                """)

                # Compute the P/E ratios and plot
                pe_ratios = {}
                for ticker in st.session_state.df["Ticker"]:
                    pe = pe_ratio(ticker)
                    if pe > 0:
                        pe_ratios[ticker] = pe
                        st.markdown(f"**{ticker}:** P/E Ratio: {pe:.2f}")
                    else:
                        st.markdown(f"**{ticker}:** P/E Ratio: N/A (ETF/Crypto/Negative earnings)")
                plot_pe_ratios(pe_ratios)

            

            ## Beta
            with subtab4:
                st.markdown("""
                - **Beta** measures stock volatility vs market. 📊
                - **Beta > 1:** More volatile than market (higher risk/reward). ⚡
                - **Beta < 1:** Less volatile than market (lower risk/reward). 🛡️
                - **Beta = 1:** Moves with market average. 📈
                - **Watch out for:** Very high beta (>2) = extreme volatility. 🚨
                """)

                # Retrieve and show the beta for the portfolio
                betas = {}
                for ticker in st.session_state.df["Ticker"]:
                    beta = beta_values(ticker)
                    if not np.isnan(beta):
                        betas[ticker] = beta
                        st.markdown(f"**{ticker}:** Beta: {beta:.2f}")
                    else:
                        st.markdown(f"**{ticker}:** Beta: N/A")
                plot_betas(betas)
            
            ## Sharpe Ratio
            with subtab5:
                st.markdown("""
                - **Sharpe Ratio** measures risk-adjusted returns. 📊
                - **> 1.0:** Good risk-adjusted performance. ✅
                - **0.5-1.0:** Acceptable risk-adjusted performance. ⚠️
                - **< 0.5:** Poor risk-adjusted performance. ❌
                - **Watch out for:** Negative Sharpe = losing money on risk-adjusted basis. 🚨
                """)

                # Test the function with our portfolio
                for ticker in st.session_state.df['Ticker']:
                    sharpe = sharpe_ratio(ticker)
                    if sharpe > 1:
                        st.markdown(f"**{ticker}:** Sharpe Ratio: {sharpe:.1f} ✅ (Good risk-adjusted performance)")
                    elif sharpe > 0.5:
                        st.markdown(f"**{ticker}:** Sharpe Ratio: {sharpe:.1f} ⚠️ (Acceptable risk-adjusted performance)")
                    elif sharpe > 0:
                        st.markdown(f"**{ticker}:** Sharpe Ratio: {sharpe:.1f} ❌ (Poor risk-adjusted performance)")
                    else:
                        st.markdown(f"**{ticker}:** Sharpe Ratio: {sharpe:.1f} 🚨 (Negative return - avoid)")
            
            ## RSI
            with subtab6:
                st.markdown("""
                - **RSI** measures overbought/oversold conditions. 📊
                - **> 70:** Overbought (potential sell signal). 🔴
                - **< 30:** Oversold (potential buy signal). 🟢
                - **30-70:** Normal trading range. 📈
                - **Watch out for:** RSI staying >70 or <30 for extended periods. 🚨
                """)

                # Test the function to retrieve the RSI for the portfolio
                rsi_values = {}
                for ticker in st.session_state.df["Ticker"]:
                    rsi_val = rsi(ticker)
                    rsi_values[ticker] = rsi_val
                    if rsi_val > 70:
                        st.markdown(f"**{ticker}:** RSI: {rsi_val:.1f} 🔴 (Overbought)")
                    elif rsi_val < 30:
                        st.markdown(f"**{ticker}:** RSI: {rsi_val:.1f} 🟢 (Oversold)")
                    else:
                        st.markdown(f"**{ticker}:** RSI: {rsi_val:.1f} 📈 (Normal range)")
                plot_rsi(rsi_values)
                
            ## MACD
            with subtab7:
                st.markdown("""
                - **MACD** shows momentum and trend changes. 📊
                - **MACD > Signal:** Bullish momentum. 🟢
                - **MACD < Signal:** Bearish momentum. 🔴
                - **Crossovers:** Signal trend changes. 📈
                - **Watch out for:** MACD diverging from price action. 🚨
                """)

                # Test the function
                for ticker in st.session_state.df["Ticker"]:
                    price_history, macd_data = compute_macd(ticker)
                    status_emoji = "🟢" if macd_data["status"] == "Bullish" else "🔴" if macd_data["status"] == "Bearish" else "⚪"
                    st.markdown(f"**{ticker}:** MACD: {macd_data['macd']:.1f} | Signal: {macd_data['signal']:.1f} | Status: {macd_data['status']} {status_emoji}")
                    plot_macd(price_history, ticker)
    
    ## Tab 5: AI Recommendation
    with tab5:
        if st.session_state.flag == True:
            st.markdown("### AI Recommendation")
            with st.spinner("Analyzing....."):
                # Connect to the api
                llm = ChatOpenAI(
                    temperature=0,
                    model="gpt-5.4"
                )

                # Define the instruction
                instructions = """
                You are an expert portfolio analysis and personal finances.

                For every ticker:
                1) Summarise the key strengths and risks based on the KPIs provides.
                2) Flag Momentum Signals based on the KPIs
                3) Make a final recommendations (Buy, sell, hold) with a 1-2 sentence rationale

                Finish with a brief overall portfolio note.
                Answer in markdown
                """

                # Aggregate the data
                agg = {
                    "Ticker": st.session_state.df["Ticker"].tolist(),
                    "Asset": st.session_state.df["Asset"].tolist(),
                    "Price Today": st.session_state.df["Price Today (INR)"].tolist(),

                    # Moving Averages
                    "MA50": [get_history(t)[0]['Close'].rolling(50).mean().iloc[-1] for t in st.session_state.df["Ticker"]],
                    "MA100": [get_history(t)[0]['Close'].rolling(100).mean().iloc[-1] for t in st.session_state.df["Ticker"]],
                    "MA200": [get_history(t)[0]['Close'].rolling(200).mean().iloc[-1] for t in st.session_state.df["Ticker"]],

                    # Remaining KPIs
                    "Volatility": [compute_volatility(t) for t in st.session_state.df["Ticker"]],
                    "PE Ratio": [pe_ratio(t) for t in st.session_state.df["Ticker"]],
                    "Beta": [beta_values(t) for t in st.session_state.df["Ticker"]],
                    "Sharpe Ratio": [sharpe_ratio(t) for t in st.session_state.df["Ticker"]],
                    "RSI": [rsi(t) for t in st.session_state.df["Ticker"]],
                    "MACD": [compute_macd(t)[0]["MACD"].iloc[-1] for t in st.session_state.df["Ticker"]],
                    "Signal": [compute_macd(t)[0]["Signal"].iloc[-1] for t in st.session_state.df["Ticker"]]
                }

                # Transform to a Pandas Dataframe
                data = pd.DataFrame(agg)

                # Setup the messages
                messages = [
                    ("system", instructions),
                    ("human", data.to_string())
                ]

                # Calling the llm
                ai_recommendations = llm.invoke(messages)

            # Display the message and KPIs to the user
            st.write("Portfolio Summary 📋")
            st.dataframe(data)
            st.markdown(ai_recommendations.content)

    ## Tab 6: Investment Possibilities
    with tab6:
        if st.session_state.flag == True:
            new_tickers = st.text_input(
                label = "Enter New Ticker symbols",
                placeholder="e.g., NVDA, TSLA, AAPL (separate with commas) 📝"
            )
        
        # Include a button to submit
            if st.button("Submit"):
                with st.spinner("✨ Analyzing...."):
                    new_tickers_list = [ticker.strip() for ticker in new_tickers.split(",")]

                    # Build a new dataframe with all the data
                    agg_new_tickers = {
                        "Ticker": new_tickers_list,
                        "Asset": [get_history(t)[1]["longName"] for t in new_tickers_list],
                        "Price Today": [get_history(t)[0]["Close"].iloc[-1] for t in new_tickers_list],

                        # Moving averages
                        "MA50": [get_history(t)[0]['Close'].rolling(50).mean().iloc[-1] for t in new_tickers_list],
                        "MA100": [get_history(t)[0]['Close'].rolling(100).mean().iloc[-1] for t in new_tickers_list],
                        "MA200": [get_history(t)[0]['Close'].rolling(200).mean().iloc[-1] for t in new_tickers_list],

                        # Remaining KPIs
                        "Volatility": [compute_volatility(t) for t in new_tickers_list],
                        "PE Ratio": [pe_ratio(t) for t in new_tickers_list],
                        "Beta": [beta_values(t) for t in new_tickers_list],
                        "Sharpe Ratio": [sharpe_ratio(t) for t in new_tickers_list],
                        "RSI": [rsi(t) for t in new_tickers_list],
                        "MACD": [compute_macd(t)[0]["MACD"].iloc[-1] for t in new_tickers_list],
                        "Signal": [compute_macd(t)[0]["Signal"].iloc[-1] for t in new_tickers_list]
                    }

                    # Transform into a df
                    df_new_tickers = pd.DataFrame(agg_new_tickers)

                    # Tailor the human instructions
                    human_instructions = f"""
                    This is my current portfolio: {data.to_string()}
                    And these are the financial instruments I am considering: {df_new_tickers.to_string()}
                    Assess these new instruments based on my current portfolio and provide recommendations.
                    """

                    # Setup the messages
                    messages = [
                        ("system", instructions),
                        ("human", human_instructions)
                    ]

                    # Calling the LLM
                    ai_recommendation = llm.invoke(messages)

                    # Display the message to the user
                    st.markdown(ai_recommendation.content)
