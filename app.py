import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import time

from tensorflow.keras.models import load_model

# Page Background Color
# st.markdown("""
# <style>
# .stApp {
#     background-color: #F8FAFC;
# }
# </style>
# """, unsafe_allow_html=True)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="CryptoVision",
    page_icon="🪙",
    layout="wide"
)

# -------------------------------------------------
# Header
# -------------------------------------------------

st.markdown("""


<h1 style='text-align:center;color:#00b4d8;'>
CryptoVision 🪙
</h1>

<h4 style='text-align:center;'>
Cryptocurrency Price Prediction System
</h4>

</div>
""", unsafe_allow_html=True)

st.write("")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("Crypto Dashboard")

crypto = st.sidebar.selectbox(
    "Choose Cryptocurrency",
    ["Bitcoin", "Ethereum", "Litecoin"]
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

if crypto == "Bitcoin":
    ticker = "BTC-USD"
    model = load_model("models/bitcoin_model.keras")
    scaler = joblib.load("scalers/btc_scaler.pkl")

elif crypto == "Ethereum":
    ticker = "ETH-USD"
    model = load_model("models/ethereum_model.keras")
    scaler = joblib.load("scalers/eth_scaler.pkl")

else:
    ticker = "LTC-USD"
    model = load_model("models/litecoin_model.keras")
    scaler = joblib.load("scalers/ltc_scaler.pkl")

# -------------------------------------------------
# Download Latest Data
# -------------------------------------------------

df = yf.download(
    ticker,
    period="5y",
    progress=False
)

# Fix MultiIndex

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close = df[["Close"]]

latest_price = float(close.iloc[-1].values[0])

# -------------------------------------------------
# Predict Button
# -------------------------------------------------

if st.button("💲 Predict Next Day Price"):

    with st.spinner("Predicting price... Please wait ⏳"):
        time.sleep(2)    

    last_60 = close.tail(60)

    scaled = scaler.transform(last_60)

    X = scaled.reshape(1, 60, 1)

    prediction = model.predict(X, verbose=0)

    predicted_price = float(
        scaler.inverse_transform(prediction)[0][0]
    )

    difference = predicted_price - latest_price

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Price",
        f"${latest_price:.2f}"
    )

    col2.metric(
        "Predicted Price",
        f"${predicted_price:.2f}"
    )

    col3.metric(
        "Expected Change",
        f"${difference:.2f}",
        delta=f"{difference:.2f}"
    )

    st.success("Prediction Generated Successfully!")

# -------------------------------------------------
# Price Chart
# -------------------------------------------------

st.subheader("Closing Price Trend")

timeperiod = st.selectbox(
    "Select Time Period",
    ["1 Year", "2 Years", "3 Years", "4 Years", "5 Years"]
)

if (timeperiod == "1 Year"):
    df =yf.download(
        ticker,
        period="1y",
        progress=False
    )
elif (timeperiod == "2 Years"):
    df =yf.download(
        ticker,
        period="2y",
        progress=False
    )
elif (timeperiod == "3 Years"):
    df =yf.download(
        ticker,
        period="3y",
        progress=False
    )
elif (timeperiod == "4 Years"):
    df =yf.download(
        ticker,
        period="4y",
        progress=False
    )
else:
    df =yf.download(
        ticker,
        period="5y",
        progress=False
    )


st.line_chart(df["Close"])

# -------------------------------------------------
# Latest Data
# -------------------------------------------------

with st.expander("Latest Market Data"):

    st.dataframe(
        df.tail(5),
        use_container_width=True
    )
