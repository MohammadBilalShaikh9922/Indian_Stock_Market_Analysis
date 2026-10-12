import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Indian Stock Market Analysis", page_icon="📈", layout="wide")
st.title("📈 Indian Stock Market Analysis & Return Prediction")
st.caption("Historical market trends (2000–2020) • Ups/downs analysis • Next-day return estimate")
st.warning("Educational mini project only. Predictions are estimates, not financial advice or guaranteed future returns.")

@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower().replace("_", " ").strip(): c for c in df.columns}
    date_candidates = ["date", "datetime", "timestamp", "trading date"]
    close_candidates = ["close", "closing price", "adj close", "adjusted close", "price", "last"]
    volume_candidates = ["volume", "traded volume", "shares traded"]
    def find(candidates):
        for candidate in candidates:
            for key, original in lower_map.items():
                if key == candidate or candidate in key:
                    return original
        return None
    date_col = find(date_candidates)
    close_col = find(close_candidates)
    volume_col = find(volume_candidates)
    return df, date_col, close_col, volume_col

def prepare_data(raw):
    df, date_col, close_col, volume_col = normalize_columns(raw)
    if date_col is None or close_col is None:
        return None, f"Could not identify Date and Close columns. Available columns: {', '.join(map(str, raw.columns))}"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    df[close_col] = pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=[date_col, close_col]).sort_values(date_col)
    df = df[df[close_col] > 0].copy()
    if df.empty:
        return None, "No valid rows remain after parsing dates and prices."
    df = df.rename(columns={date_col: "Date", close_col: "Close"})
    if volume_col and volume_col in df.columns:
        df["Volume"] = pd.to_numeric(df[volume_col].astype(str).str.replace(",", ""), errors="coerce")
    return df, None

def make_features(df):
    d = df.copy().sort_values("Date")
    d["Return"] = d["Close"].pct_change()
    d["Return_1"] = d["Return"].shift(1)
    d["Return_2"] = d["Return"].shift(2)
    d["Return_3"] = d["Return"].shift(3)
    d["MA_5"] = d["Close"].rolling(5).mean()
    d["MA_20"] = d["Close"].rolling(20).mean()
    d["Volatility_5"] = d["Return"].rolling(5).std()
    d["Momentum_5"] = d["Close"] / d["Close"].shift(5) - 1
    d["Target_Return"] = d["Return"].shift(-1)
    features = ["Return_1", "Return_2", "Return_3", "MA_5", "MA_20", "Volatility_5", "Momentum_5"]
    if "Volume" in d.columns and d["Volume"].notna().sum() > 10:
        d["Volume_Change"] = d["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        features.append("Volume_Change")
    model_data = d.dropna(subset=features + ["Target_Return"]).replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["Target_Return"])
    return d, model_data, features

uploaded = st.sidebar.file_uploader("Upload the Kaggle CSV dataset", type=["csv"])
default_files = glob.glob(os.path.join("data", "*.csv"))
raw = None
if uploaded is not None:
    try:
        raw = load_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
elif default_files:
    try:
        raw = load_csv(default_files[0])
        st.sidebar.success(f"Loaded: {os.path.basename(default_files[0])}")
    except Exception:
        raw = None

if raw is None:
    st.info("Upload the CSV downloaded from Kaggle to begin. You can also place the CSV inside the project's `data` folder and restart the app.")
    st.markdown("""
### How to get the dataset
1. Open the [Kaggle Stock Data dataset](https://www.kaggle.com/datasets/sagara9595/stock-data).
2. Sign in to Kaggle and download the dataset ZIP.
3. Extract the ZIP and upload the relevant `.csv` file using the sidebar, or copy it into the project's `data` folder.
4. The CSV should contain a date column and a closing-price column (for example, `Date` and `Close`).
""")
    st.stop()

df, error = prepare_data(raw)
if error:
    st.error(error)
    st.write("Preview of uploaded data:")
    st.dataframe(raw.head())
    st.stop()

# Filter requested period when possible; keep available rows if dataset uses a different period.
df = df[(df["Date"] >= "2000-01-01") & (df["Date"] <= "2020-12-31")].copy()
if len(df) < 30:
    st.warning("Fewer than 30 records are available for 2000–2020. Showing all valid records instead.")
    df, error = prepare_data(raw)

st.sidebar.header("Analysis settings")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if len(df) < 10:
    st.error("Not enough rows in this date range. Choose a wider range.")
    st.stop()

df = df.sort_values("Date").reset_index(drop=True)
df, model_data, feature_cols = make_features(df)
returns = df["Return"].dropna()
latest_close = float(df["Close"].iloc[-1])
total_return = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100
best_day = df.loc[df["Return"].idxmax()] if returns.notna().any() else None
worst_day = df.loc[df["Return"].idxmin()] if returns.notna().any() else None
up_days = int((returns > 0).sum())
down_days = int((returns < 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest closing price", f"{latest_close:,.2f}")
c2.metric("Period price change", f"{total_return:+.2f}%")
c3.metric("Positive-return days", f"{up_days:,}")
c4.metric("Negative-return days", f"{down_days:,}")

st.subheader("1. Closing price trend")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Date"], df["Close"], linewidth=1.5)
ax.set_xlabel("Date"); ax.set_ylabel("Closing price"); ax.grid(True, alpha=0.25)
st.pyplot(fig)
plt.close(fig)

left, right = st.columns(2)
with left:
    st.subheader("2. Daily returns: ups and downs")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Date"], df["Return"] * 100, linewidth=0.7)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Daily return (%)"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.25)
    st.pyplot(fig); plt.close(fig)
with right:
    st.subheader("3. Return distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(returns.dropna() * 100, bins=40)
    ax.set_xlabel("Daily return (%)"); ax.set_ylabel("Frequency"); ax.grid(True, alpha=0.2)
    st.pyplot(fig); plt.close(fig)

st.subheader("4. Market summary")
summary = pd.DataFrame({
    "Measure": ["First closing price", "Latest closing price", "Total price change", "Average daily return", "Daily return volatility (std.)", "Best daily return", "Worst daily return", "Positive days", "Negative days"],
    "Value": [
        f"{df['Close'].iloc[0]:,.2f}", f"{df['Close'].iloc[-1]:,.2f}", f"{total_return:+.2f}%",
        f"{returns.mean()*100:+.4f}%", f"{returns.std()*100:.4f}%",
        f"{returns.max()*100:+.2f}%", f"{returns.min()*100:+.2f}%", str(up_days), str(down_days)
    ]
})
st.dataframe(summary, use_container_width=True, hide_index=True)
if best_day is not None and worst_day is not None:
    st.write(f"**Largest up day:** {best_day['Date'].date()} ({best_day['Return']*100:+.2f}%).")
    st.write(f"**Largest down day:** {worst_day['Date'].date()} ({worst_day['Return']*100:+.2f}%).")

st.subheader("5. Next-day return prediction (Random Forest)")
st.write("The model uses lagged returns, moving averages, volatility, momentum, and volume change when available. The data is split chronologically so the model is tested on later dates.")
if len(model_data) < 40:
    st.warning(f"Only {len(model_data)} usable training examples are available. A prediction may be unreliable; use a larger dataset.")
else:
    split = int(len(model_data) * 0.8)
    train, test = model_data.iloc[:split], model_data.iloc[split:]
    X_train, y_train = train[feature_cols], train["Target_Return"]
    X_test, y_test = test[feature_cols], test["Target_Return"]
    model = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", f"{mae*100:.4f} percentage points")
    m2.metric("Test RMSE", f"{rmse*100:.4f} percentage points")
    m3.metric("Test R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test["Date"], y_test.values * 100, label="Actual next-day return", linewidth=1)
    ax.plot(test["Date"], preds * 100, label="Predicted return", linewidth=1)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Date"); ax.set_ylabel("Return (%)"); ax.legend(); ax.grid(True, alpha=0.25)
    st.pyplot(fig); plt.close(fig)

    latest_features = df.dropna(subset=feature_cols).tail(1)
    if not latest_features.empty:
        predicted_return = float(model.predict(latest_features[feature_cols])[0])
        estimated_price = latest_close * (1 + predicted_return)
        p1, p2 = st.columns(2)
        p1.metric("Estimated next trading-day return", f"{predicted_return*100:+.3f}%")
        p2.metric("Illustrative next close estimate", f"{estimated_price:,.2f}")
        st.caption("This is a model output based on historical patterns, not a reliable guarantee of tomorrow's market movement.")
    st.write("**Feature importance**")
    imp = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1])
    ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.2)
    st.pyplot(fig); plt.close(fig)

st.subheader("6. Recent data")
st.dataframe(df[["Date", "Close"] + (["Volume"] if "Volume" in df.columns else []) + ["Return"]].tail(20), use_container_width=True, hide_index=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download analyzed data as CSV", data=csv, file_name="analyzed_stock_data.csv", mime="text/csv")

with st.expander("Project methodology"):
    st.markdown("""
- **Data preparation:** Parse dates, convert closing prices to numeric values, remove invalid rows, and sort chronologically.
- **Market analysis:** Closing-price trend, daily percentage returns, return distribution, volatility, positive/negative days, and largest daily movements.
- **Prediction target:** Next trading day's percentage return, calculated as `Close(t+1) / Close(t) - 1`.
- **Model:** Random Forest Regressor using lagged returns, moving averages, volatility, momentum, and optional volume change.
- **Evaluation:** Chronological 80/20 split; Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
- **Limitations:** Prices can be affected by news, macroeconomic changes, corporate actions, liquidity, and other factors absent from this simple model.
""")
