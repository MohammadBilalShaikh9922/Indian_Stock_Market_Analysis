# Indian Stock Market Analysis & Return Prediction

A beginner-friendly Python mini project that analyzes historical Indian stock market data (target period: 2000–2020), identifies market ups and downs, and estimates the next trading day's return using a Random Forest regression model.

## Features
- Closing-price trend chart
- Daily percentage returns and return distribution
- Summary statistics, volatility, positive/negative days, best and worst day
- Next-day return prediction with a chronological train/test split
- MAE, RMSE and R² evaluation metrics
- Feature importance chart
- Downloadable analyzed CSV
- Streamlit web dashboard

## Dataset
Download the CSV from Kaggle:
https://www.kaggle.com/datasets/sagara9595/stock-data

You may need to sign in to Kaggle. Extract the downloaded archive. The app expects a CSV containing a date column (such as `Date`) and a closing-price column (such as `Close`, `Adj Close`, or `Price`). If the dataset has multiple CSV files, upload the appropriate one in the app.

## Run on Windows (VS Code / PowerShell)

1. Install Python 3.10 or newer.
2. Extract this project ZIP and open the folder in VS Code.
3. Open Terminal in VS Code and run:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

Using the explicit `venv\Scripts\python.exe` path avoids the PowerShell activation-policy issue. Streamlit will open a local browser tab.

## Add the dataset
- Simplest: use the file uploader in the left sidebar.
- Or copy a CSV file into the `data` folder, then restart the app.

The app attempts to focus on 2000-01-01 through 2020-12-31. If the uploaded dataset does not contain enough records in that period, it warns and uses the available valid records.

## Project structure

```text
Indian_Stock_Market_Analysis/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── Place_dataset_csv_here.txt
```

## Methodology
1. Clean and sort the data by date.
2. Calculate daily returns: `Return(t) = Close(t) / Close(t-1) - 1`.
3. Analyze trend, volatility, distribution and up/down days.
4. Build features from lagged returns, moving averages, volatility, momentum and optional volume change.
5. Predict the next day's return with `RandomForestRegressor`.
6. Test on the last 20% of dates to reduce look-ahead bias.

## Important limitations
This is an academic demonstration, not a trading system. Financial markets are noisy and historical patterns may not persist. A single stock's price history should not be interpreted as the entire Indian market. Predictions are not investment advice.
