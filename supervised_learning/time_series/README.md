# Time Series Forecasting - BTC Price Prediction

Uses past 24 hours of BTC data to predict the close price of the next hour.

## Data
- `coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv`
- `bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv`

## Usage
```
python3 preprocess_data.py
python3 forecast_btc.py
```

## Files
- `preprocess_data.py` — cleans and resamples raw 60s data to hourly windows
- `forecast_btc.py` — builds, trains, and validates an LSTM model using MSE loss
