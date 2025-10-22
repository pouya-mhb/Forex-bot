# %%
!pip install yfinance

# %%
! pip install pandas numpy tensorflow scikit-learn matplotlib

# %%
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# %%

class YahooFeed:
    def __init__(self, symbol=" EURUSD=X"):
        self.symbol = symbol
        # from min to max
        self.periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"]
        self.intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m",
                          "1d", "5d", "1wk", "1mo", "3mo"]

        # find valid comibination
        self.period, self.interval = self._find_valid_combination()

    def _find_valid_combination(self):
        for it in self.intervals:      # first interval (from min to max)
            for p in self.periods:     # next period
                try:
                    print(f"Trying period={p}, interval={it} ...")
                    df = yf.download(tickers=self.symbol, period=p, interval=it, progress=False)
                    if not df.empty:
                        print(f"✅ Found valid: period={p}, interval={it}")
                        return p, it
                except Exception:
                    continue
        raise ValueError(f"No valid period/interval found for {self.symbol}")

    def fetch_candles(self, count=1500):
        df = yf.download(tickers=self.symbol, period=self.period, interval=self.interval, progress=False)
        df = df[['Open','High','Low','Close','Volume']].reset_index()
        df.rename(columns={'Open':'open','High':'high','Low':'low',
                           'Close':'close','Volume':'volume',
                           'Date':'datetime'}, inplace=True)
        return df.tail(count)

# %%

# ----------------------------
# Zone Detection
# ----------------------------
@dataclass
class Zone:
    type: str
    low: float
    high: float

def detect_zones(df, distance=0.0005):
    zones = []
    if len(df) < 3:
        return zones
    lows = df['low'].values
    highs = df['high'].values

    demand_condition = (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])
    supply_condition = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])

    demand_indices = np.where(demand_condition)[0] + 1
    supply_indices = np.where(supply_condition)[0] + 1

    for i in demand_indices:
        low = float(lows[i])
        high = low + distance
        zones.append(Zone("demand", low, high))

    for i in supply_indices:
        high = float(highs[i])
        low = high - distance
        zones.append(Zone("supply", low, high))

    return zones


# %%
# ----------------------------
# Signal Engine
# ----------------------------
@dataclass
class Signal:
    time: pd.Timestamp
    kind: str
    entry: float
    sl: float
    tp: float
    remark: str = ""

class SignalEngine:
    def generate(self, df, zones, predicted_close):
        signals = []
        if df.empty:
            return signals

        last_close = float(df['close'].iloc[-1])
        last_time = df['Datetime'].iloc[-1]  # using real datetime column

        for z in zones:
            low = float(z.low)
            high = float(z.high)
            if z.type == "demand" and last_close <= high:
                signals.append(Signal(
                    time=last_time,
                    kind="buy",
                    entry=last_close,
                    sl=last_close-0.001,
                    tp=last_close+0.002,
                    remark=f"Zone + AI Pred:{predicted_close:.5f}"
                ))
            elif z.type == "supply" and last_close >= low:
                signals.append(Signal(
                    time=last_time,
                    kind="sell",
                    entry=last_close,
                    sl=last_close+0.001,
                    tp=last_close-0.002,
                    remark=f"Zone + AI Pred:{predicted_close:.5f}"
                ))

        # سیگنال AI
        if predicted_close > last_close:
            signals.append(Signal(
                time=last_time,
                kind="buy",
                entry=last_close,
                sl=last_close-0.001,
                tp=last_close+0.002,
                remark="AI bullish"
            ))
        elif predicted_close < last_close:
            signals.append(Signal(
                time=last_time,
                kind="sell",
                entry=last_close,
                sl=last_close+0.001,
                tp=last_close-0.002,
                remark="AI bearish"
            ))

        return signals

# %%

# ----------------------------
#  fetching data and  Feature Engineering
# ----------------------------

feed = YahooFeed("EURUSD=X")
df = feed.fetch_candles()
print(df.head())

if df.empty:
    raise ValueError("Failed to fetch data. Check symbol, period, and interval.")

df['body'] = df['close'] - df['open']
df['upper_wick'] = df['high'] - np.maximum(df['close'], df['open'])
df['lower_wick'] = np.minimum(df['close'], df['open']) - df['low']

features = ['open','high','low','close','volume','body','upper_wick','lower_wick']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])


# %%
df.head()

# %%

# ----------------------------
# Detecting Zones
# ----------------------------
zones = detect_zones(df)

# Print first 5 zones
for z in zones[:5]:
    print(f"{z.type} zone | low: {z.low:.5f}, high: {z.high:.5f}")


# %%

# ----------------------------
# Preparing Data for LSTM
# ----------------------------
sequence_length = 10
X, y = [], []
for i in range(sequence_length, len(df)):
    X.append(df[features].iloc[i-sequence_length:i].values)
    y.append(df['close'].iloc[i])
X = np.array(X)
y = np.array(y)


# %%
# Splitting into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# %%
# Defining the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, len(features)), return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),

    LSTM(128, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),

    LSTM(64, return_sequences=False),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # predicting next candle price
])

# compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ----------------------------
# Callbacks
# ----------------------------

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ModelCheckpoint("best_lstm_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
]

# ----------------------------
# Training the model
# ----------------------------
history = model.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=callbacks
)

# %%

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss")
plt.legend()

# MAE
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE', color='green')
if 'val_mae' in history.history:
    plt.plot(history.history['val_mae'], label='Val MAE', color='orange')
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.title("Training & Validation MAE")
plt.legend()

plt.tight_layout()
plt.show()

# %%
# predicting on validation set
y_val_pred = model.predict(X_val, verbose=0).flatten()

# Visualizing the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(range(len(y_val)), y_val, label='Real Close', color='black')
plt.plot(range(len(y_val_pred)), y_val_pred, label='Predicted Close', color='red', alpha=0.7)
plt.title("Real vs Predicted Close on Validation Set")
plt.xlabel("Time Step")
plt.ylabel("Normalized Close Price")
plt.legend()
plt.show()


# %%
# ----------------------------
# Signal Generation
# ----------------------------
engine = SignalEngine()
history_signals = []

for i in range(sequence_length, len(df)):
    seq = df[features].iloc[i-sequence_length:i].values.reshape(1, sequence_length, len(features))
    predicted_close = float(model.predict(seq, verbose=0)[0][0])

    current_df = df.iloc[:i+1].copy()
    current_df.reset_index(drop=True, inplace=True)

    current_zones = detect_zones(current_df)
    signals = engine.generate(current_df, current_zones, predicted_close)

    for s in signals[:5]:
        history_signals.append(s)
        # print(f"[SIGNAL] {s.time} | {s.kind.upper()} | entry={s.entry:.5f} | SL={s.sl:.5f} | TP={s.tp:.5f} | {s.remark}")

# %%
def export_signals(signals, filename="signals_history.csv"):
    # Convert signals to DataFrame
    data = []
    for s in signals:
        data.append({
            "time": s.time,
            "type": s.kind,
            "entry": s.entry,
            "sl": s.sl,
            "tp": s.tp,
            "remark": s.remark
        })

    df_signals = pd.DataFrame(data)

    # Save to CSV
    df_signals.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✅ Signals exported to {filename}")

    return df_signals


# %%
signals_df = export_signals(history_signals, "signals_history.csv")

# %%
signals_df.head()

# %%
print("Sample signals times:", [s.time for s in history_signals[:5]])
print("DataFrame Datetime head:", df['Datetime'].head())

# %%
def backtest_signals(df, signals, look_ahead=20):
    """
    Backtest signals by comparing future candles.
    - df: DataFrame containing Datetime and OHLC columns
    - signals: List of Signal
    - look_ahead: Number of candles to check after the signal
    """
    # Ensure Datetime column is present
    if "Datetime" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
        else:
            df = df.reset_index().rename(columns={df.columns[0]: "Datetime"})

    # UTC-naive col datetime
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)

    results = []

    for s in signals:
        # Signal time
        sig_time = pd.to_datetime(s.time).tz_localize(None) if isinstance(s.time, pd.Timestamp) else pd.to_datetime(s.time)

        # Find signal index
        idx_list = df.index[df['Datetime'] == sig_time].tolist()
        if not idx_list:
            continue
        idx = idx_list[0]

        # Future data
        future = df.iloc[idx:idx+look_ahead]

        outcome = "Open"
        exit_price = s.entry

        for _, row in future.iterrows():
            high = float(row['high'])
            low = float(row['low'])

            if s.kind == "buy":
                if low <= s.sl:
                    outcome = "Loss"
                    exit_price = s.sl
                    break
                elif high >= s.tp:
                    outcome = "Win"
                    exit_price = s.tp
                    break
            elif s.kind == "sell":
                if high >= s.sl:
                    outcome = "Loss"
                    exit_price = s.sl
                    break
                elif low <= s.tp:
                    outcome = "Win"
                    exit_price = s.tp
                    break

        results.append({
            "Datetime": sig_time,
            "kind": s.kind,
            "entry": s.entry,
            "sl": s.sl,
            "tp": s.tp,
            "exit": exit_price,
            "outcome": outcome,
            "remark": s.remark
        })

    df_results = pd.DataFrame(results)

    # Calculate Win Rate
    if not df_results.empty:
        win_rate = (df_results['outcome'] == "Win").mean() * 100
        print(f"Total signals: {len(df_results)}, Win Rate: {win_rate:.2f}%")
    else:
        print("No backtest results generated. Check signals or DataFrame alignment.")

    return df_results


# %%
results_df = backtest_signals(df, history_signals)
results_df.head()


# %%
# Save backtest results to CSV
results_df.to_csv("backtest_results.csv", index=False, encoding="utf-8-sig")

print("✅ Backtest results saved to backtest_results.csv")

# %%
# Calculate profit/loss for each trade
def calculate_trade_profit_loss(row):
    if row['outcome'] == 'Win':
        # Assuming TP is higher than entry for buy, and lower for sell
        return abs(row['exit'] - row['entry'])
    elif row['outcome'] == 'Loss':
        # Assuming SL is lower than entry for buy, and higher for sell
        return -abs(row['exit'] - row['entry'])
    else:
        return 0

results_df['profit_loss'] = results_df.apply(calculate_trade_profit_loss, axis=1)

# Calculate cumulative profit/loss
results_df['cumulative_profit_loss'] = results_df['profit_loss'].cumsum()

# Plot cumulative profit/loss
plt.figure(figsize=(12, 6))
plt.plot(results_df['Datetime'], results_df['cumulative_profit_loss'])
plt.title('Cumulative Profit/Loss Over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Profit/Loss')
plt.grid(True)
plt.show()

# %%
# Calculate additional performance metrics

# Ensure 'profit_loss' column exists
if 'profit_loss' not in results_df.columns:
     def calculate_trade_profit_loss(row):
        if row['outcome'] == 'Win':
            return abs(row['exit'] - row['entry'])
        elif row['outcome'] == 'Loss':
            return -abs(row['exit'] - row['entry'])
        else:
            return 0
     results_df['profit_loss'] = results_df.apply(calculate_trade_profit_loss, axis=1)

# Ensure 'cumulative_profit_loss' column exists
if 'cumulative_profit_loss' not in results_df.columns:
    results_df['cumulative_profit_loss'] = results_df['profit_loss'].cumsum()

# Calculate Daily Returns (assuming each row is a "trade" or point in time)
# For simplicity, let's assume each row represents a period where the profit/loss occurred
# Calculate returns (profit/loss per trade)
returns = results_df['profit_loss']

# Sharpe Ratio (annualized)
if len(returns) > 1 and returns.std() != 0:
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.4f}")
else:
    print("Sharpe Ratio cannot be calculated (not enough data).")

# Equity curve
initial_capital = 1.0
equity_curve = initial_capital + results_df['cumulative_profit_loss']

# Maximum Drawdown
peak = equity_curve.expanding(min_periods=1).max()
drawdown = (equity_curve - peak) / peak
maximum_drawdown = drawdown.min()
print(f"Maximum Drawdown: {maximum_drawdown:.4f}")

# Calmar Ratio
total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
if abs(maximum_drawdown) > 1e-8:
    calmar_ratio = total_return / abs(maximum_drawdown)
    print(f"Calmar Ratio: {calmar_ratio:.4f}")
else:
    print("Calmar Ratio undefined (drawdown is zero).")
