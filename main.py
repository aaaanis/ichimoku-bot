import numpy as np
import pandas as pd
from pandas import json_normalize 
import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from mplfinance.original_flavor import candlestick2_ohlc
import matplotlib.pyplot as plt
import decimal
import requests
import mplfinance as mpf
import datetime


def get_data(ticker, timeframe):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
      'symbol': ticker,
      'interval': timeframe,
      'limit': 500
    }
    response = requests.get(url, params=params)
    data = response.json()

    return data


def get_current_price(ticker):
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={ticker}'
    response = requests.get(url)
    data = response.json()
    data = data['price']
    current_price = float(data)

    return current_price


def generate_ohlc_dataframe(data):
    df = pd.DataFrame(data)
    ohlc_df = df.copy().iloc[:,[0,1,2,3,4]]
    ohlc_df.rename(columns={0: 'Date', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close'}, inplace=True)
    ohlc_df = ohlc_df.astype(float)

    # Date in server timestamp format
    ohlc_df['Timestamp'] = ohlc_df['Date']

    # Date in gregorian calendar format
    ohlc_df['Date'] = ohlc_df['Date'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M'))
    ohlc_df['Date'] = pd.to_datetime(ohlc_df['Date'], format = '%Y-%m-%d')

    return ohlc_df


def compute_ichimoku_dataframe(ohlc_df):
    ichimoku_df = pd.DataFrame()
    high_prices = ohlc_df['High']
    close_prices = ohlc_df['Close']
    low_prices = ohlc_df['Low']

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    nine_period_high =  ohlc_df['High'].rolling(window=9).max()
    nine_period_low = ohlc_df['Low'].rolling(window=9).min()
    ichimoku_df['tenkan_sen'] = (nine_period_high + nine_period_low) /2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = high_prices.rolling(window=26).max()
    period26_low = low_prices.rolling(window=26).min()
    ichimoku_df['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    ichimoku_df['senkou_span_a'] = ((ichimoku_df['tenkan_sen'] + ichimoku_df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = high_prices.rolling(window=52).max()
    period52_low = low_prices.rolling(window=52).min()
    ichimoku_df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # The most current closing price plotted 26 time periods behind (optional)
    ichimoku_df['chikou_span'] = close_prices.shift(-26) 

    return ichimoku_df


# https://tradingstrategyguides.com/best-ichimoku-strategy/
def signal(ohlc_df, ichimoku_df, current_price):
    ichimoku_df['Low'] = ohlc_df['Low']
    ichimoku_df['Close'] = ohlc_df['Close']
    buy_signal_df = pd.DataFrame()
    sell_signal_df = pd.DataFrame()

    for index_line in range(26, len(ichimoku_df)):
        bullish_cross = [ichimoku_df['Close'][index_line] > ichimoku_df['senkou_span_a'][index_line],
            ichimoku_df['Close'][index_line] > ichimoku_df['senkou_span_b'][index_line], 
            ichimoku_df['tenkan_sen'][index_line-1] < ichimoku_df['kijun_sen'][index_line-1], 
            ichimoku_df['tenkan_sen'][index_line] > ichimoku_df['kijun_sen'][index_line]]
        cloud_twist = [ichimoku_df['senkou_span_a'][index_line-1] < ichimoku_df['senkou_span_b'][index_line-1],
            ichimoku_df['senkou_span_a'][index_line] > ichimoku_df['senkou_span_b'][index_line],
            ichimoku_df['Close'][index_line] > ichimoku_df['senkou_span_a'][index_line],
            ichimoku_df['tenkan_sen'][index_line] > ichimoku_df['kijun_sen'][index_line]]
        bearish_cross = [ichimoku_df['tenkan_sen'][index_line-1] > ichimoku_df['kijun_sen'][index_line-1],
            ichimoku_df['tenkan_sen'][index_line] < ichimoku_df['kijun_sen'][index_line], 
            ichimoku_df['Close'][index_line] < ichimoku_df['senkou_span_a'][index_line], 
            ichimoku_df['Close'][index_line] < ichimoku_df['senkou_span_b'][index_line],
            ichimoku_df['chikou_span'][index_line-26] < ichimoku_df['Close'][index_line-26]]
        take_profit = [ichimoku_df['tenkan_sen'][index_line-1] > ichimoku_df['kijun_sen'][index_line-1],
            ichimoku_df['tenkan_sen'][index_line] < ichimoku_df['kijun_sen'][index_line]]           
        if all(bullish_cross) or all(cloud_twist):
            buy_signal_df = buy_signal_df.append(pd.DataFrame({'Buy': index_line}, index=[0]), ignore_index=True) 
        if all(bearish_cross) or all(take_profit):
            sell_signal_df = sell_signal_df.append(pd.DataFrame({'Sell': index_line}, index=[0]), ignore_index=True)

    print(buy_signal_df.T)
    print(sell_signal_df.T)

    return [buy_signal_df, sell_signal_df]


def backtest(ticker, ohlc_df, signal_df):
    wallet = 50
    buy_signal_df = signal_df[0]
    sell_signal_df = signal_df[1]
    position = pd.DataFrame()

#    sell_signal_df['Sell'] = np.where(sell_signal_df['Sell'] <= buy_signal_df['Buy'], np.NaN, sell_signal_df['Sell'])
#    position = buy_signal_df.merge(sell_signal_df,left_on="A",right_on="G")
#    position['Position'] = pd.concat([buy_signal_df['Buy'], sell_signal_df['Sell']])
#    position['Stock_returns'] = np.log(ohlc_df['Open']) - np.log(ohlc_df['Open'].shift(1))
#    position['Strategy_returns'] = position['Stock_returns'] * position['Position']
#    print(position)

    return None


def broker(signal_df):
    buy_signal_df = signal_df[0]
    sell_signal_df = signal_df[1]
#    for i in range(len(buy_signal_df)):
    return None


def plot_signals(ohlc_df, signal_df):
    buy_signal_df = signal_df[0]
    sell_signal_df = signal_df[1]

#    for index in range(len(buy_signal_df)):
#        buy_signal_df = buy_signal_df.append(pd.Series({'Index': ohlc_df['Low'][index]}, index=[1]), ignore_index=True)
#        print(buy_signal_df)
    for buy_signals in range(1, len(buy_signal_df)):                                     
        plt.text(buy_signal_df['Buy'][buy_signals], ohlc_df['Low'][buy_signals], 'Buy', bbox=dict(facecolor='lightgreen', alpha=0.5), va = "top", ha="right")
    for sell_signals in range(1, len(sell_signal_df)):                                      
        plt.text(sell_signal_df['Sell'][sell_signals], ohlc_df['Low'][sell_signals], 'Sell', bbox=dict(facecolor='lightcoral', alpha=0.5), va = "top", ha="left")

    return None


def plot_ichimoku_cloud(ohlc_df, ichimoku_df, signal_df):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15, 7))

    ichimoku_df.drop(['Low', 'Close'], axis=1, inplace=True)
    candlestick2_ohlc(ax, ohlc_df['Open'], ohlc_df['High'], ohlc_df['Low'], ohlc_df['Close'], width=0.6, colorup='g', colordown='r')

    x_values = [str(date)[:len(":00.000000000")] for date in ohlc_df["Date"].values]

    formatter = mdates.DateFormatter("%Y-%m-%d hour: %H")
    locator = mdates.HourLocator(interval = 1)

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

    plt.xticks(ticks=range(0, len(x_values), 6), labels=x_values[::6], rotation=90)
    plot_signals(ohlc_df, signal_df)

    ax.plot(x_values, ichimoku_df)
    
    ax.fill_between(ichimoku_df.index, ichimoku_df.senkou_span_a, ichimoku_df.senkou_span_b,where = ichimoku_df.senkou_span_a >= ichimoku_df.senkou_span_b, color = 'lightgreen')
    ax.fill_between(ichimoku_df.index, ichimoku_df.senkou_span_a, ichimoku_df.senkou_span_b,where = ichimoku_df.senkou_span_a < ichimoku_df.senkou_span_b, color = 'lightcoral')

    plt.title('Ichimoku Cloud')
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.grid(True, linewidth = "0.3", linestyle = "dashed")
    plt.show()

    return None


def main():
    ticker = 'BTCUSDT'
    timeframe = '1h'
    data = get_data(ticker, timeframe)
    current_price = get_current_price(ticker)
    ohlc_df = generate_ohlc_dataframe(data)
    ichimoku_df = compute_ichimoku_dataframe(ohlc_df)
    signal_df = signal(ohlc_df, ichimoku_df, current_price)
    backtesting = backtest(ticker, ohlc_df, signal_df)
    broker_result = broker(signal_df)
    plot_ichimoku_cloud(ohlc_df, ichimoku_df, signal_df)


main()